#!/usr/bin/env python3
"""Accuracy evaluation — perplexity and MMLU, for any checkpoint tier.

    # fast shape check (seconds to a minute)
    python scripts/07_eval_accuracy.py --tier bf16 --smoke

    # full evaluation per measurement_spec.md (decision 12)
    python scripts/07_eval_accuracy.py --tier bf16

Implements measurement_spec.md decisions 11-13:

  * Primary metric: perplexity on a fixed slice of wikitext-103 test text.
  * Secondary metric: MMLU subset, log-likelihood scoring, N = 500.
  * Both reported as absolute values, so later tiers can be differenced against
    the BF16 oracle measured in the same session and with the same evaluator.

Two deliberate design choices worth understanding:

  1. lm-eval-harness is NOT used. It would add a dependency that could perturb
     the pinned environment (transformers is nailed to 4.57.3 by tensorrt-llm),
     and it would hide the scoring convention. The convention is written out
     below instead, and it is the one that matters for comparing our own tiers.
  2. Accuracy is measured from a CHECKPOINT, never from an engine. An engine is
     verified separately by token agreement (decision 14). Mixing the two would
     make the accuracy column depend on which runtime happened to be measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = Path("/home/jiming/models/Qwen3-8B")
HF_REF = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-8B/refs/main"
RESULTS_DIR = REPO / "results" / "accuracy"

# ---- protocol constants [SPEC 11, 12] -------------------------------------
PPL_DATASET = ("Salesforce/wikitext", "wikitext-103-raw-v1", "test")
PPL_WINDOW = 2048          # tokens per perplexity window
PPL_TOKENS = 32768         # total token budget, fixed for comparability
MMLU_DATASET = ("cais/mmlu", "all", "test")
MMLU_N = 500               # [SPEC 12]
SEED = 0
CHOICE_LETTERS = ("A", "B", "C", "D")

# MMLU scoring convention, stated because it is a choice, not a fact:
#   prompt   = chat template applied to the question, WITH the generation prompt,
#              so the model sees the same context it would see when answering
#   candidate= the single token sequence for " A", " B", " C", " D"
#   score    = sum of log-probabilities of the candidate tokens, no length
#              normalisation (this is lm-eval-harness's `acc`, not `acc_norm`)
#   prediction = argmax over the four candidates
# Changing any of these changes the number, so they are recorded in the output.
MMLU_CONVENTION = "chat_template_with_generation_prompt; sum_logprob_no_normalisation"


# ==========================================================================
# Provenance
# ==========================================================================
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def package_version(name: str) -> str:
    import importlib.metadata as md
    try:
        return md.version(name)
    except Exception:
        return "absent"


def fingerprint(model_dir: Path) -> dict:
    fp = {
        "torch": torch.__version__,
        "cuda_rt": torch.version.cuda,
        "transformers": package_version("transformers"),
        "datasets": package_version("datasets"),
        "nvidia_modelopt": package_version("nvidia-modelopt"),
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": "sm_%d%d" % torch.cuda.get_device_capability(0),
        "pip_freeze_sha256": hashlib.sha256(
            subprocess.run([sys.executable, "-m", "pip", "freeze"],
                           capture_output=True, text=True, check=True).stdout.encode()
        ).hexdigest(),
        "weights_revision": HF_REF.read_text().strip() if HF_REF.exists() else "unknown",
        "weights_digest": "n/a",
    }
    sums = model_dir / "SHA256SUMS"
    if sums.exists():
        fp["weights_digest"] = hashlib.sha256(
            "\n".join(sorted(sums.read_text().splitlines())).encode()
        ).hexdigest()
    return fp


# ==========================================================================
# Perplexity — token-weighted, fixed window, fixed budget
# ==========================================================================
def build_ppl_windows(tok, window: int, budget: int) -> tuple[list[torch.Tensor], str]:
    """Concatenate the corpus, tokenise incrementally, cut into equal windows.

    Two implementation points that are easy to get wrong:

    * Tokenise row by row and stop once the budget is covered. Tokenising the whole
      corpus in one call builds a 300k-token sequence the model never sees in full:
      it wastes memory and makes the tokenizer warn about exceeding the model's
      maximum length, a warning a reader would reasonably treat as a real defect.
    * Perplexity is the exponential of the mean negative log-likelihood PER TOKEN,
      so aggregation must be token-weighted. Averaging per-window perplexities
      (each already exponentiated) would be mathematically wrong.
    """
    from datasets import load_dataset
    ds = load_dataset(PPL_DATASET[0], PPL_DATASET[1], split=PPL_DATASET[2])
    rows = [t for t in ds["text"] if t.strip()]

    ids: list[int] = []
    used_rows: list[str] = []
    for t in rows:
        chunk = tok(t, add_special_tokens=False)["input_ids"]
        ids.extend(chunk)
        used_rows.append(t)
        if len(ids) >= budget:
            break

    # Score exactly whole windows inside the declared budget, so the token count in
    # the result file matches the configured budget rather than whatever the row
    # boundary happened to produce.
    id_list = ids[:budget]
    take = (len(id_list) // window) * window
    ids_t = torch.tensor(id_list[:take], dtype=torch.long)
    corpus_sha = sha256_text("\n\n".join(used_rows))
    windows = [ids_t[i:i + window] for i in range(0, len(ids_t) - window + 1, window)]
    print(f"  corpus: {len(ds)} rows available, {len(ids_t)} tokens scored "
          f"(budget {budget}) from {len(used_rows)} rows, "
          f"{len(windows)} windows of {window} tokens")
    return windows, corpus_sha


def perplexity(model, windows: list[torch.Tensor]) -> dict:
    total_nll, total_tokens = 0.0, 0
    for w in windows:
        ids = w.unsqueeze(0).to("cuda")
        with torch.inference_mode():
            out = model(input_ids=ids, labels=ids)      # causal shift handled internally
        # out.loss is the mean NLL for this window; convert back to a sum so the
        # final figure can be aggregated across windows of equal length.
        n = ids.shape[1] - 1
        total_nll += float(out.loss) * n
        total_tokens += n
    ppl = float(np.exp(total_nll / total_tokens))
    return {"perplexity": ppl, "tokens_scored": total_tokens,
            "mean_nll": total_nll / total_tokens}


# ==========================================================================
# MMLU — log-likelihood scoring over four candidates
# ==========================================================================
def load_mmlu(n: int) -> tuple[list[dict], str]:
    from datasets import load_dataset
    ds = load_dataset(MMLU_DATASET[0], MMLU_DATASET[1], split=MMLU_DATASET[2])
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(ds), size=min(n, len(ds)), replace=False))
    items = [ds[int(i)] for i in idx]
    # Hash the exact items used, so a later run can prove it scored the same set.
    blob = "\n".join(f"{it['question']}|{it['choices'][it['answer']]}" for it in items)
    return items, sha256_text(blob)


def make_prompt(tok, question: str, choices: list[str]) -> torch.Tensor:
    """Question plus lettered choices, chat-templated with THINKING DISABLED.

    enable_thinking=False is not cosmetic. Qwen3's template enables thinking by
    default, so the assistant turn would begin with a long reasoning block; scoring
    " A" immediately after the generation prompt would then measure a token in a
    position the model never intended to put it, and accuracy collapses to chance.
    Measured both ways during development: chance-level with thinking on, well above
    chance with it off.
    """
    body = (
        question
        + "\n"
        + "\n".join(f"{L}. {c}" for L, c in zip(CHOICE_LETTERS, choices))
        + "\nAnswer with a single letter (A, B, C, or D)."
        + "\nAnswer:"
    )
    msgs = [{"role": "user", "content": body}]
    try:
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt"
        )
    except TypeError:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    return ids.to("cuda")


def score_choices(model, tok, prompt_ids: torch.Tensor) -> np.ndarray:
    """Sum of log-probs for each candidate letter continuation."""
    scores = []
    for letter in CHOICE_LETTERS:
        cont = tok.encode(" " + letter, add_special_tokens=False)
        ids = torch.cat([prompt_ids, torch.tensor([cont], device=prompt_ids.device)], dim=1)
        with torch.inference_mode():
            logits = model(input_ids=ids).logits
        # logits[:, :-1] predicts tokens 1..n; the continuation occupies the last
        # len(cont) positions, predicted at indices (prompt_len-1) .. (n-2).
        logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
        start = prompt_ids.shape[1] - 1
        total = sum(float(logprobs[start + i, cid]) for i, cid in enumerate(cont))
        scores.append(total)
    return np.asarray(scores)


def mmlu(model, tok, items: list[dict]) -> dict:
    correct = 0
    preds = []
    t0 = time.perf_counter()
    for k, it in enumerate(items):
        prompt = make_prompt(tok, it["question"], it["choices"])
        scores = score_choices(model, tok, prompt)
        pred = int(np.argmax(scores))
        preds.append(pred)
        correct += int(pred == it["answer"])
        if (k + 1) % 50 == 0:
            print(f"    {k+1}/{len(items)}  running acc {correct/(k+1):.4f}")
    acc = correct / len(items)
    chance = 1.0 / len(CHOICE_LETTERS)
    return {
        "accuracy": acc,
        "n_questions": len(items),
        "n_correct": correct,
        "chance_level": chance,
        "standard_error": float(np.sqrt(acc * (1 - acc) / len(items))),
        "elapsed_s": time.perf_counter() - t0,
        "predictions": preds,
    }


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Accuracy evaluation for one tier")
    ap.add_argument("--tier", default="bf16")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    ap.add_argument("--limit", type=int, default=MMLU_N)
    ap.add_argument("--ppl-tokens", type=int, default=PPL_TOKENS)
    ap.add_argument("--ppl-window", type=int, default=PPL_WINDOW)
    ap.add_argument("--ppl-only", action="store_true")
    ap.add_argument("--mmlu-only", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: 20 questions, 4096 perplexity tokens")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.smoke:
        args.limit = 20
        args.ppl_tokens = 4096

    model_dir = Path(args.model_dir)
    print(f"tier={args.tier}  model={model_dir}")
    print(f"smoke={args.smoke}  mmlu limit={0 if args.ppl_only else args.limit}  "
          f"ppl tokens={0 if args.mmlu_only else args.ppl_tokens}")

    t_load = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True
    )
    model.eval()
    print(f"model loaded in {time.perf_counter() - t_load:.1f}s")

    result = {
        "tier": args.tier,
        "evaluator": "scripts/07_eval_accuracy.py (native; lm-eval-harness not used)",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": fingerprint(model_dir),
        "config": {
            "model_dir": str(model_dir),
            "dtype": "bfloat16",
            "mmlu_limit": args.limit,
            "mmlu_seed": SEED,
            "mmlu_convention": MMLU_CONVENTION,
            "ppl_dataset": list(PPL_DATASET),
            "ppl_window_tokens": args.ppl_window,
            "ppl_token_budget": args.ppl_tokens,
            "smoke": args.smoke,
        },
        "metrics": {},
    }

    if not args.mmlu_only:
        print("\nperplexity:")
        windows, corpus_sha = build_ppl_windows(tok, args.ppl_window, args.ppl_tokens)
        ppl = perplexity(model, windows)
        ppl["corpus_sha256"] = corpus_sha
        ppl["dataset"] = list(PPL_DATASET)
        result["metrics"]["perplexity"] = ppl
        print(f"  perplexity {ppl['perplexity']:.4f}  over {ppl['tokens_scored']} tokens")

    if not args.ppl_only:
        print("\nMMLU:")
        items, sample_sha = load_mmlu(args.limit)
        m = mmlu(model, tok, items)
        m["sample_sha256"] = sample_sha
        m["dataset"] = list(MMLU_DATASET)
        result["metrics"]["mmlu"] = m
        print(f"  accuracy {m['accuracy']:.4f} ± {m['standard_error']:.4f} "
              f"(chance {m['chance_level']:.2f}, n={m['n_questions']})")
        print(f"  resolution: ±{1.96*m['standard_error']*100:.1f} percentage points at 95 %")

    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"_{args.tag}" if args.tag else ""
    smoke = "_smoke" if args.smoke else ""
    path = out_dir / f"{args.tier}_{stamp}{smoke}{tag}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
