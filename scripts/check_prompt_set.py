#!/usr/bin/env python3
"""Validate and measure the frozen prompt set (v1.1 contract).

Contract: prompt_text = '\\n\\n'.join(paragraphs from each doc_ref, sliced if
paragraph_slice present, then instruction).

Token counts use the real Qwen3-8B tokenizer, chat template applied,
thinking DISABLED (measurement_spec.md decision 5).

Writes prompts/prompt_set.stats.json (including SHA-256 of prompt_set.json).
Read-only with respect to prompt_set.json.
"""
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

REPO = Path.home() / "llm-quant-serve"
PS_PATH = REPO / "prompts" / "prompt_set.json"
TOK_DIR = Path.home() / "models" / "Qwen3-8B-tokenizer"
STATS_PATH = REPO / "prompts" / "prompt_set.stats.json"

TARGETS = {"short": (10, 30), "medium": (220, 300), "long": (900, 1100)}

from transformers import AutoTokenizer  # noqa: E402

data = json.loads(PS_PATH.read_text(encoding="utf-8"))
docs = data["documents"]
tok = AutoTokenizer.from_pretrained(str(TOK_DIR))


def build_text(p):
    paras = []
    for ref in p.get("doc_refs", []):
        section = docs[ref]["paragraphs"]
        if p.get("paragraph_slice"):
            a, b = p["paragraph_slice"]
            section = section[a:b]
        paras.extend(section)
    paras.append(p["instruction"])
    return "\n\n".join(paras)


def count_tokens(text):
    msgs = [{"role": "user", "content": text}]
    try:
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    return len(ids)


prompts = data["prompts"]
ids = [p["id"] for p in prompts]
problems = []

if len(ids) != len(set(ids)):
    problems.append("duplicate prompt ids")

by_bucket = Counter(p["bucket"] for p in prompts)
for b, n in (("short", 10), ("medium", 10), ("long", 10)):
    if by_bucket[b] != n:
        problems.append(f"bucket {b}: {by_bucket[b]} prompts, expected {n}")

for p in prompts:
    for ref in p.get("doc_refs", []):
        if ref not in docs:
            problems.append(f"{p['id']}: unknown doc_ref {ref}")
    if p["bucket"] == "short" and p.get("doc_refs"):
        problems.append(f"{p['id']}: short bucket must not carry doc_refs")
    if p["bucket"] == "medium":
        if not p.get("doc_refs"):
            problems.append(f"{p['id']}: medium bucket needs doc_refs")
        if not p.get("paragraph_slice"):
            problems.append(f"{p['id']}: medium bucket needs paragraph_slice")
    if p["bucket"] == "long" and len(p.get("doc_refs", [])) != 2:
        problems.append(f"{p['id']}: long bucket should pair exactly two documents")

rows = []
for p in prompts:
    text = build_text(p)
    rows.append(
        dict(
            id=p["id"],
            bucket=p["bucket"],
            language=p["language"],
            doc_refs=p.get("doc_refs", []),
            paragraph_slice=p.get("paragraph_slice"),
            tokens_chat_template=count_tokens(text),
            tokens_plain=len(tok(text)["input_ids"]),
            chars=len(text),
        )
    )

print("=== per-bucket token counts (chat template, thinking disabled) ===")
summary = {}
for b in ("short", "medium", "long"):
    vals = [r["tokens_chat_template"] for r in rows if r["bucket"] == b]
    lo, hi = TARGETS[b]
    med = int(statistics.median(vals))
    ok = "OK" if lo <= med <= hi else "OUT OF RANGE"
    summary[b] = dict(n=len(vals), min=min(vals), median=med,
                      mean=round(statistics.mean(vals), 1), max=max(vals),
                      target=[lo, hi], verdict=ok)
    print(f"  {b:7s} n={len(vals):2d} min={min(vals):5d} median={med:5d} "
          f"mean={statistics.mean(vals):7.1f} max={max(vals):5d} "
          f"target=[{lo},{hi}] -> {ok}")

print("\n=== per-prompt ===")
for r in rows:
    refs = "+".join(r["doc_refs"]) or "-"
    sl = r["paragraph_slice"] or ""
    print(f"  {r['id']:4s} {r['bucket']:6s} {r['language']:2s} "
          f"chat={r['tokens_chat_template']:5d} plain={r['tokens_plain']:5d} "
          f"doc={refs:8s} slice={sl}")

sha = hashlib.sha256(PS_PATH.read_bytes()).hexdigest()
stats = dict(
    prompt_set_sha256=sha,
    prompt_set_version=data["metadata"]["version"],
    tokenizer=str(TOK_DIR),
    chat_template_applied=True,
    thinking_enabled=False,
    buckets=summary,
    per_prompt=rows,
    structural_problems=problems,
)
STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\nSHA-256(prompt_set.json) = {sha}")
print("structural problems:", problems or "none")
print("stats written ->", STATS_PATH)
