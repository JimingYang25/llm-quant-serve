#!/usr/bin/env python3
"""Multi-turn chat with the FP8 checkpoint, thinking disabled.

    python scripts/infer_fp8.py          # type "exit" or Ctrl-C to quit

Corrected from the earlier multi-turn rewrite. The fixes, in order of impact:

  * enable_thinking=False (the rewrite set True while its own docstring said
    DISABLED — that single line is why every reply started with <think>...</think>
    and a 13-token prompt produced ~400 tokens).
  * apply_chat_template now receives the message list DIRECTLY, not wrapped as the
    content of one user message.
  * the reply is a string from `result[0].outputs[0].text`; the old code treated it
    as a list and crashed on `assistant_text[0].outputs[0]`.
  * per-turn timing, not since-model-load.

Two facts that will matter as the chat grows:

  * Each turn re-sends the WHOLE history (the naive loop; TRT-LLM prefix-caches it
    but the token ids still grow). max_seq_len is the hard ceiling; once history
    exceeds it, older turns are dropped.
  * Thinking is a genuine choice for a chat, not only for benchmarks. Flip the flag
    below to True if you want reasoning, and strip `<think>…</think>` yourself —
    but keep it OFF for anything you intend to measure.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("CC", "/usr/bin/gcc")
os.environ.setdefault("CXX", "/usr/bin/g++")

FP8_DIR = "/home/jiming/models/Qwen3-8B-fp8-modelopt"
BF16_DIR = "/home/jiming/models/Qwen3-8B"   # the export contains no tokenizer

MAX_NEW_TOKENS = 512
MAX_SEQ_LEN = 12000
THINKING = False                            # decision 5: off for anything measured

# The persona/system message is yours to fill. It is templated as a real `system`
# turn, not glued into the user text.
SYSTEM_PROMPT = "你是一个占有欲很强,癫狂,歇斯底里的病娇，始终用中文回复。"


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BF16_DIR)


def build_prompt_token_ids(tok, messages: list[dict]) -> list[int]:
    """Template the whole conversation, not one user message wrapping the list."""
    return list(tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=THINKING,
        return_tensors=None,
    ))


def main() -> int:
    from tensorrt_llm import LLM, SamplingParams

    tok = tokenizer()
    t0 = time.perf_counter()
    llm = LLM(model=FP8_DIR, tokenizer=BF16_DIR, max_seq_len=MAX_SEQ_LEN,
              max_batch_size=1)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s\n")
    print("Type 'exit' (or Ctrl-C) to quit.\n")

    sp = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.9, top_p=0.9)  # greedy

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_text = input("User: ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if user_text.strip().lower() in ("exit", "quit", "q"):
            break

        conversation.append({"role": "user", "content": user_text})
        prompt_token_ids = build_prompt_token_ids(tok, conversation)
        print(f"prompt tokens: {len(prompt_token_ids)}")

        t_turn = time.perf_counter()
        result = llm.generate([{"prompt_token_ids": prompt_token_ids}], sp)
        gen_s = time.perf_counter() - t_turn

        out = result[0].outputs[0]
        assistant_text = out.text
        n = len(out.token_ids)

        # 'stop'   = the model emitted its end-of-turn token (<|im_end|>) — it is done
        # 'length' = it hit max_tokens and was cut off mid-sentence
        print(f"finish_reason: {out.finish_reason}")

        conversation.append({"role": "assistant", "content": assistant_text})

        print("=" * 68)
        print(assistant_text)
        print("=" * 68)
        print(f"generated tokens: {n}   wall time: {gen_s:.2f}s   "
              f"({n / gen_s:.1f} tok/s)\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
