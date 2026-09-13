import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

t0 = time.time()
tok = AutoTokenizer.from_pretrained("/home/jiming/models/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained(
    "/home/jiming/models/Qwen3-8B", dtype=torch.bfloat16, device_map="cuda"
)
print("load %.1fs" % (time.time() - t0))

msgs = [{"role": "user", "content": "what does yandere mean"}]

tok.padding_side = "left"

ids = tok.apply_chat_template(
    msgs,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
    return_dict=True,
).to("cuda")

prompt_len = ids["input_ids"].shape[-1]

out = model.generate(
    **ids,
    do_sample=False,
    max_new_tokens=256,
)

gen = out[0][prompt_len:]
print(tok.decode(gen, skip_special_tokens=True))
print("generated tokens:", gen.shape[-1])
print("peak VRAM %.2f GB" % (torch.cuda.max_memory_allocated() / 2**30))