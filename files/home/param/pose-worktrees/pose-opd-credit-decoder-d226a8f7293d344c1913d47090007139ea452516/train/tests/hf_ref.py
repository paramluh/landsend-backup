"""step-0 check 1, reference side: stock HF Qwen3-32B (bf16, sdpa, device_map auto over the node's GPUs) logits on the plain token
sequence (step0_tokens) -> OUT (node-local /scratch). usage: python tests/hf_ref.py OUT [N]"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step0_tokens as S0  # noqa: E402

OUT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
DT = {"bf16": torch.bfloat16, "fp32": torch.float32}[sys.argv[3] if len(sys.argv) > 3 else "bf16"]
assert OUT.startswith("/scratch/")
from transformers import AutoModelForCausalLM  # noqa: E402

ids = torch.as_tensor(S0.plain_tokens(N))
t0 = time.time()
m = AutoModelForCausalLM.from_pretrained("/data/home/amartyavarma/models/Qwen3-32B", dtype=DT, device_map="auto",
                                         attn_implementation="sdpa")
m.eval()
print(f"HF Qwen3-32B loaded in {time.time() - t0:.0f}s", flush=True)
with torch.no_grad():
    lg = m(input_ids=ids[None].to("cuda:0"), use_cache=False).logits[0]
lg32 = lg.float()
ce = torch.nn.functional.cross_entropy(lg32[:-1], ids[1:].to(lg.device)).item()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save({"ids": ids, "logits": lg.cpu(), "dtype": str(DT), "ce": ce}, OUT)
print(f"HF reference: {len(ids)} tokens, next-token CE {ce:.4f}, logits {tuple(lg.shape)} {lg.dtype} -> {OUT}", flush=True)
