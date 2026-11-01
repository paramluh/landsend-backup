"""a16861 rerun: check an ASR adapter export (COMPLETE, no-norm, vadapter.* keys) and print sha256(bf16) exactly as train_samsep.init_adapter
computes it (= the VOX_ADAPTER_SHA prefix). usage (compute node, qwen-s2 venv): python -I prep_adapter_a16861.py <adapter export dir>"""
import os, sys, json, pickle, hashlib
import torch, torch.distributed.checkpoint as dcp
p = sys.argv[1]
assert os.path.isfile(f"{p}/COMPLETE"), p
md = pickle.load(open(f"{p}/.metadata", "rb")).state_dict_metadata
keys = sorted(k for k in md if k.startswith("vadapter."))
sd = {k: torch.empty(tuple(md[k].size), dtype=md[k].properties.dtype) for k in keys}
dcp.load(sd, checkpoint_id=p, no_dist=True)
ts = json.load(open(f"{p}/train_state.json"))
assert ts["cfg"].get("s4_vadapt_norm") == "none" and ts["cfg"].get("vadapt_norm") == "none", "not the no-norm ASR adapter export"
assert "vadapter.target_rms" not in sd and "vadapter.gain" not in sd
h = hashlib.sha256()                                   # = train_samsep.adapter_sha on the bf16 tensors
for k in sorted(sd):
    v = sd[k].to(torch.bfloat16)
    h.update(k.encode()); h.update(v.contiguous().view(torch.uint8).numpy().tobytes() if v.dim() else v.reshape(1).view(torch.uint8).numpy().tobytes())
print(json.dumps({"path": p, "run_tag": ts["cfg"].get("run_tag"), "step": ts["step"], "n_tensors": len(sd),
                  "params_M": round(sum(v.numel() for v in sd.values()) / 1e6, 1), "sha256_bf16": h.hexdigest(), "sha16": h.hexdigest()[:16]}))
