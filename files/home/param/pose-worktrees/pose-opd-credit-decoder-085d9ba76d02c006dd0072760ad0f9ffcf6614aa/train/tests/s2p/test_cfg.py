#!/usr/bin/env python3
"""pose5 s2p test (a1): the trainer's RESOLVED configuration + the REAL-SIZE parameter table of one code dir under the sourced env.

usage: test_cfg.py <ORIG | COPY> <out.json>   (env: prod_811_fresh_append3.env sourced, RANK=0 WORLD_SIZE=128 LOCAL_RANK=0 LOCAL_WORLD_SIZE=8;
PPM_S2P as the variant wants). Dumps CFG (exactly what the trainer logs at start and saves in train_state.json), the 8,1,1 per-micro
coefficients, RESUME_KEYS, the stats layout NV, GA, the main-AdamW group list, and the full-size PPM's parameters (64 / 36 / 28 layers,
D33 decp_bb as in the 8,1,1 run, built on the META device: names, shapes, LR groups; no memory) with the code's own build arguments."""
import json
import os
import sys

import torch

sys.path.insert(0, "/data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p")
import s2p_cpu as H  # noqa: E402

code = {"ORIG": H.ORIG, "COPY": H.COPY}[sys.argv[1]]
sys.path.insert(0, code)
import train_ppm as T  # noqa: E402

M = T.M
assert os.path.dirname(os.path.abspath(M.__file__)) == code
C = T.CFG
s2p = getattr(T, "S2P", None)
kw = {"s2p": s2p} if s2p else {}
with torch.device("meta"):
    m = M.PPM(M.cfg32(C["layers"]), M.cfg4b(C["tower_layers"]), M.cfg06(C["dec_layers"]), C["proj_r"], C["tower_vocab"],
              pose_head=bool(C["pose_head"]), decp_bb=bool(C["decp_bb"]), **kw)
params = [[n, list(p.shape), M.group_of(n)] for n, p in m.named_parameters()]
out = {"code": code, "env_PPM_S2P": os.environ.get("PPM_S2P"), "S2P": s2p if hasattr(T, "S2P") else "<attribute absent: original code>",
       "CFG": C, "CFG_json": json.dumps(C), "COEF": T.COEF, "RESUME_KEYS": list(T.RESUME_KEYS), "NV": T.NV, "SEQ": list(T.SEQ),
       "GROUPS": list(M.GROUPS), "params": params, "n_params": len(params), "numel": sum(int(torch.Size(s).numel()) for _, s, _ in params)}
os.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True)
json.dump(out, open(sys.argv[2], "w"), indent=0, default=str)
H.log(f"{code} PPM_S2P={os.environ.get('PPM_S2P')}: cfg keys {len(C)}, params {len(params)} ({out['numel'] / 1e9:.4f}B) -> {sys.argv[2]}")
