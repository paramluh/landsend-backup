"""E4-E6 vdata (12 s = GT length: 76 pose latents, 150 Mimi codes), CPU only.
  E4 generated pose (the sample's own vdata latents) + GT Mimi
  E5 GT pose latents + the sample's generated Mimi (first 150 codes)
  E6 GT pose + GT Mimi (= the existing GT render's inputs)
  E9 generated pose + generated Mimi, both trimmed to 12 s (length-matched control for E4-E6)
   python make_swaps.py ROW:RID:K:CLIP [...]"""
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

REQ = "/data/home/amartyavarma/ppm/ppm-pose-server/requests"
GT = "/data/home/amartyavarma/video/studio-posemono/sam8b-eval-cfg/vdata"
OUT = "/data/home/amartyavarma/ppm/gaze-drift/render/vdata"
rep = {}
for spec in sys.argv[1:]:
    row, rid, k, clip = spec.split(":")
    sv, gv = f"{REQ}/{rid}/s{k}/vdata", f"{GT}/GT-{clip}"
    gp, gm = load_file(f"{gv}/pose_latents_electron_v3.safetensors"), load_file(f"{gv}/mimi_local.safetensors")["codes"]
    sp, sm = load_file(f"{sv}/pose_latents_electron_v3.safetensors"), load_file(f"{sv}/mimi_local.safetensors")["codes"]
    L, C = gp["latents"].shape[0], gm.shape[1]
    assert C == 2 * (L - 1), (L, C)
    trim = lambda p: {"latents": p["latents"][:L].contiguous(), "n_frames": torch.tensor(4 * (L - 1) + 1), "valid_frames": torch.tensor(4 * (L - 1))}  # noqa: E731
    sets = {"E4": (trim(sp), gm), "E5": (gp, sm[:, :C].contiguous()), "E6": (gp, gm), "E9": (trim(sp), sm[:, :C].contiguous())}
    for e, (p, m) in sets.items():
        key = f"GAZE-{e}-{row}"
        vd = f"{OUT}/{key}"
        os.makedirs(vd, exist_ok=True)
        save_file({kk: v.contiguous() for kk, v in p.items()}, f"{vd}/pose_latents_electron_v3.safetensors")
        save_file({"codes": m.contiguous()}, f"{vd}/mimi_local.safetensors")
        if not os.path.lexists(f"{vd}/video_latents.safetensors"):
            os.symlink(f"{gv}/video_latents.safetensors", f"{vd}/video_latents.safetensors")
    s0g = load_file(f"{gv}/video_latents.safetensors")["latents"][0]
    s0s = load_file(f"{sv}/video_latents.safetensors")["latents"][0]
    rep[row] = {"gt_L": L, "gt_codes": C, "sample_L": sp["latents"].shape[0], "sample_codes": sm.shape[1],
                "sink_equal": bool(torch.equal(s0g.to(torch.bfloat16), s0s.to(torch.bfloat16))),
                "mimi_prefill_equal_25": bool(torch.equal(sm[:, :25].long(), gm[:, :25].long())),
                "mimi_frac_equal_after_prefill": float((sm[:, 25:C].long() == gm[:, 25:].long()).float().mean()),
                "pose_lat_prefill_rel_diff": float((sp["latents"][:13].float() - gp["latents"][:13].float()).norm() / gp["latents"][:13].float().norm())}
    print(row, json.dumps(rep[row]))
json.dump(rep, open(f"{OUT}/swap_report.json", "w"), indent=1)
