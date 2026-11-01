"""posefix NAME-C: the LIVE server's prefix (prompts_ppm.build_prefix('', upload_clip(codes)), = server voice_clip_only / test_full.prefix_for)
for the SAME clip codes, diffed against the trainer-format prefix; plus the package clip check."""
import json, sys
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-posedup-live")
import numpy as np
import prompts_ppm as PR
from safetensors.numpy import load_file
D = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/name/c"
t = np.load(f"{D}/train_side_prefix.npz")
codes = t["clip_codes"]
pk = load_file("/data/home/amartyavarma/ppm/sam-sep-handoff/test/06_kate_fdba6c86/voice_clip.safetensors")
pkc = np.asarray(next(iter(pk.values())), np.int64)
print("package voice_clip == row codes 2416:2541:", pkc.shape, bool(pkc.shape == codes.shape and (pkc == codes).all()))
spx, info = PR.build_prefix("", PR.upload_clip(codes), prose=False)
res = {"P_train": int(t["P"]), "P_live": int(spx["P"])}
for k in ("ids", "depth", "present"):
    a, b = t[k], np.asarray(spx[k])
    res[k] = {"shape_eq": a.shape == b.shape, "equal": bool(a.shape == b.shape and (a == b).all())}
    if a.shape == b.shape and not (a == b).all():
        res[k]["diff_positions"] = np.argwhere(a != b)[:10].tolist()
print(json.dumps(res))
print("live info", info)
print("prefix ids B (first 8, last 6):", spx["ids"][1][:8].tolist(), spx["ids"][1][-6:].tolist())
