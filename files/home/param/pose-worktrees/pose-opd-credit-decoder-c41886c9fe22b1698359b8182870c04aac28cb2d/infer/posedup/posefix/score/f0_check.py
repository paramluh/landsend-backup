import json, sys
sys.path.insert(0, "/data/home/amartyavarma/video/code")
import numpy as np, torch
import pose as P
from safetensors.torch import load_file
f0 = load_file("/data/home/amartyavarma/video/demo-data/prompts/01_ab63d745/llm_inputs.safetensors")
print({k: tuple(v.shape) for k, v in f0.items()})
yxc = f0["frame0_pose"].float().numpy()
kp0 = np.ascontiguousarray(yxc[:, [1, 0, 2]])
gs = P.global_state(kp0)
print("demo package frame0 global_state (decode):", np.asarray(gs).round(4).tolist())
m = json.load(open("/data/home/amartyavarma/ppm/aloe-posedup-live/genweb/openings/01_ab63d745/meta.json"))
fr = m["sources"]["pose_tokens"]["frame0"]["fields"]
print("opening window frame0 (tokens' reference): root", fr["frame0_root"], "width", fr["frame0_width"])
import inspect
print(inspect.getsource(P.global_state)[:1500])
