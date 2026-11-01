import numpy as np
from safetensors.numpy import load_file
P="/data/home/amartyavarma/video/demo-data/prompts/13_956c09af"
a=load_file(f"{P}/llm_inputs.safetensors"); b=load_file(f"{P}/gt/pose_tokens.safetensors")
print({k:v.shape for k,v in a.items()}); print({k:v.shape for k,v in b.items()})
print(np.asarray(a["global_state"]).reshape(-1)[:6]); print(np.asarray(b.get("global_state",np.zeros(1))).reshape(-1)[:6])
