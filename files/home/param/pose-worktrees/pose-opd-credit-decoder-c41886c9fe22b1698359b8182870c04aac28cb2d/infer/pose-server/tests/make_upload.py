import numpy as np
from safetensors.numpy import load_file
pd = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/prompts/21_8dc5d622"
li = load_file(f"{pd}/llm_inputs.safetensors")
gp = load_file(f"{pd}/gt/pose_tokens.safetensors")["codes"]
import safetensors.torch as ST
vl = ST.load_file(f"{pd}/gt/video_latents.safetensors")["latents"].float().numpy()
np.savez("/data/home/amartyavarma/ppm/ppm-pose-server/tests/upload_pose_21.npz", prefill_pose=gp[:, :50].astype(np.int16), global_state=li["global_state"],
         frame0_pose=li["frame0_pose"], sink_latent=vl[0].astype(np.float32))
print(gp.shape, vl.shape)
