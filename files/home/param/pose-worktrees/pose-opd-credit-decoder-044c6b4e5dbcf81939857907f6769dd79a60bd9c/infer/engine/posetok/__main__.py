"""python -m posetok POSE.safetensors FRAME0.safetensors OUT.safetensors
POSE 'codes' int [32, F]; FRAME0 'frame0_pose' float [133, 3] (y, x, conf) px."""
import sys

from safetensors.torch import load_file, save_file

from . import tokens_to_latents

pose, frame0, out = sys.argv[1:4]
r = tokens_to_latents(load_file(pose)['codes'], load_file(frame0)['frame0_pose'])
save_file(r, out)
print(out, tuple(r['latents'].shape), 'n_frames', int(r['n_frames']))
