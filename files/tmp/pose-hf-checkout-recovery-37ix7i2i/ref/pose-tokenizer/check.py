from safetensors.torch import load_file
import numpy as np, sys; sys.path.insert(0, "/home/james/pose-tokenizer")
from pose_tokenizer.data.rendering import kp_flat_to_positions
from pose_tokenizer.data.kinematic import NUM_JOINTS   # 55

# A specific processed-pose chunk (reconstruction + global_state, 77 frames).
clip = "6694724883101781254_00-00.00_to_00-09.70"
f = f"/mnt/somfs/hands/poses/v2/6694724883101781254/{clip}/{clip}_poses_chunk0000.safetensors"

st = load_file(f)
recon, gs = st["reconstruction"].numpy(), st["global_state"].numpy()   # (77,165), (77,3)
print("file:", f)
print("reconstruction:", recon.shape, " global_state:", gs.shape)

t = 0
r = recon[t].reshape(NUM_JOINTS, 3)        # (55, [y, x, conf]) — normalized offsets
print(f"\n--- frame {t} ---")
print("global_state [root_y, root_x, shoulder_width]:", np.round(gs[t], 4))
print("reconstruction offsets (first 6 joints) [y_off, x_off, conf]:")
for j in range(6):
    print(f"  joint{j:>2}: {r[j,0]:+.4f} {r[j,1]:+.4f}  conf={r[j,2]:.3f}")

xy = recon[t:t+1].reshape(1, NUM_JOINTS, 3)[..., :2].reshape(1, NUM_JOINTS * 2)
abs_pos = kp_flat_to_positions(xy, gs[t:t+1])[0]   # (55, [y,x]) absolute 0-1 coords
print("absolute positions (first 6 joints) [y, x] in 0-1 image coords:")
for j in range(6):
    print(f"  joint{j:>2}: {abs_pos[j,0]:.4f} {abs_pos[j,1]:.4f}")
