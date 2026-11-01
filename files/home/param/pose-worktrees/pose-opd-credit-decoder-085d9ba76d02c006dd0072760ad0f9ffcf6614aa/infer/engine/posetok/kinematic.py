# wan-vae-collection@588eaeb utils/kinematic.py (verbatim)
"""
55-joint kinematic chain (COCO-WholeBody face + upper body + hands), the
form aloe v3 stores poses in: per joint (off_y, off_x, conf) with offsets
parent-relative in shoulder widths, root = mid-shoulder (joints 5 and 6).
"""

import numpy as np
import torch

NUM_JOINTS = 55  # 5 face + 8 upper body + 21 + 21 hand joints
# COCO-WholeBody 133 -> the 55: face 0-4, upper body 5-12, hands 91-132
WHOLEBODY_INDEX = list(range(13)) + list(range(91, 133))


def _hand_parents(hand_root):
    parents = [-1]
    for finger in range(5):
        base = hand_root + 1 + finger * 4
        parents.append(hand_root)
        for j in range(1, 4):
            parents.append(base + j - 1)
    return parents


_LEFT_HAND_PARENTS = _hand_parents(13)
_LEFT_HAND_PARENTS[0] = 9    # left wrist root hangs off body joint 9
_RIGHT_HAND_PARENTS = _hand_parents(34)
_RIGHT_HAND_PARENTS[0] = 10

PARENT_INDEX = np.array(
    [-1, 0, 0, 1, 2,        # nose, eyes, ears
     -1, -1, 5, 6, 7, 8,    # shoulders, elbows, wrists
     5, 6]                  # hips
    + _LEFT_HAND_PARENTS + _RIGHT_HAND_PARENTS, dtype=np.int64)


def _topo_order(parents):
    order, seen = [], set()

    def visit(i):
        if i < 0 or i in seen:
            return
        visit(parents[i])
        seen.add(i)
        order.append(i)

    for i in range(len(parents)):
        visit(i)
    return np.array(order, dtype=np.int64)


TOPO_ORDER = _topo_order(PARENT_INDEX)

FACE_CONNECTIONS = [(0, 1), (0, 2), (1, 3), (2, 4)]
BODY_CONNECTIONS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
                    (11, 12)]
_HAND = [(0, 1), (1, 2), (2, 3), (3, 4),
         (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20)]
LEFT_HAND_CONNECTIONS = [(i + 13, j + 13) for i, j in _HAND]
RIGHT_HAND_CONNECTIONS = [(i + 34, j + 34) for i, j in _HAND]


def offsets_to_positions(root: torch.Tensor, offsets: torch.Tensor):
    """root (..., 2) + offsets (..., 55, 2) -> absolute positions (..., 55, 2)."""
    positions = torch.empty_like(offsets)
    for i in TOPO_ORDER:
        parent = PARENT_INDEX[i]
        base = root if parent == -1 else positions[..., parent, :]
        positions[..., i, :] = base + offsets[..., i, :]
    return positions


def raw_to_recon(kp: torch.Tensor, frame0: torch.Tensor, src_h: int,
                 src_w: int):
    """
    Raw COCO-WholeBody kp [T, 133, 3] (x, y, conf) in px of a src_h x src_w
    canvas -> (pose_recon [T, 165], global_state [T, 3]) under the aloe
    `frame0-root-frame0-horizontal-shoulder-width-v1` normalisation: root
    (mid-shoulder) and shoulder width come from `frame0` [133, 3], the
    chunk's frame 0, and are held for every frame; confidence untouched.
    Inverse of what render_pose_canvas does, so the canvas is the raw pose.
    """
    def yx(p):
        p = p[..., WHOLEBODY_INDEX, :]
        return torch.stack([p[..., 1] / src_h, p[..., 0] / src_w], -1)

    pos, pos0 = yx(kp), yx(frame0)
    root = (pos0[5] + pos0[6]) / 2
    width = (pos0[5, 1] - pos0[6, 1]).abs()
    parents = torch.as_tensor(PARENT_INDEX)
    base = pos[:, parents.clamp(min=0)]
    base[:, parents < 0] = root
    recon = torch.cat([(pos - base) / width, kp[:, WHOLEBODY_INDEX, 2:]], -1)
    gs = torch.cat([root, width[None]]).repeat(len(kp), 1)
    return recon.flatten(1), gs
