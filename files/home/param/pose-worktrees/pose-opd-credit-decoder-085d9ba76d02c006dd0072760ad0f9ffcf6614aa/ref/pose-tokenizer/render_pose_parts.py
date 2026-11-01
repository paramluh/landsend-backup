#!/usr/bin/env python3
"""Render the 55-joint skeleton and the proposed N=20, M=5 part decomposition.

Produces a 4x5 grid: each panel shows the full skeleton in grey with one
part's joints/edges highlighted, so overlaps are visible (a shared joint
lights up in multiple panels).
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = Path(__file__).resolve().parent / "pose_parts_render.png"

# ---------------------------------------------------------------------------
# Canonical schematic coordinates for the 55 joints (x right, y up).
# ---------------------------------------------------------------------------
pos = np.zeros((55, 2), dtype=float)

# Body / face (subject-left drawn on +x).
pos[0] = (0.00, 0.95)    # nose
pos[1] = (0.06, 1.02)    # left eye
pos[2] = (-0.06, 1.02)   # right eye
pos[3] = (0.12, 1.00)    # left ear
pos[4] = (-0.12, 1.00)   # right ear
pos[5] = (0.22, 0.75)    # left shoulder
pos[6] = (-0.22, 0.75)   # right shoulder
pos[7] = (0.42, 0.45)    # left elbow
pos[8] = (-0.42, 0.45)   # right elbow
pos[9] = (0.55, 0.15)    # left wrist
pos[10] = (-0.55, 0.15)  # right wrist
pos[11] = (0.12, 0.45)   # left hip
pos[12] = (-0.12, 0.45)  # right hip


def place_hand(wrist_xy, sign):
    """Return positions for 21 hand joints (wrist + 5 fingers x 4)."""
    out = np.zeros((21, 2), dtype=float)
    wx, wy = wrist_xy
    out[0] = (wx, wy)
    scale = 0.16
    # Fingers fan downward/outward from the wrist.
    finger_angles = [-145, -120, -100, -80, -60]  # thumb..pinky (deg)
    for f, ang in enumerate(finger_angles):
        rad = math.radians(ang)
        dx, dy = math.cos(rad), math.sin(rad)
        base = 1 + f * 4
        for k in range(4):
            r = scale * (1 + k)
            out[base + k] = (wx + sign * dx * r, wy + dy * r)
    return out


pos[13:34] = place_hand(pos[9], sign=+1.0)   # left hand
pos[34:55] = place_hand(pos[10], sign=-1.0)  # right hand

# ---------------------------------------------------------------------------
# Edges for drawing the skeleton.
# ---------------------------------------------------------------------------
FACE_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4)]
BODY_EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (0, 5), (0, 6)]
ARM_HAND_EDGES = [(9, 13), (10, 34)]
HAND_LOCAL = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
LEFT_HAND_EDGES = [(i + 13, j + 13) for i, j in HAND_LOCAL]
RIGHT_HAND_EDGES = [(i + 34, j + 34) for i, j in HAND_LOCAL]
ALL_EDGES = FACE_EDGES + BODY_EDGES + ARM_HAND_EDGES + LEFT_HAND_EDGES + RIGHT_HAND_EDGES

# ---------------------------------------------------------------------------
# The N=20, M=5 parts.
# ---------------------------------------------------------------------------
PARTS = {
    "L thumb": [13, 14, 15, 16, 17],
    "L index": [13, 18, 19, 20, 21],
    "L middle": [13, 22, 23, 24, 25],
    "L ring": [13, 26, 27, 28, 29],
    "L pinky": [13, 30, 31, 32, 33],
    "R thumb": [34, 35, 36, 37, 38],
    "R index": [34, 39, 40, 41, 42],
    "R middle": [34, 43, 44, 45, 46],
    "R ring": [34, 47, 48, 49, 50],
    "R pinky": [34, 51, 52, 53, 54],
    "L arm": [11, 5, 7, 9, 13],
    "R arm": [12, 6, 8, 10, 34],
    "L face": [0, 1, 3, 5, 6],
    "R face": [0, 2, 4, 6, 5],
    "L knuckles": [14, 18, 22, 26, 30],
    "L mids": [15, 19, 23, 27, 31],
    "L tips": [17, 21, 25, 29, 33],
    "R knuckles": [35, 39, 43, 47, 51],
    "R mids": [36, 40, 44, 48, 52],
    "R tips": [38, 42, 46, 50, 54],
}


def draw_panel(ax, title, joints):
    joint_set = set(joints)
    # Base skeleton in light grey.
    for i, j in ALL_EDGES:
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                color="#d0d0d0", lw=0.8, zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], s=4, color="#d0d0d0", zorder=2)
    # Highlight edges fully inside the part.
    for i, j in ALL_EDGES:
        if i in joint_set and j in joint_set:
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    color="#d62728", lw=1.8, zorder=3)
    # Highlight the part's joints.
    pts = pos[joints]
    ax.scatter(pts[:, 0], pts[:, 1], s=22, color="#1f77b4", zorder=4)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")


fig, axes = plt.subplots(4, 5, figsize=(16, 13))
for ax, (title, joints) in zip(axes.ravel(), PARTS.items()):
    draw_panel(ax, title, joints)

fig.suptitle("Pose skeleton: N=20, M=5 parts (blue = part joints, red = within-part bones)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(OUT_PATH, dpi=130)
print(f"saved={OUT_PATH}")
