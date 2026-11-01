"""Production duplex examples shared by the student and its frozen teachers.

No separate teacher prompt sampling occurs. B's prefix, audio history, text
history, pose history, positions and targets remain identical. The native
monoplex teacher receives empty A, as in its production fine-tune. Student and
audio teacher receive the same raw A waveform; the trainer can encode it once.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from .recipe import (DEFAULT_CHECKPOINT_CFG, DEFAULT_DATA_ROOT, EVAL_ITEMS,
                     PRODUCTION_DIR, TRAIN_ITEMS_PER_EPOCH,
                     configure_environment, load_recipe)


def teacher_views(student, pad_id):
    """Return three aligned numpy micros without resampling or altering student."""
    mono = dict(student)
    for key in ("ids", "depth", "present"):
        mono[key] = np.array(student[key], copy=True)
    mono["ids"][0] = int(pad_id)
    mono["depth"][0] = -1
    mono["present"][0] = 0
    for key in ("vin", "wav", "aref"):
        mono.pop(key, None)
    audio = dict(student)
    for key in ("pdepth", "pcb0", "plight", "pose_ok", "pose_src"):
        audio.pop(key, None)
    audio["kind"] = "speech"
    audio["own"] = np.asarray(student["own"])[:, :3]
    return {"student": student, "pose_teacher": mono, "audio_teacher": audio}


class ProductionData:
    """Thin adapter around the unchanged production data implementation."""

    def __init__(self, data_root=DEFAULT_DATA_ROOT, checkpoint_cfg=DEFAULT_CHECKPOINT_CFG):
        self.data_root = Path(data_root).resolve()
        self.recipe = load_recipe(checkpoint_cfg)
        self.cfg = self.recipe["data"]
        self.eval_cfg = self.recipe["eval_data"]
        old = sys.modules.get("data_aloe_posedup")
        if old is not None and Path(old.POSE_INDEX) != self.data_root / "pose/POSE_INDEX.json":
            raise RuntimeError("Production data was already imported for a different data root")
        configure_environment(self.data_root)
        for path in (PRODUCTION_DIR / "lib", PRODUCTION_DIR):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        self.module = importlib.import_module("data_aloe_posedup")
        if Path(self.module.__file__).resolve().parent != PRODUCTION_DIR:
            raise RuntimeError("A different data_aloe_posedup module is already loaded")
        if self.module.aloe_cfg() != self.cfg or self.module.eval_cfg() != self.eval_cfg:
            raise ValueError("Resolved data config differs from production export-498")
        self._stores = None

    @property
    def stores(self):
        if self._stores is None:
            self._stores = self.module.Stores()
        return self._stores

    def make_plan(self, world=48, epochs=2):
        if int(world) != world or world < 1 or int(epochs) != epochs or epochs < 1:
            raise ValueError("world and epochs must be positive integers")
        steps = ((TRAIN_ITEMS_PER_EPOCH + int(world) - 1) // int(world)) * int(epochs)
        plan = self.module.make_plan(int(world), steps=steps)
        if plan["n_items0"] != TRAIN_ITEMS_PER_EPOCH or len(plan["eval"]) != EVAL_ITEMS:
            raise ValueError("Dataset split membership differs from the production run")
        if plan["n_train_micros"] != TRAIN_ITEMS_PER_EPOCH * int(epochs):
            raise ValueError("Epoch padding dropped or duplicated real training examples")
        plan["digest"], plan["epoch0_digest"] = self.module.plan_digest(plan)
        if plan["epoch0_digest"] != "d1a3e3721328da0a":
            raise ValueError("First-epoch item order differs from production")
        plan["world"] = int(world)
        return plan

    def batch(self, item, rank, *, load_audio=True, dummy=False, evaluation=False):
        """Load one production item; dummy requires a real fallback item."""
        if item is None:
            raise ValueError("Supply a real fallback item when dummy=True")
        if load_audio and not os.environ.get("S4_FFMPEG"):
            raise RuntimeError("Set S4_FFMPEG to production FFmpeg 7.0.2, or install imageio-ffmpeg==0.6.0")
        cfg = self.eval_cfg if evaluation else self.cfg
        micro = self.stores.conv(tuple(map(int, item)), cfg, int(rank), audio=load_audio)
        student = self.module.as_micro(micro, dummy=bool(dummy))
        student["row_key"] = micro["row_key"]
        student["conditioning"] = {
            "voice_drawn": int(micro["voice_drawn"]), "voice_clip": int(micro["voice_clip"]),
            "prefix_frames": int(micro["P"]), "pose_teacher_a": "native_empty",
        }
        return teacher_views(student, self.module.D.PAD_ID)

    def smoke_items(self, plan, count=8, *, frames=7500, require_pose=True):
        """Deterministic real full-length rows for memory/conditioning GPU checks."""
        if count < 1:
            raise ValueError("count must be positive")
        index = self.module.pose_index()["rows"]
        selected = []
        for item in plan["items0"]:
            k, row, _ = item
            if int(index[str(k)][str(row)]["n"]) != int(frames):
                continue
            if require_pose and not self.module.pose_ok_item(item):
                continue
            selected.append(item)
            if len(selected) == count:
                return selected
        raise ValueError(f"Only {len(selected)} qualifying rows have {frames} frames")

    def rank_dataset(self, plan, rank, world, *, start=0, load_audio=True):
        return RankDataset(self, plan, rank, world, start=start, load_audio=load_audio)


class RankDataset:
    """One example per rank per step, including production zero-weight dummies."""

    def __init__(self, data, plan, rank, world, *, start=0, load_audio=True):
        if not 0 <= int(rank) < int(world) or plan.get("world", world) != world:
            raise ValueError("Rank/world do not match the data plan")
        if not 0 <= int(start) <= plan["steps"]:
            raise ValueError("Resume step lies outside the data plan")
        self.data, self.rank, self.world = data, int(rank), int(world)
        self.items, self.eval_items = data.module.rank_items(plan, self.rank, self.world)
        self.fallback = plan["items0"][0]
        self.start, self.load_audio = int(start), bool(load_audio)

    def __len__(self):
        return len(self.items) - self.start

    def __getitem__(self, index):
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        step = self.start + int(index)
        item = self.items[step]
        bundle = self.data.batch(item if item is not None else self.fallback, self.rank,
                                 load_audio=self.load_audio, dummy=item is None)
        for micro in bundle.values():
            micro["step"] = step
        return bundle


def smoke_report(data_root=DEFAULT_DATA_ROOT, checkpoint_cfg=DEFAULT_CHECKPOINT_CFG,
                 *, load_audio=False):
    """Exercise complete epochs and representative real conditioning branches."""
    data = ProductionData(data_root, checkpoint_cfg)
    plan = data.make_plan()
    cases = {}
    for item in plan["items0"][:512]:
        batch = data.batch(item, 0, load_audio=load_audio)
        s, p, a = (batch[k] for k in ("student", "pose_teacher", "audio_teacher"))
        case = ("pose_missing" if not s["pose_ok"] else
                "voice_clip" if s["voice_clip"] else
                "voice_draw_missing" if s["voice_drawn"] else "unconditioned")
        if case in cases:
            continue
        assert np.array_equal(s["ids"][1], p["ids"][1])
        assert np.array_equal(s["depth"][1], p["depth"][1])
        assert np.array_equal(s["present"][1], p["present"][1])
        assert p["pdepth"] is s["pdepth"]
        assert (p["depth"][0] == -1).all() and (p["present"][0] == 0).all()
        assert (p["ids"][0] == data.module.D.PAD_ID).all()
        assert a["ids"] is s["ids"] and a["depth"] is s["depth"]
        assert "pdepth" not in a and "vin" not in p and "wav" not in p
        assert s["segs"] == p["segs"] == a["segs"]
        if load_audio:
            assert s["wav"] is a["wav"]
        # Compare against a fresh invocation of the original production loader.
        reference = data.module.as_micro(data.stores.conv(item, data.cfg, 0, audio=load_audio))
        for key, value in reference.items():
            if isinstance(value, np.ndarray):
                assert np.array_equal(s[key], value), key
            else:
                assert s[key] == value, key
        cases[case] = {"item": list(item), "frames": len(s["pid"]),
                       "pose_ok": int(s["pose_ok"]), "conditioning": s["conditioning"]}
        if {"pose_missing", "voice_clip", "unconditioned"}.issubset(cases):
            break
    assert {"pose_missing", "voice_clip", "unconditioned"}.issubset(cases)
    dummy = data.batch(plan["items0"][0], 0, load_audio=False, dummy=True)
    assert all(not np.asarray(m["own"]).any() for m in dummy.values())
    return {"ok": True, "steps": plan["steps"], "epoch_steps": plan["epoch_steps"],
            "real_micros": plan["n_train_micros"], "dummy_micros": plan["n_dummy"],
            "eval_micros": len(plan["eval"]), "plan_digest": plan["digest"],
            "epoch0_digest": plan["epoch0_digest"], "raw_audio_tested": load_audio,
            "cases": cases, "dummy_zero_counts": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--checkpoint-cfg", default=str(DEFAULT_CHECKPOINT_CFG))
    parser.add_argument("--audio", action="store_true")
    args = parser.parse_args()
    print(json.dumps(smoke_report(args.data_root, args.checkpoint_cfg, load_audio=args.audio), indent=2))
