"""Check teacher conditioning isolation and aligned masked supervision."""
import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from finetune.duplex_distill.data import ProductionData, shared_training_depth_frames, teacher_views
from finetune.duplex_distill.recipe import DEFAULT_CHECKPOINT_CFG, DEFAULT_DATA_ROOT, load_recipe


def test_teacher_views_keep_shared_b_conditioning_without_mutating_student():
    student = {
        "kind": "pose", "ids": np.arange(20).reshape(2, 10),
        "depth": np.arange(2 * 10 * 32).reshape(2, 10, 32),
        "present": np.ones((2, 10), dtype=np.float32),
        "pdepth": np.arange(320).reshape(10, 32),
        "pcb0": np.arange(10), "plight": np.arange(10),
        "own": np.asarray([[5., 8., 2., 8., 8.]]),
        "pid": np.arange(10), "segs": [10], "P": np.asarray([5]),
        "vin": np.asarray([5, 3]), "wav": np.ones(5760, dtype=np.float32),
        "aref": "real-other-speaker", "pose_ok": 1, "pose_src": "valid",
        "voice_drawn": 1, "voice_clip": 1,
    }
    original = {k: v.copy() for k, v in student.items() if isinstance(v, np.ndarray)}
    bundle = teacher_views(student, pad_id=151670)
    mono, audio = bundle["pose_teacher"], bundle["audio_teacher"]
    for key, value in original.items():
        np.testing.assert_array_equal(student[key], value)
    for key in ("ids", "depth", "present"):
        np.testing.assert_array_equal(mono[key][1], student[key][1])
    assert (mono["ids"][0] == 151670).all()
    assert (mono["depth"][0] == -1).all()
    assert not mono["present"][0].any()
    assert mono["pdepth"] is student["pdepth"]
    assert audio["wav"] is student["wav"]
    assert audio["ids"] is student["ids"]
    assert "wav" not in mono and "vin" not in mono and "pdepth" not in audio
    assert mono["voice_clip"] == audio["voice_clip"] == student["voice_clip"]
    assert mono["segs"] == audio["segs"] == student["segs"]
    np.testing.assert_array_equal(audio["own"], [[5., 8., 2.]])


def test_dummy_and_missing_pose_do_not_create_teacher_supervision():
    student = {
        "kind": "pose", "ids": np.zeros((2, 3), dtype=np.int64),
        "depth": np.full((2, 3, 32), -1), "present": np.zeros((2, 3)),
        "pdepth": np.full((3, 32), -1), "pcb0": np.full(3, -100),
        "plight": np.zeros(0, dtype=np.int64), "own": np.zeros((1, 5)),
        "pose_ok": 0, "pose_src": "missing", "dummy": 1,
        "voice_drawn": 0, "voice_clip": 0,
    }
    bundle = teacher_views(student, 151670)
    assert all(not m["own"].any() and m["dummy"] == 1 for m in bundle.values())
    assert not bundle["pose_teacher"]["plight"].size
    assert (bundle["pose_teacher"]["pcb0"] == -100).all()


@unittest.skipUnless(DEFAULT_CHECKPOINT_CFG.is_file(), "Pinned production export is not installed")
def test_recipe_preserves_production_training_and_pads_each_epoch():
    source = json.loads(DEFAULT_CHECKPOINT_CFG.read_text())
    cfg = load_recipe(world=48, epochs=2)
    for key in ("eval_data", "lr", "clip", "dec_lr", "dec_clip", "dec_k",
                "dec_p", "adam_beta2", "w_text"):
        assert cfg[key] == source["cfg"][key], key
    assert cfg["source_loader_data"] == source["cfg"]["data"]
    assert cfg["source_loader_posedup"] == source["cfg"]["posedup"]
    assert cfg["data"] == dict(source["cfg"]["data"], pose_light_frac=0.1)
    assert cfg["posedup"] == dict(source["cfg"]["posedup"], pose_light_frac=0.1)
    assert cfg["depth_frame_sampling"]["speech"] == cfg["depth_frame_sampling"]["pose"] == 0.1
    assert cfg["depth_frame_sampling"]["shared_conversation_frames"]
    assert cfg["depth_frame_sampling"]["evaluation"] == "all_valid_then_cap"
    assert cfg["depth_frame_sampling"]["evaluation_max"] == 2048
    assert cfg["provenance"]["user_sampling_override"] == cfg["depth_frame_sampling"]
    assert cfg["eval_every"] == 0 and cfg["export_steps"] == []
    assert cfg["checkpoint_interval_seconds"] == 7200
    assert cfg["keep_latest_optimizer_state"] and cfg["save_final"]
    assert cfg["provenance"]["source_checkpoint_schedule"] == {
        "eval_every": source["cfg"]["eval_every"], "export_steps": source["cfg"]["export_steps"]}
    assert cfg["modality_weights"] == {"text": 1 / 3, "audio": 1 / 3, "pose": 1 / 3}
    assert cfg["w_cb0"] == cfg["w_pcb0"] == 0.25
    assert cfg["w_cb"] == cfg["w_pcb"] == 0.75
    assert cfg["provenance"]["user_loss_overrides"] == {
        key: cfg[key] for key in ("modality_weights", "w_cb0", "w_cb", "w_pcb0", "w_pcb")}
    for key in ("w_text", "w_cb0", "w_cb", "w_pcb0", "w_pcb"):
        assert cfg["provenance"]["source_loss_weights"][key] == source["cfg"][key]
    assert json.loads(DEFAULT_CHECKPOINT_CFG.read_text()) == source
    assert cfg["steps"] == 664 and cfg["epoch_steps"] == 332
    assert cfg["expected_train_micros"] == 31_796
    assert cfg["steps"] * cfg["world"] - cfg["expected_train_micros"] == 76
    assert cfg["warmup"] == 22 and cfg["lw_word"] == 1.0
    for key in ("sched", "dec_sched", "dec_p_sched"):
        assert cfg[key]["warmup"] == 22
        assert cfg[key]["wsd_decay_start"] == 22
        assert cfg[key]["wsd_decay_end"] == 664
    with tempfile.TemporaryDirectory() as tmp:
        changed = Path(tmp) / "train_cfg.json"
        source["cfg"]["data"]["voice_frac"] = 0.5
        changed.write_text(json.dumps(source))
        with unittest.TestCase().assertRaisesRegex(ValueError, "voice_frac"):
            load_recipe(changed)


class DataTests(unittest.TestCase):
    test_shared_conditioning = staticmethod(test_teacher_views_keep_shared_b_conditioning_without_mutating_student)
    test_masked_supervision = staticmethod(test_dummy_and_missing_pose_do_not_create_teacher_supervision)
    test_pinned_recipe = staticmethod(test_recipe_preserves_production_training_and_pads_each_epoch)


def sampling_micro(*, missing=False, dummy=False):
    # Prefix P=2; six conversation frames at delayed positions 4..9.
    pose = np.full((10, 32), -1, np.int64)
    if not missing:
        pose[4:] = np.arange(6 * 32).reshape(6, 32)
    micro = {
        "kind": "pose", "ids": np.arange(20).reshape(2, 10),
        "depth": np.arange(640).reshape(2, 10, 32), "present": np.ones((2, 10)),
        "pdepth": pose, "pcb0": np.where(pose[:, 0] >= 0, pose[:, 0], -100),
        "plight": np.arange(4, 10) if not missing else np.zeros(0, np.int64),
        "light": np.asarray([4, 7, 9]), "cb0": np.arange(10),
        "own": np.asarray([[5., 6., 3., 0. if missing else 6., 0. if missing else 6.]]),
        "pose_ok": int(not missing), "P": np.asarray([2]), "pid": np.arange(10),
        "dummy": dummy, "wav": np.arange(24, dtype=np.float32),
        "voice_clip": 1, "voice_drawn": 1,
    }
    if dummy:
        micro["own"][:] = 0
    return micro


class SharedDepthSamplingTests(unittest.TestCase):
    def test_reuses_existing_sample_and_changes_no_input_or_cb0(self):
        raw = sampling_micro()
        snapshots = {k: v.copy() for k, v in raw.items() if isinstance(v, np.ndarray)}
        sampled = shared_training_depth_frames(raw, speech_delay=2, pose_delay=2)
        again = shared_training_depth_frames(raw, speech_delay=2, pose_delay=2)
        np.testing.assert_array_equal(sampled["plight"], [4, 7, 9])
        # The original sample denotes conversation frames 0,3,5, not 2,5,7.
        np.testing.assert_array_equal(sampled["plight"] - int(raw["P"][0]) - 2, [0, 3, 5])
        np.testing.assert_array_equal(sampled["plight"], again["plight"])
        np.testing.assert_array_equal(sampled["own"], [[5., 6., 3., 6., 3.]])
        for key, value in snapshots.items():
            np.testing.assert_array_equal(raw[key], value)
            if key not in ("plight", "own"):
                self.assertIs(sampled[key], raw[key], key)
        views = teacher_views(sampled, 151670)
        self.assertIs(views["pose_teacher"]["plight"], sampled["plight"])
        self.assertIs(views["audio_teacher"]["light"], raw["light"])

    def test_delay_mapping_does_not_add_an_extra_full_delay(self):
        raw = sampling_micro()
        raw["light"] = np.asarray([3, 6, 8])
        sampled = shared_training_depth_frames(raw, speech_delay=1, pose_delay=2)
        np.testing.assert_array_equal(sampled["plight"], [4, 7, 9])

    def test_missing_pose_and_dummy_stay_zero_weight(self):
        missing = shared_training_depth_frames(sampling_micro(missing=True), speech_delay=2, pose_delay=2)
        self.assertEqual(len(missing["plight"]), 0)
        self.assertEqual(missing["own"][0, 4], 0)
        self.assertTrue((missing["pcb0"] == -100).all())
        dummy = shared_training_depth_frames(sampling_micro(dummy=True), speech_delay=2, pose_delay=2)
        self.assertFalse(dummy["own"].any())
        np.testing.assert_array_equal(dummy["plight"], dummy["light"])

    def test_evaluation_bypasses_training_selection(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        raw = sampling_micro()
        raw.update(row_key="example", P=2)
        adapter = object.__new__(ProductionData)
        adapter.cfg, adapter.eval_cfg = {"training": True}, {"evaluation": True}
        adapter.recipe = {"depth_frame_sampling": {"speech_delay": 2, "pose_delay": 2}}
        calls = []
        def conv(item, cfg, rank, audio):
            calls.append(cfg)
            return raw
        adapter._stores = SimpleNamespace(conv=conv)
        adapter.module = SimpleNamespace(D=SimpleNamespace(PAD_ID=151670),
                                         as_micro=lambda micro, dummy: dict(micro))
        with patch("finetune.duplex_distill.data.shared_training_depth_frames",
                   side_effect=AssertionError("evaluation must not resample")):
            result = adapter.batch((0, 0, 0), 0, load_audio=False, evaluation=True)
        self.assertEqual(calls, [adapter.eval_cfg])
        np.testing.assert_array_equal(result["student"]["plight"], np.arange(4, 10))
        self.assertIs(result["student"]["own"], raw["own"])


@unittest.skipUnless(DEFAULT_CHECKPOINT_CFG.is_file(), "Pinned production export is not installed")
class PlanLayoutTests(unittest.TestCase):
    def test_world56_batches_preserve_epochs_and_resume_in_update_units(self):
        import ast
        from collections import Counter
        from types import SimpleNamespace
        from finetune.duplex_distill.recipe import PRODUCTION_DIR, TRAIN_ITEMS_PER_EPOCH, EVAL_ITEMS
        source = (PRODUCTION_DIR / "data_aloe_posedup.py").read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "make_plan")
        training = [(0, r, side) for r in range(TRAIN_ITEMS_PER_EPOCH // 2) for side in (0, 1)]
        heldout = [(0, 10000 + r, side) for r in range(EVAL_ITEMS // 2) for side in (0, 1)]
        # Exercise the actual production epoch planner with metadata-only rows.
        namespace = {"V": SimpleNamespace(make_plan=lambda world, roots: {"items": training, "eval": heldout}),
                     "STORES": [], "ITEMS": "all", "EPOCH_SEED": 20261011, "np": np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "production.make_plan", "exec"), namespace)
        streams = {}
        for mbs, steps, warmup in ((1, 568, 19), (2, 284, 9)):
            cfg = load_recipe(world=56, micro_batch_size=mbs)
            self.assertEqual((cfg["steps"], cfg["warmup"], cfg["global_batch_size"]), (steps, warmup, 56 * mbs))
            self.assertEqual(cfg["gradient_accumulation"], 1)
            data = object.__new__(ProductionData)
            data.recipe = cfg
            data.module = SimpleNamespace(make_plan=namespace["make_plan"],
                plan_digest=lambda plan: ("metadata-only-test", "d1a3e3721328da0a"))
            data.batch = lambda item, rank, **kw: {"student": {"item": item, "dummy": kw["dummy"]}}
            plan = data.make_plan()
            self.assertEqual((plan["steps"], plan["world"], plan["logical_world"]), (steps, 56, 56 * mbs))
            self.assertEqual((plan["n_train_micros"], plan["n_dummy"]), (31796, 12))
            self.assertEqual(plan["micro_steps_per_rank"], 568)
            capacity = plan["epoch_steps"] * plan["logical_world"]
            for epoch in range(2):
                real = [x for x in plan["items"][epoch * capacity:(epoch + 1) * capacity] if x is not None]
                self.assertEqual(Counter(real), Counter(training))
            datasets = [data.rank_dataset(plan, rank, 56) for rank in range(56)]
            self.assertEqual(Counter(x for ds in datasets for x in ds.items if x is not None), Counter(training * 2))
            self.assertEqual(sum(x is None for ds in datasets for x in ds.items), 12)
            self.assertEqual(Counter(x for ds in datasets for x in ds.eval_items), Counter(heldout))
            streams[mbs] = [ds.items for ds in datasets]
            # Both mixed and all-dummy batches occur at epoch tails. Resume
            # skips whole optimizer updates, never just one row of a pair.
            for start in (0, 3, plan["epoch_steps"], steps - 1, steps):
                resumed = data.rank_dataset(plan, 55, 56, start=start)
                self.assertEqual(len(resumed), (steps - start) * mbs)
                self.assertEqual(resumed.items[start * mbs:], datasets[55].items[start * mbs:])
                if len(resumed):
                    item = resumed[0]["student"]
                    self.assertEqual((item["step"], item["slot"]), (start, 0))
                    self.assertEqual(item["dummy"], datasets[55].items[start * mbs] is None)
            with self.assertRaisesRegex(ValueError, "Resume"):
                data.rank_dataset(plan, 0, 56, start=0.5)
        self.assertEqual(streams[1], streams[2])


@unittest.skipUnless((DEFAULT_DATA_ROOT / "pose/POSE_INDEX.json").is_file()
                     and DEFAULT_CHECKPOINT_CFG.is_file(), "Production rows are not installed on this node")
class RealProductionSamplingTests(unittest.TestCase):
    def test_full_short_missing_and_eval_rows_keep_original_inputs(self):
        data = ProductionData()
        plan = data.make_plan()
        index = data.module.pose_index()["rows"]
        cases = {}
        for item in plan["items0"]:
            k, row, sw = item
            entry = index[str(k)][str(row)]
            ok = bool(entry["sides"]["user" if sw else "assistant"]["ok"])
            key = "missing" if not ok else ("full" if entry["n"] == 7500 else "short")
            cases.setdefault(key, item)
            if len(cases) == 3:
                break
        self.assertEqual(set(cases), {"full", "short", "missing"})
        for case, item in cases.items():
            for evaluation in (False, True):
                cfg = data.eval_cfg if evaluation else data.cfg
                raw = data.module.as_micro(data.stores.conv(item, cfg, 0, audio=False))
                result = data.batch(item, 0, load_audio=False, evaluation=evaluation)["student"]
                for key, value in raw.items():
                    if not evaluation and key in ("plight", "own"):
                        continue
                    if isinstance(value, np.ndarray):
                        np.testing.assert_array_equal(result[key], value, err_msg=f"{case}/{evaluation}/{key}")
                    else:
                        self.assertEqual(result[key], value, f"{case}/{evaluation}/{key}")
                if not evaluation:
                    expected = raw["light"][raw["pcb0"][raw["light"]] >= 0]
                    np.testing.assert_array_equal(result["plight"], expected)
                    np.testing.assert_array_equal(result["own"][:, :4], raw["own"][:, :4])
                    self.assertEqual(result["own"][0, 4], len(expected))
                    again = data.batch(item, 7, load_audio=False)["student"]
                    np.testing.assert_array_equal(result["plight"], again["plight"])
                    if case == "full":
                        self.assertEqual(len(raw["plight"]), 7500)
                        self.assertTrue(500 < len(result["plight"]) < 1000)


if __name__ == "__main__":
    unittest.main()
