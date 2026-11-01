"""Actual delay, information-boundary, available-clip and teacher-view checks."""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.data import (
    OFFSETS, ProductionRows, UnavailableRow, conditioning_segment, prepare_rollout_seed,
    replay_from_rollout, sampling_masks, invert_row_map, _RawMappedRows, ROW_IDENTITY_COLUMNS,
)


PAD = 151670


def raw_row(n=40, p=8):
    length = p + n + 2
    ids = np.full((2, length), PAD, np.int64)
    ids[1, p:p+n] = 50 + np.arange(n)
    depth = np.full((2, length, 32), -1, np.int64)
    depth[1, 3:6] = OFFSETS + 7  # Actual nonempty voice prefix.
    depth[1, p+2:] = OFFSETS + np.arange(n)[:, None]
    pose = np.full((length, 32), -1, np.int64)
    pose[p+2:] = OFFSETS + np.arange(n)[:, None] + 100
    return {"kind": "pose", "ids": ids, "depth": depth, "pdepth": pose,
            "present": np.ones((2, length), np.float32), "P": np.asarray([p]),
            "pid": np.arange(length), "segs": [length], "vin": np.asarray([p, n]),
            "wav": np.arange(n*1920, dtype=np.float32), "aref": json.dumps({"n": n}),
            "voice_clip": 1, "voice_drawn": 1, "row_key": "row", "sel": np.arange(length-1),
            "y": ids[1, 1:].copy(), "cb0": depth[1, :, 0].copy(), "plight": np.arange(length)}


def generated(seed):
    text = seed["ids"][1].copy()
    audio = seed["depth"][1].copy()
    pose = seed["pdepth"].copy()
    text[seed["sampled_masks"]["text"]] = 101
    audio[seed["sampled_masks"]["audio"]] = OFFSETS + 55
    pose[seed["sampled_masks"]["pose"]] = OFFSETS + 66
    return text, audio, pose


class DataTests(unittest.TestCase):
    def test_empty_clip_policy_preserves_native_prefix_and_strict_prefill(self):
        raw = raw_row(p=5)
        raw["voice_clip"] = 0
        raw["ids"][1, :5] = [151644, 8948, 198, 151645, 198]
        raw["depth"][1, :7] = -1
        seed = prepare_rollout_seed(raw, pad_id=PAD, missing_clip_policy="empty")
        self.assertEqual(seed["voice_clip"], 0)
        self.assertEqual(seed["voice_drawn"], 1)
        self.assertEqual(np.flatnonzero(seed["sampled_masks"]["text"])[0], 30)
        self.assertEqual(np.flatnonzero(seed["sampled_masks"]["audio"])[0], 32)
        bundle = replay_from_rollout(seed, *generated(seed))
        self.assertIs(bundle["student"]["wav"], bundle["audio_teacher"]["wav"])
        for view in bundle.values():
            np.testing.assert_array_equal(view["ids"][1, :5], raw["ids"][1, :5])
            self.assertTrue((view["depth"][1, :5] == -1).all())
        raw["ids"][1, 0] = PAD
        with self.assertRaisesRegex(ValueError, "native empty system prefix"):
            prepare_rollout_seed(raw, pad_id=PAD, missing_clip_policy="empty")
        raw["ids"][1, 0] = 151644
        raw["pdepth"][7] = -1
        with self.assertRaisesRegex(UnavailableRow, "ground-truth pose"):
            prepare_rollout_seed(raw, pad_id=PAD, missing_clip_policy="empty")

    def test_audio_only_reference_suppresses_words_without_changing_source(self):
        clip = {"w0": 9, "L": 38, "blank_lane": False, "src_key": "clip"}
        segment = conditioning_segment(clip, "audio")
        self.assertTrue(segment["blank_lane"])
        self.assertFalse(clip["blank_lane"])
        self.assertEqual(segment["kind"], "speech")
        self.assertEqual(segment["track"], 1)
        self.assertEqual(segment["L"], clip["L"])

    def test_physical_prefill_and_model_delay_boundaries(self):
        masks = sampling_masks(8, 40)
        np.testing.assert_array_equal(np.flatnonzero(masks["text"]), np.arange(33, 48))
        np.testing.assert_array_equal(np.flatnonzero(masks["audio"]), np.arange(35, 50))
        np.testing.assert_array_equal(masks["pose"], masks["audio"])
        self.assertFalse(masks["text"][48:].any())

    def test_seed_erases_future_b_and_source_labels_without_mutation(self):
        raw = raw_row()
        originals = {key: value.copy() for key, value in raw.items() if isinstance(value, np.ndarray)}
        seed = prepare_rollout_seed(raw, pad_id=PAD)
        self.assertTrue((seed["ids"][1, 33:] == PAD).all())
        self.assertTrue((seed["depth"][1, 35:] == -1).all())
        self.assertTrue((seed["pdepth"][35:] == -1).all())
        np.testing.assert_array_equal(seed["depth"][1, :35], raw["depth"][1, :35])
        np.testing.assert_array_equal(seed["pdepth"][:35], raw["pdepth"][:35])
        for key in ("sel", "y", "cb0", "pcb0", "light", "plight"):
            self.assertNotIn(key, seed)
        for key, value in originals.items():
            np.testing.assert_array_equal(raw[key], value)
        self.assertTrue((seed["ids"][0] == PAD).all())
        self.assertTrue((seed["depth"][0] == -1).all())

    def test_replay_uses_generated_targets_full_depth_and_shared_conditioning(self):
        seed = prepare_rollout_seed(raw_row(), pad_id=PAD)
        arrays = generated(seed)
        bundle = replay_from_rollout(seed, *arrays)
        student, audio, pose = (bundle[key] for key in ("student", "audio_teacher", "pose_teacher"))
        np.testing.assert_array_equal(student["sel"], np.arange(32, 47))
        np.testing.assert_array_equal(student["y"], np.full(15, 101))
        np.testing.assert_array_equal(student["light"], np.arange(35, 50))
        np.testing.assert_array_equal(student["plight"], np.arange(35, 50))
        np.testing.assert_array_equal(student["own"], [[15, 15, 15, 15, 15]])
        self.assertTrue((student["cb0"][:35] == -100).all())
        self.assertIs(audio["wav"], student["wav"])
        self.assertIs(audio["ids"], student["ids"])
        self.assertIs(pose["pdepth"], student["pdepth"])
        np.testing.assert_array_equal(pose["depth"][1], student["depth"][1])
        self.assertNotIn("wav", pose)
        self.assertNotIn("pdepth", audio)

    def test_replay_rejects_changed_prefill_or_missing_sample(self):
        seed = prepare_rollout_seed(raw_row(), pad_id=PAD)
        for kind in range(3):
            values = list(generated(seed))
            values[kind][12] += 1
            with self.assertRaisesRegex(ValueError, "changed forced"):
                replay_from_rollout(seed, *values)
        text, audio, pose = generated(seed)
        audio[35, 14] = -1
        with self.assertRaisesRegex(ValueError, "sampled actions"):
            replay_from_rollout(seed, text, audio, pose)

    def test_nonempty_clip_and_complete_pose_prefill_are_required(self):
        for mutate in (lambda r: r.update(voice_clip=0),
                       lambda r: r["depth"][1, :8].fill(-1),
                       lambda r: r["pdepth"][10:35].fill(-1)):
            row = raw_row()
            mutate(row)
            with self.assertRaises(UnavailableRow):
                prepare_rollout_seed(row, pad_id=PAD)
        with self.assertRaises(UnavailableRow):
            prepare_rollout_seed(raw_row(n=25), pad_id=PAD)

    def test_future_ground_truth_cannot_influence_seed(self):
        a, b = raw_row(), raw_row()
        b["ids"][1, 33:] = 900
        b["depth"][1, 35:] = OFFSETS + 333
        b["pdepth"][35:] = OFFSETS + 444
        aa, bb = (prepare_rollout_seed(row, pad_id=PAD) for row in (a, b))
        for key in ("ids", "depth", "pdepth", "present", "wav"):
            np.testing.assert_array_equal(aa[key], bb[key])


class ConfigurationTests(unittest.TestCase):
    def test_roundtrip_and_custom_export_data_defaults(self):
        config = TrainingConfig()
        self.assertEqual(TrainingConfig.from_dict(config.to_dict()), config)
        self.assertEqual(config.lookback_frames, 8)
        self.assertEqual(config.learning_rate, 3e-6)
        self.assertEqual(config.conditioning_mode, "audio")
        self.assertEqual(config.missing_clip_policy, "empty")
        self.assertTrue(config.exclude_invalid_pose_frame0)
        self.assertEqual(config.audio_teacher_gain_policy, "require_fp32")
        self.assertEqual(config.codebook_weights[0], 0.5)
        self.assertTrue(all(x == 0.5 / 31 for x in config.codebook_weights[1:]))
        for metadata in ("train_cfg.json", "meta.json"):
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as tmp:
                Path(tmp, metadata).write_text(json.dumps({"run": "arbitrary", "step": 999,
                    "cfg": {"data": {"clip_min": 20, "crop_conv": True, "light_frac": 0.1}}}))
                resolved = replace(config, student=tmp).checkpoint_data_config()
                self.assertEqual(resolved["clip_min"], 20)
                self.assertFalse(resolved["crop_conv"])
                self.assertEqual(resolved["light_frac"], 1)
                self.assertEqual(resolved["pose_light_frac"], 1)

    def test_explicit_policy_validation(self):
        for kwargs in ({"temperature": 0.8}, {"sampler": "probit"},
                       {"missing_clip_policy": "invented"}, {"data_config": {"crop_conv": True}},
                       {"window_mode": "including_current", "lookback_frames": 0},
                       {"max_row_frames": 20}, {"modality_weights": {"pose": 1}},
                       {"exclude_invalid_pose_frame0": 1}, {"audio_teacher_gain_policy": "guessed"}):
            with self.assertRaises(ValueError):
                TrainingConfig(**kwargs).validate()
        with self.assertRaisesRegex(ValueError, "explicitly select"):
            TrainingConfig.from_dict({"lookback_frames": 8})
        self.assertEqual(TrainingConfig(data_variant="raw").validate().data_variant, "raw")
        with self.assertRaisesRegex(ValueError, "data_variant"):
            TrainingConfig(data_variant="guessed").validate()


class RawMappingTests(unittest.TestCase):
    def test_bijective_map_inversion_and_rejections(self):
        np.testing.assert_array_equal(invert_row_map({"4": 0, "1": 1}, 2, 5), [4, 1])
        for bad in ({"4": 0, "1": 0}, {"5": 0, "1": 1}, {"4": 0}, {"01": 0, "1": 1}):
            with self.assertRaisesRegex(ValueError, "bijectively"):
                invert_row_map(bad, 2, 5)

    def test_codes_only_replaced_and_bad_masks_or_identity_rejected(self):
        def rows(offset):
            return [{"sentinel": f"metadata-{i}", "codes_user": np.full((32, 3), i + offset, np.int16),
                     "codes_assistant": np.full((32, 3), i + offset + 5, np.int16)} for i in range(3)]
        original = rows(100)
        selected = [rows(0)[2], rows(0)[0]]
        raw_meta = {key: [f"{key}-{i}" for i in range(3)] for key in ROW_IDENTITY_COLUMNS}
        raw_meta["row"] = [0, 1, 2]
        selected_meta = {key: [values[2], values[0]] for key, values in raw_meta.items()}
        selected_meta["row"] = [0, 1]
        def table(path, columns):
            value = selected_meta if str(path).startswith("selected") else raw_meta
            return SimpleNamespace(to_pydict=lambda: value)
        module = SimpleNamespace(D=SimpleNamespace(R3=SimpleNamespace(Rows=lambda node: original)))
        selected_node = {"length": np.asarray([3, 3]), "flat_root": "selected"}
        raw_node = {"length": np.asarray([3, 3, 3]), "flat_root": "raw"}
        mapping = {"old_rows": 3, "new_of_old": {"2": 0, "0": 1}}
        with patch("pyarrow.parquet.read_table", side_effect=table):
            wrapped = _RawMappedRows(module, selected_node, selected, raw_node, mapping)
            out = wrapped[0]
            self.assertEqual(out["sentinel"], selected[0]["sentinel"])
            self.assertIs(out["codes_user"], original[2]["codes_user"])
            self.assertIs(out["codes_assistant"], original[2]["codes_assistant"])
            self.assertEqual(int(selected[0]["codes_user"][0, 0]), 2)
            original[2]["codes_user"][0, 0] = -1
            with self.assertRaisesRegex(ValueError, "valid-frame"):
                wrapped[0]
            selected_meta["user_speaker_dir"][0] = "wrong speaker"
            with self.assertRaisesRegex(ValueError, "user_speaker_dir"):
                _RawMappedRows(module, selected_node, selected, raw_node, mapping)


@unittest.skipUnless(Path("/home/param/datasets/aloe-duplex/pose/POSE_INDEX.json").is_file(), "Production data not installed")
class ProductionDataTests(unittest.TestCase):
    def test_unavailable_clip_uses_native_empty_prefix_and_shared_teacher_history(self):
        for mode in ("audio", "voice_pose"):
            with self.subTest(mode=mode):
                rows = ProductionRows(TrainingConfig(data_variant="raw", conditioning_mode=mode))
                item = (0, 1837, 0)
                seed = rows.load(item, load_audio=False)
                with rows.arena.enter() as module:
                    row, clip, _, _ = rows._source(module, item, load_pose=False)
                    _, _, siblings = rows._open(module, item[0])
                    native, provenance = module.A2.prefix_v2(row, item[2], siblings, rows.cfg)
                self.assertIsNone(clip)
                self.assertEqual(provenance["voice_clip"], 0)
                self.assertEqual(int(seed["P"][0]), 5)
                self.assertEqual(seed["voice_clip"], 0)
                self.assertEqual(seed["voice_drawn"], 1)
                for key in ("ids", "depth", "present"):
                    np.testing.assert_array_equal(seed[key][:, :5], native[key])
                self.assertTrue((seed["pdepth"][:5] == -1).all())
                self.assertFalse(seed["conditioning"]["available"])
                self.assertFalse(seed["conditioning"]["reference_text"])
                self.assertFalse(seed["conditioning"]["reference_pose"])
                self.assertEqual(seed["conditioning"]["clip_frames"], 0)
                self.assertIsNone(seed["conditioning"]["source_row"])
                self.assertEqual(int(np.flatnonzero(seed["sampled_masks"]["text"])[0]), 30)
                self.assertEqual(int(np.flatnonzero(seed["sampled_masks"]["pose"])[0]), 32)
                bundle = replay_from_rollout(seed, *generated(seed))
                for view in bundle.values():
                    np.testing.assert_array_equal(view["ids"][1], bundle["student"]["ids"][1])
                    np.testing.assert_array_equal(view["depth"][1], bundle["student"]["depth"][1])
                    self.assertEqual(view["voice_clip"], 0)
                self.assertEqual(bundle["audio_teacher"]["aref"], seed["aref"])
                self.assertNotIn("aref", bundle["pose_teacher"])
        for policy in ("skip", "error"):
            with self.subTest(policy=policy):
                rows = ProductionRows(TrainingConfig(data_variant="raw", missing_clip_policy=policy))
                with rows.arena.enter() as module, self.assertRaisesRegex(UnavailableRow, "No supported later-sibling"):
                    rows._source(module, item, load_pose=False)

    def test_source_frame_zero_filter_excludes_only_matching_bad_lanes(self):
        filtered = ProductionRows(TrainingConfig(data_variant="raw"))
        unfiltered = ProductionRows(TrainingConfig(data_variant="raw", exclude_invalid_pose_frame0=False))
        expected = [
            ((3, 1160, 0), "9c9ae2c1-e392-4ee2-bbe6-ddb879841418/seg00", "SPEAKER_00"),
            ((3, 1416, 1), "bffa5f01-8124-4a2c-82b1-59ddd9fce724/seg01", "SPEAKER_01"),
        ]
        for item, conversation, speaker in expected:
            with self.subTest(item=item), filtered.arena.enter() as module:
                with self.assertRaisesRegex(UnavailableRow, "Source frame-zero pose normalization"):
                    filtered._source(module, item, load_pose=False)
                entry = filtered._frame0_drops[item]
                self.assertEqual(entry["conversation"], conversation)
                self.assertEqual(entry["speaker"], speaker)
            with unfiltered.arena.enter() as module:
                unfiltered._source(module, item, load_pose=False)
        for swap in (0, 1):
            with filtered.arena.enter() as module:
                filtered._source(module, (0, 0, swap), load_pose=False)
        with filtered.arena.enter() as module:
            for item in filtered._frame0_drops:
                with self.subTest(invalid_frame0=item), self.assertRaisesRegex(UnavailableRow, "Source frame-zero"):
                    filtered._source(module, item, load_pose=False)
            with self.assertRaisesRegex(UnavailableRow, "lacks B pose"):
                filtered._source(module, (0, 1841, 1), load_pose=False)
        meta = filtered.metadata["pose_frame0_filter"]
        self.assertTrue(meta["enabled"])
        self.assertFalse(unfiltered.metadata["pose_frame0_filter"]["enabled"])
        self.assertEqual(meta["source_drop_count"], 12)
        self.assertEqual(meta["source_annotation_release"], "eb7f3745dc916e6876fd")
        self.assertEqual(len(meta["manifest_sha256"]), 64)

    def test_raw_and_samsep_share_identity_but_use_selected_reference_and_prefill_codes(self):
        samsep = ProductionRows(TrainingConfig(data_variant="samsep"))
        raw = ProductionRows(TrainingConfig(data_variant="raw"))
        for swap in (0, 1):
            with self.subTest(swap=swap):
                a = samsep.load((0, 0, swap), load_audio=False)
                b = raw.load((0, 0, swap), load_audio=False)
                p = int(a["P"][0])
                self.assertEqual(a["aref"], b["aref"])
                for key in ("ids", "pdepth", "present", "pid", "P", "vin"):
                    np.testing.assert_array_equal(a[key], b[key])
                for key in ("source_row", "source_start", "clip_frames", "prefix_frames"):
                    self.assertEqual(a["conditioning"][key], b["conditioning"][key])
                self.assertFalse(np.array_equal(a["depth"][1, :p], b["depth"][1, :p]))
                self.assertFalse(np.array_equal(a["depth"][1, p:p+27], b["depth"][1, p:p+27]))
                np.testing.assert_array_equal(a["depth"] >= 0, b["depth"] >= 0)
                np.testing.assert_array_equal(a["depth"][1, p+27:], b["depth"][1, p+27:])
        self.assertEqual(raw.metadata["variant"], "raw")
        self.assertEqual(len(raw.metadata["raw_store_digests"]), 4)
        self.assertEqual(len(raw.metadata["raw_row_map_sha256"]), 64)
        self.assertEqual(raw.metadata["selected_store_digests"], samsep.metadata["selected_store_digests"])

    def test_real_row_shared_clip_and_module_environment_restoration(self):
        before_env = dict(os.environ)
        before_module = sys.modules.get("data_aloe_posedup")
        for mode in ("audio", "voice", "voice_pose"):
            rows = ProductionRows(TrainingConfig(conditioning_mode=mode))
            seed = rows.load((0, 0, 0), load_audio=False)
            p, n = int(seed["P"][0]), seed["conversation_frames"]
            self.assertEqual(seed["ids"].shape[1], p + n + 2)
            self.assertEqual(int(seed["sampled_masks"]["pose"].sum()), n - 25)
            self.assertGreater(seed["conditioning"]["clip_frames"], 0)
            if mode == "voice_pose":
                np.testing.assert_array_equal(seed["pdepth"][:p, 0] >= 0, seed["depth"][1, :p, 0] >= 0)
            elif mode == "audio":
                self.assertFalse(seed["conditioning"]["reference_text"])
                self.assertFalse(seed["conditioning"]["reference_pose"])
                self.assertTrue((seed["pdepth"][:p] == -1).all())
                # Native system header, SOSB and EOSB survive; the speech span
                # carries only PAD, including its codec-delay tail positions.
                lane = seed["ids"][1, :p]
                np.testing.assert_array_equal(lane[:3], [151644, 8948, 198])
                self.assertEqual(lane[3], 151678)
                self.assertEqual(lane[-3], 151679)
                self.assertTrue((lane[4:-3] == PAD).all())
                np.testing.assert_array_equal(lane[-2:], [151645, 198])
            self.assertEqual(dict(os.environ), before_env)
            self.assertIs(sys.modules.get("data_aloe_posedup"), before_module)


if __name__ == "__main__":
    unittest.main()
