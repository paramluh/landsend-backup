"""Small CPU-only regression checks; no source corpus scan or network calls.

Run: python scripts/reference_bank_rebuild/video_selftest.py
"""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

import video_scan as scan
import video_pipeline as pipeline
from hf_sources import VIDEO_PREFIX

ARCHIVE = Path("/mnt/paramfs/vansh/richmond-7/reference-bank/code")


def reference_functions():
    tree = ast.parse((ARCHIVE / "s2bundle/select_s2.py").read_text())
    wanted = {"sfeat", "design", "face_box", "crop"}
    subset = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted], type_ignores=[])
    namespace = dict(np=np, W=512, HH=960, CW=256, CH=480, PAD=.35, GROW=1.1)
    exec(compile(subset, "archived-selection-functions", "exec"), namespace)
    return namespace


def candidate(seconds=0, track="r/speaker_1", **overrides):
    keypoints = np.zeros((133, 3), np.float32)
    keypoints[23:91, 0] = np.linspace(200, 300, 68)
    keypoints[23:91, 1] = np.linspace(200, 320, 68)
    keypoints[23:91, 2] = 1
    row = dict(uuid="u", bank="bank", track=track, recording_id=track.split("/")[0],
               speaker=1, session_id="s", frame_index=round(seconds * 25), time_s=seconds,
               source=f"hf://buckets/test/source/{track}/full.mp4#frame={round(seconds * 25)}",
               kp=keypoints.reshape(-1).tolist(), thumb=[seconds * 2 % 256] * 1920,
               mp_ok=True, pose_valid=True, yaw=0, blinkL=.1, blinkR=.1, blur=100,
               hand_in_face=0, oof=0, bx0=200, by0=200, bx1=300, by1=320, n_hands=0)
    row.update(overrides)
    return row


WEIGHTS = np.r_[np.zeros(12), .5]


class TargetPlanningTests(unittest.TestCase):
    @staticmethod
    def track(name, actor, chunks, **extra):
        return dict(track=f"{name}/speaker_1", recording_id=name, speaker=1,
                    session_id=f"session-{name}", uuid=actor, usable_chunks=list(range(chunks)), **extra)

    @staticmethod
    def video(track, version="original"):
        return {"path": f"{VIDEO_PREFIX}/{track['recording_id']}/session/speaker_1/video/{version}/full.mp4",
                "size": 1024, "xet_hash": "test-content-hash"}

    def plan(self, tracks, index, upstream=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sources"
            selection = root / "upstream.parquet"
            (upstream if upstream is not None else pd.DataFrame()).to_parquet(selection)
            index_path = root / "index.json"
            index_path.write_text(json.dumps(index))
            corpus = SimpleNamespace(tracks=tracks, ranges_eval={}, ranges_blackbands={},
                                     ranges_lighting={}, ranges_excess={}, source_hashes={})
            args = SimpleNamespace(csv_root=root, blackbands_csv=root / "bb.csv", upstream=selection,
                                   video_index=index_path, source_root=source, output_dir=root, allow_incomplete=True)
            with patch.object(pipeline, "load_corpus", return_value=corpus), redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.targets(args), 0)
            self.assertFalse(source.exists(), "Target planning must not download videos")
            return {name: json.loads((root / f"{name}.json").read_text()) for name in
                    ("tracks", "targets-all", "targets-missing", "targets-unavailable", "targets-actors", "targets-summary")}

    def test_unavailable_top_track_falls_back_before_two_track_limit(self):
        tracks = [self.track(name, "partial", chunks) for name, chunks in
                  [("top-missing", 5), ("second", 4), ("third", 3), ("unused", 2)]]
        tracks.append(self.track("full-actor", "full", 20))
        index = {row["track"]: [self.video(row)] for row in tracks[1:3]}
        upstream = pd.DataFrame([dict(uuid="full", property=prop) for prop in range(1, 10)])
        result = self.plan(tracks, index, upstream)
        self.assertEqual([row["track"] for row in result["tracks"]], ["second/speaker_1", "third/speaker_1"])
        self.assertEqual([row["track"] for row in result["targets-unavailable"]], ["top-missing/speaker_1"])
        self.assertEqual(result["targets-missing"], [])
        self.assertTrue(result["targets-summary"]["complete"])
        self.assertEqual(result["targets-summary"]["actors_by_available_track_count"], {"0": 0, "1": 0, "2": 1})
        evidence = result["targets-actors"][0]
        self.assertEqual([row["status"] for row in evidence["ranked_tracks"]],
                         ["unavailable", "selected", "selected", "not-examined-two-sources-selected"])
        self.assertFalse(evidence["ranking_exhausted"])
        self.assertFalse(evidence["source_limited"])

    def test_one_eligible_video_is_complete_but_unavailable_alternatives_are_source_limited(self):
        only = self.track("only", "single", 1)
        available = self.track("available", "limited", 2)
        missing = self.track("missing", "limited", 3)
        absent = self.track("absent", "none", 1)
        index = {row["track"]: [self.video(row)] for row in [only, available]}
        single = self.plan([only], index)
        self.assertTrue(single["targets-summary"]["complete"])
        self.assertTrue(single["targets-actors"][0]["ranking_exhausted"])
        result = self.plan([missing, available, only, absent], index)
        self.assertFalse(result["targets-summary"]["complete"])
        self.assertEqual(result["targets-summary"]["source_limited_actors"], 2)
        self.assertEqual(result["targets-summary"]["actors_by_available_track_count"], {"0": 1, "1": 2, "2": 0})
        evidence = {row["uuid"]: row for row in result["targets-actors"]}
        self.assertTrue(all(row["ranking_exhausted"] for row in evidence.values()))
        self.assertFalse(evidence["single"]["source_limited"])
        self.assertEqual(evidence["limited"]["selected_tracks"], ["available/speaker_1"])

    def test_ambiguous_video_is_not_picked_and_explicit_original_source_is_respected(self):
        ambiguous = self.track("ambiguous", "u", 3)
        fallback = self.track("fallback", "u", 2)
        original, other = self.video(ambiguous), self.video(ambiguous, "other")
        index = {ambiguous["track"]: [other, original], fallback["track"]: [self.video(fallback)]}
        result = self.plan([ambiguous, fallback], index)
        self.assertEqual([row["track"] for row in result["tracks"]], [fallback["track"]])
        self.assertIn("found 2", result["targets-unavailable"][0]["error"])
        ambiguous["original_blob"] = original["path"]
        resolved = self.plan([ambiguous, fallback], index)
        self.assertEqual(resolved["tracks"][0]["video_source_metadata"], original)
        self.assertTrue(resolved["tracks"][0]["source"].endswith(original["path"]))
        self.assertTrue(resolved["targets-summary"]["complete"])


class SelectionTests(unittest.TestCase):
    def test_archived_feature_and_crop_equivalence(self):
        reference = reference_functions()
        random = np.random.default_rng(16)
        for _ in range(300):
            keypoints = random.uniform(0, 1, (133, 3)).astype(np.float32)
            keypoints[:, 0] = keypoints[:, 0] * random.uniform(40, 350) + random.uniform(-10, 180)
            keypoints[:, 1] = keypoints[:, 1] * random.uniform(80, 650) + random.uniform(-20, 300)
            np.testing.assert_allclose(scan.selector_features(keypoints), reference["sfeat"](keypoints), rtol=1e-6)
            box = reference["face_box"](keypoints)
            expected = reference["crop"](box) if box is not None else None
            self.assertEqual(scan.face_crop_box(keypoints), expected)
        features = random.normal(size=(40, 7))
        for sign in (-1, 1):
            np.testing.assert_array_equal(scan._design(features, sign), reference["design"](features, sign))

    def test_track_choice_requires_empty_slot_and_uses_best_two(self):
        tracks = [dict(track=name, uuid=actor, usable_chunks=list(range(count)))
                  for name, actor, count in [("a", "full", 5), ("b", "partial", 1), ("c", "partial", 4), ("d", "partial", 3)]]
        upstream = pd.DataFrame([dict(uuid="full", property=p) for p in range(1, 10)])
        self.assertEqual([row["track"] for row in scan.choose_scan_tracks(tracks, upstream)], ["c", "d"])

    def test_contact_needs_three_valid_consecutive_frames(self):
        points = np.zeros((5, 133, 3), np.float32)
        for start in (91, 112):
            points[:, start:start + 21] = [100, 200, 1]
            points[:, start + 5, 0] = 110
            points[:, start + 17, 0] = 90
        self.assertTrue(scan.contact_gate(points, np.ones(5, bool), 2))
        self.assertFalse(scan.contact_gate(points, np.array([1, 1, 0, 1, 1], bool), 2))
        self.assertFalse(scan.contact_gate(points, np.ones(5, bool), 8))

    def test_excluded_boundaries_and_allowlist(self):
        track = dict(excluded_ranges=[(295, 605)], allowed_ranges=[(0, 900)])
        self.assertFalse(scan.timestamp_allowed(track, 295))
        self.assertFalse(scan.timestamp_allowed(track, 605))
        self.assertTrue(scan.timestamp_allowed(track, 605.04))
        self.assertFalse(scan.timestamp_allowed(track, 901))

    def test_all_nine_strict_slot_rules(self):
        hands = dict(yaw=60, n_hands=2, h0_inframe=21, h1_inframe=21,
                     h0_sharp=80, h1_sharp=80, h0_ext=3, h1_ext=3,
                     hh_dist=.2, h0_face_ov=0, h1_face_ov=0)
        rows = [candidate(0), candidate(10, yaw=-30), candidate(20, yaw=30),
                candidate(30, yaw=12, t_ok=True, up=1., lo=1., span=.4, jaw=.3),
                candidate(40, **hands),
                candidate(50, **{**hands, "h0_sharp": 60, "h1_sharp": 60, "gate": True}),
                candidate(60, yaw=60, n_hands=1, h0_sharp=80, h0_target="hair"),
                candidate(70, yaw=60, n_hands=1, h0_sharp=80, h0_target="clothing", h0_face_ov=0)]
        scored = scan.score_candidates(pd.DataFrame(rows), proxy_weights=WEIGHTS)
        for index, prop in enumerate(range(1, 9)):
            self.assertTrue(scored.loc[index, f"pass{prop}"], prop)
        self.assertTrue(scored.loc[7, "pass9"])
        failures = [dict(row, blur=59.99) for row in rows]
        failed = scan.score_candidates(pd.DataFrame(failures), proxy_weights=WEIGHTS)
        self.assertFalse(failed[[f"pass{p}" for p in range(1, 10)]].any().any())
        hand_failed = scan.score_candidates(pd.DataFrame([candidate(hand_in_face=1)]), proxy_weights=WEIGHTS)
        self.assertFalse(hand_failed.pass1.iloc[0])

    def test_pose_and_reserved_flags_fail_closed(self):
        for flag in ("pose_valid", "in_eval", "in_bb", "in_light"):
            row = candidate(**{flag: flag != "pose_valid"})
            scored = scan.score_candidates(pd.DataFrame([row]), proxy_weights=WEIGHTS)
            self.assertFalse(scored[[f"pass{p}" for p in range(1, 10)]].any().any())

    def test_temporal_and_thumbnail_dedup_with_upstream(self):
        upstream = pd.DataFrame([dict(uuid="u", recording_id="r", speaker=1, property=1, time_s=0,
                                     thumb=[200] * 1920)])
        rows = [candidate(9.96, blur=140), candidate(10, blur=130),
                candidate(20, blur=120, thumb=[20] * 1920), candidate(30, blur=110)]
        selected = scan.select_scanned(pd.DataFrame(rows), upstream, proxy_weights=WEIGHTS)
        # 9.96 rejected by upstream distance; 20s rejected as same thumbnail as 10s.
        self.assertEqual(selected.time_s.tolist(), [10, 30])
        self.assertEqual(selected["index"].tolist(), [1, 2])

    def test_track_round_robin_uses_one_per_track_before_seconds(self):
        upstream = pd.DataFrame([dict(uuid="u", recording_id="old", speaker=1, property=1,
                                     time_s=i * 20, thumb=[200] * 1920) for i in range(18)])
        rows = [candidate(0, track="a/speaker_1", blur=200), candidate(20, track="a/speaker_1", blur=190),
                candidate(0, track="b/speaker_1", blur=100, thumb=[80] * 1920),
                candidate(20, track="b/speaker_1", blur=90, thumb=[120] * 1920)]
        selected = scan.select_scanned(pd.DataFrame(rows), upstream, proxy_weights=WEIGHTS)
        self.assertEqual(selected.track.tolist(), ["a/speaker_1", "b/speaker_1"])
        self.assertEqual(selected["index"].tolist(), [18, 19])

    def test_hair_clothing_share_dedup_and_frames_are_not_reused(self):
        rows = [candidate(0, yaw=60, n_hands=1, h0_target="hair", h0_sharp=100),
                candidate(5, yaw=60, n_hands=1, h0_target="clothing", h0_sharp=90),
                candidate(20, yaw=60, n_hands=1, h0_target="clothing", h0_sharp=80)]
        selected = scan.select_scanned(pd.DataFrame(rows), pd.DataFrame(), proxy_weights=WEIGHTS)
        self.assertEqual(selected["property"].tolist(), [7, 8])
        self.assertEqual(selected.time_s.tolist(), [0, 20])
        self.assertFalse(selected.duplicated(["track", "frame_index"]).any())

    def test_scan_rejects_upstream_and_cross_track_visual_duplicates(self):
        upstream = pd.DataFrame([dict(uuid='u', recording_id='old', speaker=1, property=1,
                                     time_s=0, thumb=[50] * 1920)])
        rows = [candidate(0, track='a/speaker_1', blur=140, thumb=[55] * 1920),
                candidate(20, track='a/speaker_1', blur=130, thumb=[56] * 1920),
                candidate(0, track='b/speaker_1', blur=120, thumb=[56] * 1920),
                candidate(20, track='b/speaker_1', blur=110, thumb=[62] * 1920)]
        selected = scan.select_scanned(pd.DataFrame(rows), upstream, proxy_weights=WEIGHTS)
        self.assertEqual(list(zip(selected.track, selected.time_s)), [('a/speaker_1', 20), ('b/speaker_1', 20)])
        self.assertEqual(selected['index'].tolist(), [1, 2])
        with self.assertRaisesRegex(ValueError, 'thumbnail'):
            scan.select_scanned(pd.DataFrame(rows), upstream.drop(columns='thumb'), proxy_weights=WEIGHTS)

    def test_scan_hair_group_compares_upstream_other_properties_and_tracks(self):
        upstream = pd.DataFrame([dict(uuid='u', recording_id='old', speaker=1, property=7,
                                     time_s=0, thumb=[50] * 1920)])
        rows = [candidate(0, track='a/speaker_1', yaw=60, n_hands=1,
                          h0_target='clothing', h0_sharp=100, thumb=[55] * 1920),
                candidate(20, track='a/speaker_1', yaw=60, n_hands=1,
                          h0_target='clothing', h0_sharp=90, thumb=[56] * 1920)]
        selected = scan.select_scanned(pd.DataFrame(rows), upstream, proxy_weights=WEIGHTS)
        self.assertEqual(selected['property'].tolist(), [8])
        self.assertEqual(selected.time_s.tolist(), [20])

    def test_duplicate_candidates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            scan.select_scanned(pd.DataFrame([candidate(), candidate()]), pd.DataFrame(), proxy_weights=WEIGHTS)


class DecodeTests(unittest.TestCase):
    def test_one_second_scan_skips_invalid_and_excluded_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "test.mkv"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                            "testsrc2=size=512x960:rate=25:duration=1", "-threads", "1", "-c:v", "ffv1", str(video)], check=True)
            scan.validate_video(video)
            points = np.tile(np.asarray(candidate()["kp"], np.float32).reshape(1, 133, 3), (25, 1, 1))
            points[:, :, :2] /= 2
            valid = np.ones(25, bool)
            valid[12] = False
            point_bytes, valid_bytes = points.tobytes(), valid.tobytes()
            header = json.dumps({"keypoints": {"dtype": "F32", "shape": list(points.shape), "data_offsets": [0, len(point_bytes)]},
                                 "valid": {"dtype": "BOOL", "shape": [25], "data_offsets": [len(point_bytes), len(point_bytes) + len(valid_bytes)]}}).encode()
            pose = root / "poses.safetensors"
            pose.write_bytes(len(header).to_bytes(8, "little") + header + point_bytes + valid_bytes)
            track = dict(uuid="u", track="r/speaker_1", recording_id="r", speaker=1, session_id="s",
                         video_path=video, source="hf://buckets/test/videos/full.mp4", usable_chunks=[0], excluded_ranges=[(0, 0)])
            with patch.object(scan, "measure_video_frame", return_value={"mp_ok": False}):
                candidates = scan.scan_chunk(track, 0, pose, modules=None)
            self.assertEqual(candidates.frame_index.tolist(), [24])
            self.assertEqual(candidates.time_s.tolist(), [.96])
            self.assertTrue(candidates.source.iloc[0].endswith("#frame=24"))
            selected = candidates.assign(property=1, index=0, cx0=0, cy0=0, cw=256, ch=480)
            decoded = scan.decode_selected_crops(selected, {track["track"]: track}, root / "frames")
            self.assertFalse(decoded.prepared_image.iloc[0])
            expected = list(scan._decode_frames(video, 0, "eq(n\\,24)"))[0]
            with Image.open(decoded.image_path.iloc[0]) as image:
                self.assertEqual(image.size, (512, 960))
                np.testing.assert_array_equal(np.asarray(image), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
