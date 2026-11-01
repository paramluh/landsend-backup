"""Contract tests use only temporary log files; no trainer dependencies."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "pose_doc.py"
spec = importlib.util.spec_from_file_location("pose_doc", SOURCE)
pose_doc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pose_doc)


class PoseDocTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def log(self, name, rows, suffix=""):
        (self.root / name).write_text("".join(json.dumps(row) + "\n" for row in rows) + suffix)

    def doc(self):
        result = pose_doc.build_doc(self.root)
        json.dumps(result, allow_nan=False)
        return result

    @staticmethod
    def values(doc, key):
        index = doc["train_cols"].index(key)
        return [row[index] for row in doc["train"]]

    def test_pending_has_no_fabricated_values(self):
        doc = pose_doc.build_doc(None)
        self.assertEqual(doc["state"], "pending")
        self.assertEqual(doc["train"], [])
        for key in ("step", "max_steps", "rows_per_step", "epoch_rows", "save_every_min", "t_last", "last_save", "phase"):
            self.assertIsNone(doc[key], key)

    def test_metadata_exact_plan_and_padding_coverage(self):
        self.write("config.json", {"world": 2, "method": {"effective_batch_per_rank": 2, "checkpoint_interval_seconds": 7200}})
        self.write("plan.json", {"world": 2, "rows_per_rank": 2, "effective_batch": 4, "rows_per_epoch": 5, "epochs": 1,
                                 "updates": [[[0, n, 0] for n in range(4)], [[0, 4, 0], None, None, None]]})
        self.log("metrics.jsonl", [{"update": 1, "loss/direct": 2, "seconds": 30}, {"update": 2, "loss/direct": 1, "seconds": 10}])
        doc = self.doc()
        self.assertEqual((doc["max_steps"], doc["rows_per_step"], doc["epoch_rows"], doc["save_every_min"]), (2, 4, 5, 120))
        self.assertEqual([row[1] for row in doc["train"]], [4, 8])
        self.assertEqual(doc["processed_rows"], 5)
        self.assertEqual(doc["mean50_s"], 20)
        self.assertIsNone(doc["t_last"], "File modification time is not a training timestamp")
        self.assertEqual(doc["state"], "observed", "Reaching the plan end is not terminal process evidence")

    def test_stale_preparation_ignored_when_identity_differs(self):
        self.write("preparation.json", {"identity": {"world": 1, "method": {}}, "plan": {"updates": [[1]] * 99, "world": 1, "rows_per_epoch": 99}})
        self.write("config.json", {"world": 8, "method": {"effective_batch_per_rank": 4}})
        doc = self.doc()
        self.assertIsNone(doc["max_steps"])
        self.assertEqual(doc["rows_per_step"], 32)
        self.assertTrue(any("identity differs" in message for message in doc["warnings"]))

    def test_prepare_only_metadata_is_supported(self):
        self.write("preparation.json", {"identity": {"world": 2, "method": {"checkpoint_interval_seconds": 600}},
                                        "plan": {"world": 2, "rows_per_epoch": 3, "updates": [[1, 2], [3, None]]}})
        doc = self.doc()
        self.assertEqual((doc["max_steps"], doc["rows_per_step"], doc["save_every_min"]), (2, 2, 10))
        self.assertEqual(doc["state"], "pending")

    def test_metrics_rewind_merge_partial_and_nonfinite(self):
        self.log("metrics.jsonl", [
            {"update": 1, "loss/direct": 10}, {"update": 2, "loss/direct": 20},
            {"update": 3, "loss/direct": 30}, {"update": 2, "loss/direct": 200},
            {"update": 2, "credit/absolute_sum": 4, "loss/pose": float("nan")},
            {"update": True, "loss/direct": 999},
        ], '{"update": 3')
        doc = self.doc()
        self.assertEqual([row[0] for row in doc["train"]], [1, 2])
        self.assertEqual(self.values(doc, "loss/direct"), [10, 200])
        self.assertEqual(self.values(doc, "credit/absolute_sum"), [None, 4])
        self.assertEqual(len(doc["warnings"]), 2)
        self.assertNotIn("loss/pose", doc["train_cols"])

    def test_rank_zero_update_is_timestamped_global_metric_source(self):
        self.log("metrics.jsonl", [{"update": 1, "loss/direct": 2, "seconds": 15}])
        self.log("events-rank0.jsonl", [{"event": "update", "rank": 0, "time": 1700000001, "update": 1, "loss/direct": 2, "seconds": 15}])
        doc = self.doc()
        self.assertEqual(doc["t_last"], 1700000001)
        self.assertEqual(doc["update_times"], [1700000001])
        self.assertEqual(self.values(doc, "train/step_time"), [15])
        self.assertEqual(doc["latest_event"]["event"], "update")

    def test_event_update_can_arrive_before_its_metrics_line(self):
        self.log("events-rank0.jsonl", [{"event": "update", "rank": 0, "time": 1700000001, "update": 1, "loss/direct": 2, "seconds": 15}])
        doc = self.doc()
        self.assertEqual(doc["step"], 1)
        self.assertEqual(self.values(doc, "loss/direct"), [2])

    def test_restart_before_first_redone_update_removes_stale_tail(self):
        original = [{"update": n, "loss/direct": n, "seconds": 10} for n in (1, 2, 3)]
        self.log("metrics.jsonl", original)
        self.log("events-rank0.jsonl", [{"event": "update", "rank": 0, "time": 1700000000 + row["update"], **row} for row in original] + [
            {"event": "resumed", "rank": 0, "time": 1700000010, "next_update": 1},
            {"event": "phase", "rank": 0, "time": 1700000011, "update": 2, "phase": "data_audio_begin"},
        ])
        doc = self.doc()
        self.assertEqual(doc["step"], 1)
        self.assertEqual([row[0] for row in doc["train"]], [1])
        self.assertEqual(doc["phase"]["update"], 2)

    def test_restart_event_wins_over_old_same_update_metric(self):
        self.log("metrics.jsonl", [{"update": 1, "loss/direct": 2}, {"update": 2, "loss/direct": 3}])
        self.log("events-rank0.jsonl", [
            {"event": "update", "rank": 0, "time": 1700000001, "update": 1, "loss/direct": 2},
            {"event": "update", "rank": 0, "time": 1700000002, "update": 2, "loss/direct": 3},
            {"event": "update", "rank": 0, "time": 1700000010, "update": 1, "loss/direct": 42},
        ])
        doc = self.doc()
        self.assertEqual(self.values(doc, "loss/direct"), [42])
        self.assertEqual(doc["t_last"], 1700000010)

    def test_real_progress_is_submission_only_and_clears_for_replay(self):
        events = [
            {"event": "phase", "rank": 0, "time": 1700000001, "update": 1, "phase": "actor_rollout_begin"},
            {"event": "rollout_progress", "rank": 0, "time": 1700000002, "update": 1, "microbatch": 0,
             "current_frame": 275, "total_frames": 1002, "seconds_elapsed": 8},
        ]
        self.log("events-rank0.jsonl", events)
        doc = self.doc()
        self.assertEqual(doc["rollout_progress"]["kind"], "submitted_frames")
        self.assertEqual(doc["rollout_progress"]["current_frame"], 275)
        self.log("events-rank0.jsonl", events + [{"event": "phase", "rank": 0, "time": 1700000003, "update": 1, "phase": "replay_microbatch_begin", "microbatch": 0}])
        self.assertIsNone(self.doc()["rollout_progress"])

    def test_profiles_explicit_rank_zero_and_checkpoint_evidence(self):
        self.log("events-rank0.jsonl", [
            {"event": "profile", "rank": 0, "update": 1, "time": 1700000000, "rollout_seconds": 20, "peak_reserved_gib": 50},
            {"event": "profile", "rank": 1, "update": 1, "time": 1700000001, "rollout_seconds": 999},
            {"event": "update", "rank": 0, "update": 1, "time": 1700000002, "loss/direct": 2},
            {"event": "checkpoint", "rank": 0, "time": 1700000003,
             "path": {"step": 1, "path": "/example/step1", "resume_path": "/example/step1/resume", "export_path": "/example/step1/export"}},
        ])
        doc = self.doc()
        self.assertEqual(doc["profile_rank"], 0)
        self.assertEqual(doc["profile"][0][doc["profile_cols"].index("rollout_seconds")], 20)
        self.assertEqual(doc["last_save"], [1700000003, 1])
        self.assertEqual(doc["checkpoint"]["evidence"], "events-rank0.jsonl")

    def test_terminal_result_is_ignored_after_resume(self):
        self.log("events-rank0.jsonl", [{"event": "update", "rank": 0, "update": 1, "time": 1700000002, "loss/direct": 2}])
        result = self.write("RESULT.json", {"status": "bounded_or_stopped", "next_update": 1, "checkpointing_enabled": False})
        os.utime(result, (1700000003, 1700000003))
        doc = self.doc()
        self.assertEqual(doc["state"], "bounded_or_stopped")
        self.assertFalse(doc["checkpointing_enabled"])
        self.log("events-rank0.jsonl", [{"event": "resumed", "rank": 0, "time": 1700000010, "next_update": 1}])
        doc = self.doc()
        self.assertEqual(doc["state"], "observed")
        self.assertIsNone(doc["checkpointing_enabled"])

    def test_configured_overrides_are_explicit_and_validated(self):
        doc = pose_doc.build_doc(None, {"run": "selected", "max_steps": 10, "rows_per_step": 8})
        self.assertEqual((doc["run"], doc["max_steps"], doc["rows_per_step"]), ("selected", 10, 8))
        for invalid in (-1, 0, True, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                pose_doc.build_doc(None, {"max_steps": invalid})


if __name__ == "__main__":
    unittest.main()
