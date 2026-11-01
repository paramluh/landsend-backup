import copy
import importlib.util
import math
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("pose_metrics", Path(__file__).resolve().parents[1] / "pose_metrics.py")
pose_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pose_metrics)


def document(*rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    return {"run": "fixture", "train_cols": ["step", "rows"] + keys,
            "train": [[step, step * 8] + [row.get(key) for key in keys] for step, row in enumerate(rows, 1)]}


def value(doc, key, row=0):
    return doc["train"][row][doc["train_cols"].index(key)]


class PoseMetricsTest(unittest.TestCase):
    def method(self):
        return {"modality_weights": {"text": .2, "audio": .3, "pose": .5},
                "codebook_weights": [.5] + [.5 / 31] * 31, "reduction": "row_mean",
                "pose_teacher_cfg_scale": 3}

    def test_unweights_codebooks_and_depth_means_with_actual_coefficients(self):
        method = self.method()
        row = {"loss/text": .8, "loss/direct": 900, "loss/audio": 700, "loss/pose": 800}
        for stream in ("audio", "pose"):
            for cb in range(32):
                raw = 100 if cb == 0 else cb
                row[f"loss/{stream}_cb{cb}"] = raw * method["modality_weights"][stream] * method["codebook_weights"][cb]
        got = pose_metrics.enrich_doc(document(row), method)
        self.assertEqual(value(got, "kl/text"), 4)
        for stream in ("audio", "pose"):
            self.assertAlmostEqual(value(got, f"kl/{stream}/tower"), 100)
            self.assertAlmostEqual(value(got, f"kl/{stream}/depth_mean"), 16)
            for cb in range(32):
                self.assertAlmostEqual(value(got, f"kl/{stream}/cb{cb}"), 100 if cb == 0 else cb)
        self.assertEqual(got["kl_metadata"]["depth_codebooks"], list(range(1, 32)))
        self.assertEqual(got["kl_metadata"]["pose_teacher_cfg_scale"], 3)

    def test_no_default_weights_and_no_aggregate_loss_proxy(self):
        source = document({"loss/text": 1, "loss/audio": 2, "loss/pose": 3, "loss/audio_cb0": 4})
        got = pose_metrics.enrich_doc(source, {})
        for key in ("kl/text", "kl/audio/cb0", "kl/audio/tower", "kl/audio/depth_mean"):
            self.assertIsNone(value(got, key))
        got = pose_metrics.enrich_doc(document({"loss/audio": 2, "loss/pose": 3}), self.method())
        self.assertIsNone(value(got, "kl/audio/tower"))
        self.assertIsNone(value(got, "kl/audio/depth_mean"))

    def test_zero_or_invalid_coefficient_never_inverted(self):
        for invalid in (0, -1, None, float("nan"), float("inf"), True):
            method = self.method()
            method["modality_weights"]["text"] = invalid
            method["codebook_weights"][0] = invalid
            got = pose_metrics.enrich_doc(document({"loss/text": 1, "loss/audio_cb0": 1}), method)
            self.assertIsNone(value(got, "kl/text"))
            self.assertIsNone(value(got, "kl/audio/cb0"))

    def test_depth_requires_all_31_valid_raw_books(self):
        row = {f"kl/audio/cb{cb}": cb for cb in range(1, 31)}
        got = pose_metrics.enrich_doc(document(row), {})
        self.assertIsNone(value(got, "kl/audio/depth_mean"))
        row["kl/audio/cb31"] = 31
        got = pose_metrics.enrich_doc(document(row), {})
        self.assertEqual(value(got, "kl/audio/depth_mean"), 16)
        self.assertIsNone(value(got, "kl/audio/tower"))
        row["kl/audio/cb5"] = float("inf")
        self.assertIsNone(value(pose_metrics.enrich_doc(document(row), {}), "kl/audio/depth_mean"))

    def test_explicit_metrics_and_gradient_terms_pass_through(self):
        source = document({"kl/text": 11, "loss/text": .8, "kl/lexical": .7, "kl/category": .08,
                           "kl/audio/tower": 22, "kl/audio/depth_mean": 33,
                           "grad_norm/direct/layers": 2, "grad_norm/credit/layers": 3,
                           "grad_norm/direct/pose_head": .4})
        got = pose_metrics.enrich_doc(source, self.method())
        for key, expected in (("kl/text", 11), ("kl/lexical", .7), ("kl/category", .08),
                              ("kl/audio/tower", 22), ("kl/audio/depth_mean", 33),
                              ("grad_norm/direct/layers", 2), ("grad_norm/credit/layers", 3)):
            self.assertEqual(value(got, key), expected)
        self.assertEqual(got["metric_definitions"]["kl/text"]["source"], "logged")
        group = next(group for group in got["grad_groups"] if group["component"] == "layers")
        self.assertEqual(group["available"], {"direct": True, "credit": True})
        self.assertIn("pose_head", [group["component"] for group in got["grad_groups"]])

    def test_gradient_total_and_credit_coefficient_are_not_proxies(self):
        got = pose_metrics.enrich_doc(document({"gradient_norm": 4, "credit/absolute_sum": 9}), {})
        self.assertTrue(all(not any(group["available"].values()) for group in got["grad_groups"]))
        self.assertNotIn("grad_norm/direct/layers", got["train_cols"])
        self.assertIsNone(value(got, "kl/lexical"))
        self.assertIsNone(value(got, "kl/category"))
        self.assertEqual(len(got["critical_metrics"]), 7)

    def test_per_row_logged_priority_and_provenance(self):
        source = document({"kl/text": 9, "loss/text": .8}, {"kl/text": None, "loss/text": 1.2}, {"kl/text": None})
        got = pose_metrics.enrich_doc(source, self.method())
        self.assertEqual(value(got, "kl/text", 0), 9)
        self.assertAlmostEqual(value(got, "kl/text", 1), 6)
        self.assertIsNone(value(got, "kl/text", 2))
        definition = got["metric_definitions"]["kl/text"]
        self.assertEqual((definition["source"], definition["logged_rows"], definition["derived_rows"], definition["missing_rows"]), ("mixed", 1, 1, 1))

    def test_no_input_mutation_and_idempotent_values(self):
        source, method = document({"loss/text": 1, "grad_norm/direct/layers": 2}), self.method()
        saved_source, saved_method = copy.deepcopy(source), copy.deepcopy(method)
        got = pose_metrics.enrich_doc(source, method)
        self.assertEqual(source, saved_source)
        self.assertEqual(method, saved_method)
        again = pose_metrics.enrich_doc(got, method)
        self.assertEqual(again["train_cols"], got["train_cols"])
        self.assertEqual(again["train"], got["train"])

    def test_empty_pending_document_has_honest_metadata(self):
        got = pose_metrics.enrich_doc({"train_cols": ["step", "rows"], "train": []}, {})
        self.assertEqual(got["train"], [])
        self.assertEqual(got["kl_metadata"]["reduction"], "unspecified")
        self.assertEqual(len(got["missing_telemetry"]), 27)
        self.assertEqual(len(got["grad_groups"]), 10)

    def test_rejects_ambiguous_matrix(self):
        for source in ({"train_cols": ["step", "step"], "train": [[1, 1]]},
                       {"train_cols": ["step", "rows"], "train": [[1]]}):
            with self.assertRaises(ValueError):
                pose_metrics.enrich_doc(source, {})


if __name__ == "__main__":
    unittest.main()
