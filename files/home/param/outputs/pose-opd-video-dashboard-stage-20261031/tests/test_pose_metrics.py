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
    def test_conversation_metrics_keep_only_requested_categories_without_renormalizing(self):
        logged = {'conversation/student/probability/pad': .4,
                  'conversation/student/probability/new_word': .1,
                  'conversation/student/probability/word': .3,
                  'conversation/student/sampled_count/pad': 12,
                  'conversation/student/decisions': 20,
                  'conversation/partner/probability/word': .75}
        got = pose_metrics.enrich_doc(document(logged, {}), {})
        for key, expected in logged.items():
            self.assertEqual(value(got, key), expected)
            self.assertIsNone(value(got, key, 1))
        self.assertEqual(len(got['conversation_groups']), 2)
        self.assertTrue(all(len(group['series']) == 6 for group in got['conversation_groups']))
        self.assertFalse(any(key.endswith('/other') for key in got['metric_definitions'] if key.startswith('conversation/')))
        self.assertAlmostEqual(sum(value(got, 'conversation/student/probability/' + name)
                                   for name in ('pad', 'new_word', 'word')), .8)

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
        self.assertEqual(group["available"], {"combined": False, "direct": True, "credit": True})
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
        self.assertEqual(len(got["missing_telemetry"]), 37)
        self.assertEqual(len(got["grad_groups"]), 10)

    def test_pose_override_unweights_pose_independently_of_audio(self):
        method = self.method()
        method['pose_codebook_weights'] = [1 / 3] + [2 / 93] * 31
        row = {}
        for stream in ('audio', 'pose'):
            weights = method['pose_codebook_weights'] if stream == 'pose' else method['codebook_weights']
            for cb in range(32):
                row[f'loss/{stream}_cb{cb}'] = (cb + 1) * method['modality_weights'][stream] * weights[cb]
        got = pose_metrics.enrich_doc(document(row), method)
        for stream in ('audio', 'pose'):
            self.assertAlmostEqual(value(got, f'kl/{stream}/tower'), 1)
            self.assertAlmostEqual(value(got, f'kl/{stream}/depth_mean'), 17)
        method['pose_codebook_weights'] = [0] * 32
        missing = pose_metrics.enrich_doc(document(row), method)
        self.assertIsNone(value(missing, 'kl/pose/tower'))
        self.assertAlmostEqual(value(missing, 'kl/audio/tower'), 1)

    def test_combined_parameter_and_logit_gradient_spaces_are_separate(self):
        row = {'grad_norm/combined/layers': 7, 'grad_norm/combined/other': 2,
               'grad_norm/combined/pose_head': 1, 'logit_grad_norm/direct/pose': .1,
               'logit_grad_norm/combined/all': .15, 'logit_grad_norm/combined/pose': .12,
               'logit_grad_norm/direct/all': .2, 'logit_grad_norm/credit_from/all': .3,
               'logit_grad_dot/all': -.01,
               'logit_grad_norm/credit_from/pose': .2, 'logit_grad_norm/credit_on/audio': .3,
               'logit_grad_alignment/pose': -.5, 'logit_grad_credit_direct_ratio/pose': 2,
               'kl/lexical_contribution': .05}
        got = pose_metrics.enrich_doc(document(row), self.method())
        for key, expected in row.items():
            self.assertEqual(value(got, key), expected)
        groups = {group['component']: group for group in got['grad_groups']}
        self.assertEqual(groups['layers']['available'], {'combined': True, 'direct': False, 'credit': False})
        self.assertIn('pose_head', groups)
        self.assertIn('other', groups)
        self.assertNotIn('grad_norm/direct/dec_p', got['train_cols'])
        self.assertIn('not a parameter-gradient norm',
                      got['metric_definitions']['logit_grad_norm/credit_from/pose']['definition'])
        self.assertEqual(len(got['logit_grad_groups']), 4)
        self.assertEqual(len(got['logit_grad_diagnostics']), 8)
        self.assertEqual(len(got['logit_grad_groups'][0]['series']), 3)

    def test_periodic_parameter_norms_keep_unmeasured_updates_missing(self):
        method = {**self.method(), 'loss_gradient_interval_steps': 10}
        source = document({'gradient_norm': 1},
                          {'gradient_norm': 2, 'grad_norm/direct/text/all': .7,
                           'grad_norm/credit/text/all': .4, 'grad_norm/credit/pose/dec_p': 0},
                          {'gradient_norm': 3},
                          {'gradient_norm': 4, 'grad_norm/direct/text/all': .8})
        for row, update in zip(source['train'], (9, 10, 11, 20)):
            row[0] = update
        got = pose_metrics.enrich_doc(source, method)
        self.assertEqual([value(got, 'grad_norm/direct/text/all', row) for row in range(4)],
                         [None, .7, None, .8])
        self.assertEqual(value(got, 'grad_norm/credit/pose/dec_p', 1), 0)
        self.assertIsNone(value(got, 'grad_norm/credit/pose/dec_p', 2))
        self.assertEqual(got['parameter_grad_cadence']['interval_steps'], 10)
        self.assertEqual(got['parameter_grad_cadence']['extra_backward_passes'], 5)
        definition = got['metric_definitions']['grad_norm/credit/pose/dec_p']
        self.assertEqual(definition['measurement_interval_steps'], 10)
        self.assertIn('floating-point residual roundoff', definition['definition'])
        self.assertEqual(len(got['parameter_loss_grad_groups']), 13)

    def test_rejects_ambiguous_matrix(self):
        for source in ({"train_cols": ["step", "step"], "train": [[1, 1]]},
                       {"train_cols": ["step", "rows"], "train": [[1]]}):
            with self.assertRaises(ValueError):
                pose_metrics.enrich_doc(source, {})


if __name__ == "__main__":
    unittest.main()
