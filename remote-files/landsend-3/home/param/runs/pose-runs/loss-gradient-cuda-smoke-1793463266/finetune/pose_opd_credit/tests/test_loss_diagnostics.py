"""Independent autograd checks of cheap, globally reduced logit diagnostics."""
import contextlib
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import unittest
from unittest import mock

import torch

from finetune.pose_opd_credit import runtime
from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.loss_diagnostics import (
    MODALITIES, STREAM_SLICES, STAT_NAMES, categorical_statistics,
    scored_categorical_statistics, logit_gradient_sums, finalize_loss_diagnostics,
)
from finetune.pose_opd_credit.text_head import chunked_text_objective
from finetune.pose_opd_credit.tests.test_runtime import ToyPPM, fixture


class LogitDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        torch.set_num_threads(1)

    def test_origin_and_combined_norms_match_independent_autograd(self):
        logits = torch.randn(3, 65, 4, dtype=torch.float64, requires_grad=True)
        teacher = torch.randn_like(logits)
        actions = torch.randint(4, (3, 65))
        _, _, statistics = scored_categorical_statistics(logits, teacher, actions)
        p, q = logits.softmax(-1), teacher.softmax(-1)
        kl = (p * (p.log() - q.log())).sum(-1)
        lp = p.log().gather(-1, actions[..., None]).squeeze(-1)
        weights = torch.rand(3, 65, dtype=torch.float64) * .01
        weights[0, :] = 0  # forced/padded positions have no direct contribution
        origins = {name: torch.rand_like(weights) * .02 for name in MODALITIES}
        for value in origins.values():
            value[0, :] = 0
        credit = sum(origins.values())
        actual = finalize_loss_diagnostics(logit_gradient_sums(statistics, weights, credit, origins), 1)
        source_combined_squared = 0.
        for name, columns in zip(MODALITIES, STREAM_SLICES):
            direct = torch.autograd.grad((weights[:, columns] * kl[:, columns]).sum(), logits,
                                         retain_graph=True)[0]
            score = torch.autograd.grad((origins[name] * lp).sum(), logits, retain_graph=True)[0]
            self.assertAlmostEqual(actual[f'logit_grad_norm/direct/{name}'], float(direct.norm()), places=12)
            self.assertAlmostEqual(actual[f'logit_grad_norm/credit_from/{name}'], float(score.norm()), places=12)
            self.assertAlmostEqual(actual[f'logit_grad_norm/combined/{name}'], float((direct + score).norm()), places=12)
            self.assertAlmostEqual(actual[f'logit_grad_dot/{name}'], float((direct * score).sum()), places=12)
            source_combined_squared += float((direct + score).square().sum())
        all_gradient = torch.autograd.grad((weights * kl + credit * lp).sum(), logits)[0]
        self.assertAlmostEqual(actual['logit_grad_norm/combined/all'], float(all_gradient.norm()), places=12)
        self.assertGreater(abs(actual['logit_grad_norm/combined/all'] ** 2 - source_combined_squared), .001)

    def test_rank_and_microbatch_reduction_sum_squares_before_sqrt(self):
        logits = torch.randn(4, 65, 5, dtype=torch.float64)
        _, _, stats = scored_categorical_statistics(logits, torch.randn_like(logits), torch.randint(5, (4, 65)))
        weights = torch.rand(4, 65, dtype=torch.float64) * .1
        origins = {name: torch.rand_like(weights) for name in MODALITIES}
        credits = sum(origins.values())
        expected = finalize_loss_diagnostics(logit_gradient_sums(stats, weights, credits, origins), 1)
        local = []
        for rank in range(2):
            summed = {}
            for row in range(rank * 2, rank * 2 + 2):
                part = slice(row, row + 1)
                values = logit_gradient_sums({k: v[part] for k, v in stats.items()},
                    weights[part] * 2, credits[part] * 2,
                    {k: v[part] * 2 for k, v in origins.items()}, world=2)
                for key, value in values.items():
                    summed[key] = summed.get(key, 0.) + value
            local.append(summed)
        averaged = {key: (local[0][key] + local[1][key]) / 2 for key in local[0]}
        actual = finalize_loss_diagnostics(averaged, 2)
        self.assertEqual(actual.keys(), expected.keys())
        for key in actual:
            self.assertAlmostEqual(actual[key], expected[key], places=12, msg=key)
        zero = {key: 0. for key in averaged}
        dummy = finalize_loss_diagnostics(zero, 2)
        self.assertFalse(any(key.startswith(('logit_grad_alignment/', 'logit_grad_credit_direct_ratio/')) for key in dummy))
        self.assertTrue(all(value == 0. for value in dummy.values()))

    def test_confident_float32_score_norm_does_not_cancel_to_zero(self):
        logits = torch.tensor([[0., -18., -19., -20.]], requires_grad=True)
        teacher = torch.tensor([[.2, -.3, .7, .1]])
        actions = torch.tensor([0])
        _, _, statistics = scored_categorical_statistics(logits, teacher, actions)
        logp = logits.log_softmax(-1)
        probability = logp.exp()
        lp = logp.gather(-1, actions[:, None]).sum()
        score = torch.autograd.grad(lp, logits, retain_graph=True)[0]
        self.assertGreater(float(statistics['score_sq']), 0.)
        torch.testing.assert_close(statistics['score_sq'], score.square().sum(-1), rtol=1e-6, atol=0)
        kl = (probability * (logp - teacher.log_softmax(-1))).sum(-1)
        direct = torch.autograd.grad(kl.sum(), logits)[0]
        torch.testing.assert_close(statistics['direct_score_dot'], (direct * score).sum(-1), rtol=1e-5, atol=1e-21)

    def test_full_text_is_category_plus_weighted_conditional_lexical_kl(self):
        student = torch.randn(11, 9, dtype=torch.float64)
        teacher = torch.randn_like(student)
        full, _, values = scored_categorical_statistics(student, teacher, torch.zeros(11, dtype=torch.long), word_count=5)
        p, q = student.softmax(-1), teacher.softmax(-1)
        word_p, word_q = p[:, :5].sum(-1), q[:, :5].sum(-1)
        conditional_p = p[:, :5] / word_p[:, None]
        conditional_q = q[:, :5] / word_q[:, None]
        expected_lexical = (conditional_p * (conditional_p.log() - conditional_q.log())).sum(-1)
        category_p = torch.cat((word_p[:, None], p[:, 5:]), -1)
        category_q = torch.cat((word_q[:, None], q[:, 5:]), -1)
        expected_category = (category_p * (category_p.log() - category_q.log())).sum(-1)
        torch.testing.assert_close(values['lexical'], expected_lexical, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(values['category'], expected_category, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(full, values['category'] + values['lexical_contribution'], rtol=1e-13, atol=1e-13)
        # The conditional lexical diagnostic remains meaningful when p(WORD)
        # is too small to represent; its full-KL contribution is then zero.
        _, _, tiny = scored_categorical_statistics(torch.tensor([[-1000., -1001., 0.]]),
            torch.tensor([[-1001., -1000., 0.]]), torch.tensor([2]), word_count=2)
        self.assertGreater(float(tiny['lexical']), .1)
        self.assertEqual(float(tiny['lexical_contribution']), 0.)

    def test_text_chunks_collect_stats_without_gradient_gemms_or_changed_gradients(self):
        heads = [torch.randn(shape, dtype=torch.float64, requires_grad=True)
                 for shape in ((7, 3), (8, 3), (1, 3), (7, 4), (8, 4), (1, 4))]
        actions = torch.arange(7)
        weights, credit = torch.rand(7, dtype=torch.float64), torch.rand(7, dtype=torch.float64)
        recorded = []
        with torch.no_grad(), mock.patch('finetune.pose_opd_credit.text_head._mm_accum',
                                          side_effect=AssertionError('Unexpected gradient GEMM')):
            _, kl, lp = chunked_text_objective(*heads, actions, weights, credit,
                special_ids=(6,), executable_vocab=7, chunk_size=3,
                statistics_callback=lambda start, values: recorded.append((start, values)))
        self.assertEqual([start for start, _ in recorded], [0, 3, 6])
        self.assertEqual(sum(len(values['direct_sq']) for _, values in recorded), 7)
        for _, values in recorded:
            self.assertTrue(all(not value.requires_grad and value.ndim == 1 for value in values.values()))
        for callback in (None, lambda *_: None):
            result = chunked_text_objective(*heads, actions, weights, credit,
                special_ids=(6,), executable_vocab=7, chunk_size=2, statistics_callback=callback)
            torch.testing.assert_close(result[1], kl)
            torch.testing.assert_close(result[2], lp)
            gradients = torch.autograd.grad(result[0], heads[:3])
            if callback is None:
                expected = gradients
            else:
                for actual, wanted in zip(gradients, expected):
                    torch.testing.assert_close(actual, wanted, rtol=1e-13, atol=1e-13)

    def test_replay_diagnostics_add_no_model_calls_and_mask_dummy_rows(self):
        config = TrainingConfig(decoder_chunk_frames=3, text_chunk_frames=2)
        for dummy in (False, True):
            student, audio, pose = ToyPPM(), ToyPPM().requires_grad_(False), ToyPPM().requires_grad_(False)
            bundle, latents = fixture(dummy=dummy)
            with mock.patch.object(runtime, 'resident_decoders', side_effect=lambda *_: contextlib.nullcontext()):
                metrics, _, _ = runtime.replay_update(student, audio, pose, bundle, latents, config,
                                                     None, torch.device('cpu'))
            self.assertEqual(student.calls, [('student', False), ('student', True)])
            self.assertEqual(audio.calls, [('audio_teacher', False)])
            self.assertEqual(pose.calls, [('pose_teacher', False)])
            self.assertEqual([grad for _, grad in student.dec_s.calls], [False, False, True, True])
            self.assertEqual([grad for _, grad in student.dec_p.calls], [False, False, True, True])
            self.assertAlmostEqual(metrics['kl/text'], metrics['kl/category'] + metrics['kl/lexical_contribution'], places=6)
            final = finalize_loss_diagnostics(metrics, 1)
            if dummy:
                self.assertEqual(final['kl/text'], 0.)
                self.assertEqual(final['logit_grad_norm/combined/all'], 0.)
            else:
                self.assertGreater(final['logit_grad_norm/combined/all'], 0.)

    def test_packed_padded_rows_match_accumulated_diagnostic_statistics(self):
        class PaddedToy(ToyPPM):
            @staticmethod
            def media_embedding(module, codes):
                # Unlike the minimal unpadded fixture, accept absent -1 codes.
                local = codes.clamp_min(0).remainder(2048)
                return (module(local) * (codes >= 0)[..., None]).mean(1)

        examples = [fixture(frames=6), fixture(frames=8), fixture(frames=7, dummy=True)]
        bundles, latents = map(list, zip(*examples))
        for bundle in bundles:
            bundle['student']['pad_id'] = 0
        original = [PaddedToy(), PaddedToy().requires_grad_(False), PaddedToy().requires_grad_(False)]
        normalization = runtime.update_normalization(bundles, torch.device('cpu'))
        config = TrainingConfig(decoder_chunk_frames=3, text_chunk_frames=2)
        results = []
        with mock.patch.object(runtime, 'resident_decoders', side_effect=lambda *_: contextlib.nullcontext()):
            for size, padding in ((1, None), (3, 40)):
                models = copy.deepcopy(original)
                accumulated = {}
                for start in range(0, len(bundles), size):
                    metrics, _, _ = runtime.replay_update(*models, bundles[start:start + size],
                        latents[start:start + size], replace(config, fixed_sequence_frames=padding),
                        None, torch.device('cpu'), normalization=normalization)
                    for key, value in metrics.items():
                        accumulated[key] = accumulated.get(key, 0.) + value
                results.append(finalize_loss_diagnostics(accumulated, 1))
        for key in results[0]:
            if key.startswith(('logit_grad_', 'kl/')):
                self.assertAlmostEqual(results[0][key], results[1][key], places=6, msg=key)

    def test_concurrent_text_collectors_are_call_local(self):
        heads = [torch.randn(shape, dtype=torch.float64)
                 for shape in ((5, 3), (8, 3), (1, 3), (5, 4), (8, 4), (1, 4))]
        def collect(offset):
            recorded = []
            local = [heads[0] + offset, *heads[1:]]
            with torch.no_grad():
                chunked_text_objective(*local, torch.arange(5), torch.ones(5), torch.zeros(5),
                    special_ids=(6,), executable_vocab=7, chunk_size=2,
                    statistics_callback=lambda start, values: recorded.append((start, values['direct_sq'].clone())))
            self.assertEqual([start for start, _ in recorded], [0, 2, 4])
            return torch.cat([value for _, value in recorded])
        expected = [collect(0.), collect(.7)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            actual = list(pool.map(collect, (0., .7)))
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
