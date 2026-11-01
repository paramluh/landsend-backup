"""CPU proofs of objective direction, production weighting and gradient ownership."""
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from finetune.duplex_distill.losses import (
    distribution_kl,
    factorized_text_kl_from_hidden,
    paired_metric_namespaces,
)


def project(x, w, r, spec):
    z = x @ w.t()
    z[:, list(spec)] = x @ r.t()
    return z


def factor_reference(student, teacher, word_end, lexical_weight, weight):
    # Independently build category distributions: aggregate WORD plus every
    # special ID. Avoid using fullKL - p(WORD)*lexicalKL from the implementation.
    slog = student.log_softmax(-1)
    tlog = teacher.detach().log_softmax(-1)
    scat = torch.cat((slog[:, :word_end].logsumexp(-1, keepdim=True), slog[:, word_end:]), -1)
    tcat = torch.cat((tlog[:, :word_end].logsumexp(-1, keepdim=True), tlog[:, word_end:]), -1)
    cat = (tcat.exp() * (tcat - scat)).sum(-1)
    sl = student[:, :word_end].log_softmax(-1)
    tl = teacher.detach()[:, :word_end].log_softmax(-1)
    lexical = (tl.exp() * (tl - sl)).sum(-1)
    return ((cat + lexical_weight * tcat[:, 0].exp() * lexical) * weight).sum()


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(394)
        torch.set_num_threads(1)

    def test_forward_kl_direction_and_no_teacher_gradient(self):
        student = torch.tensor([[0.0, 2.0, -1.0]], dtype=torch.float64, requires_grad=True)
        teacher = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64, requires_grad=True)
        loss, stats = distribution_kl(student, teacher, torch.tensor([2]))
        expected = F.kl_div(student.log_softmax(-1), teacher.detach().softmax(-1), reduction='batchmean')
        reverse = F.kl_div(teacher.detach().log_softmax(-1), student.detach().softmax(-1), reduction='batchmean')
        torch.testing.assert_close(loss, expected)
        self.assertGreater(abs(loss.item() - reverse.item()), 0.05)
        loss.backward()
        torch.testing.assert_close(student.grad, student.detach().softmax(-1) - teacher.detach().softmax(-1))
        self.assertIsNone(teacher.grad)
        self.assertTrue(all(not x.requires_grad for x in stats.values()))

    def test_equal_distributions_zero_loss_gradient_despite_wrong_data_label(self):
        student = torch.tensor([[8.0, -2.0, -4.0]], requires_grad=True)
        teacher = student.detach().clone().requires_grad_(True)
        loss, stats = distribution_kl(student, teacher, torch.tensor([2]))
        self.assertEqual(loss.item(), 0.0)
        self.assertGreater(stats['data_ce_sum'].item(), 11.0)
        loss.backward()
        torch.testing.assert_close(student.grad, torch.zeros_like(student))
        self.assertIsNone(teacher.grad)

    def make_heads(self):
        # Unequal student/teacher hidden widths also exercise independent heads.
        return [torch.randn(*shape, dtype=torch.float64, requires_grad=True)
                for shape in ((7, 4), (11, 4), (3, 4), (7, 5), (11, 5), (3, 5))]

    def test_chunked_text_loss_and_all_gradients_match_reference(self):
        for lexical_weight in (1.0, 5.0):
            for chunk_size in (1, 3, 20):
                with self.subTest(lexical_weight=lexical_weight, chunk_size=chunk_size):
                    heads = self.make_heads()
                    target = torch.tensor([0, 1, 9, 10, 6, -100, 8])
                    weights = torch.tensor([0.1, 0.2, 0.1, 0.05, 0.2, 0.4, 0.15], dtype=torch.float64)
                    spec, word_end = (8, 9, 10), 7
                    loss, stats = factorized_text_kl_from_hidden(*heads, target, weights, spec,
                        word_end=word_end, lexical_weight=lexical_weight, chunk_size=chunk_size)
                    clones = [h.detach().clone().requires_grad_() for h in heads]
                    sz = project(*clones[:3], spec)
                    tz = project(*clones[3:], spec)
                    effective_weight = weights * (target >= 0)
                    expected = factor_reference(sz, tz, word_end, lexical_weight, effective_weight)
                    torch.testing.assert_close(loss, expected, atol=1e-12, rtol=1e-12)
                    loss.backward(); expected.backward()
                    for actual, reference in zip(heads[:3], clones[:3]):
                        torch.testing.assert_close(actual.grad, reference.grad, atol=2e-12, rtol=2e-12)
                    for teacher in heads[3:]:
                        self.assertIsNone(teacher.grad)
                    torch.testing.assert_close(heads[1].grad[list(spec)], torch.zeros_like(heads[1].grad[list(spec)]))
                    self.assertEqual(stats['count'].item(), 6)
                    self.assertTrue(all(not v.requires_grad for v in stats.values()))
                    self.assertEqual(stats['kl'][5].item(), 0.0)
                    if lexical_weight == 1:
                        torch.testing.assert_close(stats['kl'], stats['full_kl'])

    def test_text_kl_ignores_dataset_category_and_lexical_target(self):
        heads = self.make_heads()
        weights = torch.full((7,), 1 / 7, dtype=torch.float64)
        # Swapping all lexical labels to PAD-like special labels must leave the
        # distillation loss and gradient unchanged, but change the CE diagnostics.
        a, sa = factorized_text_kl_from_hidden(*heads, torch.zeros(7, dtype=torch.long), weights,
                                              (8, 9, 10), word_end=7)
        b, sb = factorized_text_kl_from_hidden(*heads, torch.full((7,), 8), weights,
                                              (8, 9, 10), word_end=7)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        ga = torch.autograd.grad(a, heads[:3], retain_graph=True)
        gb = torch.autograd.grad(b, heads[:3])
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        self.assertNotEqual(sa['data_weighted_ce_sum'].item(), sb['data_weighted_ce_sum'].item())

    def test_text_equal_distributions_zero_gradient_with_lexical_weight_five(self):
        sx, sw, sr, _, _, _ = self.make_heads()
        tx, tw, tr = [t.detach().clone().requires_grad_() for t in (sx, sw, sr)]
        loss, stats = factorized_text_kl_from_hidden(sx, sw, sr, tx, tw, tr,
            torch.tensor([1, 2, 8, 9, 10, 5, 6]), None, (8, 9, 10), word_end=7,
            lexical_weight=5.0, chunk_size=2)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        for s in (sx, sw, sr):
            torch.testing.assert_close(s.grad, torch.zeros_like(s))
        self.assertIsNone(tx.grad)
        self.assertGreater(stats['data_ce_sum'].item(), 0)

    def test_data_metrics_match_production_factorized_ce(self):
        heads = self.make_heads()
        target = torch.tensor([0, 2, 7, 8, 9, 10, -100])
        weight = torch.full((7,), 0.1, dtype=torch.float64)
        _, stats = factorized_text_kl_from_hidden(*heads, target, weight, (8, 9, 10), word_end=7,
                                                lexical_weight=1.0, data_lexical_weight=5.0)
        z = project(*heads[:3], (8, 9, 10)).detach()
        valid = target >= 0
        safe = target.clamp_min(0)
        ce = -z.log_softmax(-1).gather(1, safe[:, None]).squeeze(1)
        word = (target >= 0) & (target < 7)
        lex = -z[:, :7].log_softmax(-1).gather(1, safe.clamp_max(6)[:, None]).squeeze(1)
        expected = (ce + 4 * lex * word) * valid
        torch.testing.assert_close(stats['data_weighted_ce'], expected)
        torch.testing.assert_close(stats['weighted_data_ce'], (expected * weight).sum())

    def test_no_grad_eval_does_not_materialize_head_gradients(self):
        heads = self.make_heads()
        with mock.patch('finetune.duplex_distill.losses._mm_accum', side_effect=AssertionError('gradient GEMM in eval')):
            with torch.no_grad():
                loss, stats = factorized_text_kl_from_hidden(*heads, torch.zeros(7, dtype=torch.long), None,
                                                            (8, 9, 10), word_end=7)
        self.assertFalse(loss.requires_grad)
        self.assertTrue(all(not v.requires_grad for v in stats.values()))

    def test_decoder_codebooks_masking_and_global_weight_denominator(self):
        student = torch.randn(3, 31, 8, dtype=torch.float64, requires_grad=True)
        teacher = torch.randn(3, 31, 8, dtype=torch.float64, requires_grad=True)
        targets = torch.randint(0, 8, (3, 31)); targets[1] = -100
        # Production denominator includes other ranks/micros, and MUST NOT be
        # renormalized over this micro's two valid frames.
        frame_weights = torch.full((3,), 1 / (31 * 10), dtype=torch.float64)
        loss, stats = distribution_kl(student, teacher, targets, frame_weights)
        element = (teacher.detach().softmax(-1) * (teacher.detach().log_softmax(-1) - student.log_softmax(-1))).sum(-1)
        expected = (element[[0, 2]].sum()) / (31 * 10)
        torch.testing.assert_close(loss, expected)
        self.assertEqual(stats['count'].item(), 62)
        torch.testing.assert_close(stats['kl_by_codebook'], element.detach()[[0, 2]].sum(0))
        torch.testing.assert_close(stats['count_by_codebook'], torch.full((31,), 2))
        loss.backward()
        torch.testing.assert_close(student.grad[1], torch.zeros_like(student.grad[1]))
        self.assertIsNone(teacher.grad)

    def test_all_masked_micro_keeps_zero_student_gradient(self):
        s = torch.randn(4, 8, requires_grad=True)
        loss, stats = distribution_kl(s, torch.randn_like(s), torch.full((4,), -100))
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(stats['count'].item(), 0)
        loss.backward()
        torch.testing.assert_close(s.grad, torch.zeros_like(s))

    def test_metric_namespaces_keep_kl_and_detached_data_ce_separate(self):
        kl = torch.tensor(0.25, requires_grad=True)
        ce = torch.tensor(3.5, requires_grad=True)
        ka, ca = torch.arange(32.0, requires_grad=True), torch.arange(32.0, requires_grad=True) + 10
        metrics = paired_metric_namespaces({'loss': kl, 'light': kl}, {'loss': ce, 'light': ce},
            ka, ca, ka + 2, ca + 2, text_kl={'full': kl}, text_ce={'full': ce})
        self.assertEqual(metrics['train/loss'].item(), 0.25)
        self.assertEqual(metrics['data_loss/loss'].item(), 3.5)
        self.assertEqual(metrics['train/light_kl'].item(), 0.25)
        self.assertEqual(metrics['data_loss/light_ce'].item(), 3.5)
        self.assertEqual(metrics['text_kl/full'].item(), 0.25)
        self.assertEqual(metrics['data_text_ce/full'].item(), 3.5)
        for i in range(32):
            self.assertEqual(metrics[f'audio_codebooks/cb{i}'].item(), i)
            self.assertEqual(metrics[f'data_audio_codebooks/cb{i}'].item(), i + 10)
            self.assertEqual(metrics[f'pose_codebooks/cb{i}'].item(), i + 2)
            self.assertEqual(metrics[f'data_pose_codebooks/cb{i}'].item(), i + 12)
        self.assertTrue(all(not v.requires_grad for v in metrics.values()))
        with self.assertRaises(ValueError):
            paired_metric_namespaces({'loss': kl}, {'other': ce})
        with self.assertRaises(ValueError):
            paired_metric_namespaces({'loss': kl}, {'loss': ce}, ka[:31], ca[:31])
        with self.assertRaises(ValueError):
            paired_metric_namespaces({'cb0_ce': kl}, {'cb0_ce': ce})
        with self.assertRaises(ValueError):
            paired_metric_namespaces({'loss': kl}, {'loss': ce}, text_kl={'full': kl})
        with self.assertRaises(ValueError):
            paired_metric_namespaces({'loss': kl}, {'loss': ce},
                                    text_kl={'full': kl}, text_ce={'word': ce})


if __name__ == '__main__':
    unittest.main()
