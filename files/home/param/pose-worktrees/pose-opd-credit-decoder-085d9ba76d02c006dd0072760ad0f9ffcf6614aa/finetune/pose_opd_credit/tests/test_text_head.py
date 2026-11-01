"""Independent dense-autograd checks for the bounded reverse-KL text head."""
import unittest
from unittest import mock

import torch

from finetune.pose_opd_credit.text_head import chunked_text_objective, logits_objective


def project(x, weight, special, ids, vocab):
    z = x @ weight[:vocab].to(x.dtype).t()
    if ids:
        z[:, list(ids)] = x @ special.to(x.dtype).t()
    return z


def dense_reference(heads, actions, direct, credit, ids, vocab):
    sx, sw, sr, tx, tw, tr = heads
    dtype = torch.float64 if sx.dtype == torch.float64 else torch.float32
    sz = project(sx, sw, sr, ids, vocab).to(dtype)
    tz = project(tx, tw, tr, ids, vocab).detach().to(dtype)
    p, q = sz.softmax(-1), tz.softmax(-1)
    # Independent dense-autograd construction; no analytic-gradient helper.
    kl = (p * (p.log() - q.log())).sum(-1)
    logp = p.log().gather(1, actions[:, None]).squeeze(1)
    loss = (direct.detach() * kl + credit.detach() * (logp - logp.detach())).sum()
    return loss, kl, logp


class TextHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(811)
        torch.set_num_threads(1)

    def fixture(self, dtype=torch.float64, special_count=3):
        # Different hidden widths ensure teacher projection remains independent.
        shapes = ((7, 4), (13, 4), (special_count, 4),
                  (7, 5), (13, 5), (special_count, 5))
        heads = [(torch.randn(shape, dtype=dtype) * .3).requires_grad_() for shape in shapes]
        actions = torch.tensor([0, 2, 8, 10, 6, 7, 1])
        direct = torch.tensor([.1, .0, .2, .3, .0, .1, .2], dtype=dtype, requires_grad=True)
        credit = torch.tensor([.0, .7, .2, .0, .3, .4, .1], dtype=dtype, requires_grad=True)
        return heads, actions, direct, credit

    def test_dense_loss_and_all_gradients_with_special_rows(self):
        for chunk in (1, 3, 20):
            with self.subTest(chunk=chunk):
                heads, actions, direct, credit = self.fixture()
                copies = [t.detach().clone().requires_grad_() for t in heads]
                actual, kl, logp = chunked_text_objective(*heads, actions, direct, credit,
                    special_ids=(8, 9, 10), executable_vocab=11, chunk_size=chunk)
                expected, ref_kl, ref_logp = dense_reference(copies, actions, direct, credit, (8, 9, 10), 11)
                torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)
                torch.testing.assert_close(kl, ref_kl, rtol=2e-13, atol=2e-13)
                torch.testing.assert_close(logp, ref_logp, rtol=2e-13, atol=2e-13)
                (.37 * actual).backward()
                (.37 * expected).backward()
                for actual_head, reference_head in zip(heads[:3], copies[:3]):
                    torch.testing.assert_close(actual_head.grad, reference_head.grad, rtol=2e-12, atol=2e-12)
                self.assertTrue(all(t.grad is None for t in heads[3:]))
                self.assertIsNone(direct.grad)
                self.assertIsNone(credit.grad)
                self.assertFalse(kl.requires_grad)
                self.assertFalse(logp.requires_grad)
                # Replaced rows and padded vocabulary never receive gradient.
                torch.testing.assert_close(heads[1].grad[[8, 9, 10, 11, 12]], torch.zeros(5, 4, dtype=torch.float64))

    def test_chunk_invariance(self):
        heads, actions, direct, credit = self.fixture()
        values = []
        for chunk in (1, 2, 7, 100):
            copies = [t.detach().clone().requires_grad_() for t in heads]
            loss, kl, lp = chunked_text_objective(*copies, actions, direct, credit,
                special_ids=(8, 9, 10), executable_vocab=11, chunk_size=chunk)
            gradients = torch.autograd.grad(loss, copies[:3])
            values.append((loss.detach(), kl, lp, *gradients))
        for candidate in values[1:]:
            for actual, expected in zip(candidate, values[0]):
                torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)

    def test_full_vocabulary_gradient_not_sampled_label_only(self):
        heads, actions, direct, credit = self.fixture()
        direct = torch.ones_like(direct)
        credit = torch.zeros_like(credit)
        loss, _, _ = chunked_text_objective(*heads, torch.zeros_like(actions), direct, credit,
            special_ids=(8, 9, 10), executable_vocab=11, chunk_size=3)
        loss.backward()
        self.assertTrue(bool((heads[1].grad[:8].abs().sum(-1) > 1e-8).all()))
        self.assertTrue(bool((heads[2].grad.abs().sum(-1) > 1e-8).all()))

    def test_zero_direct_loss_still_has_sampled_score_gradient(self):
        heads, actions, direct, credit = self.fixture()
        copies = [t.detach().clone().requires_grad_() for t in heads]
        direct = torch.zeros_like(direct)
        loss, _, _ = chunked_text_objective(*heads, actions, direct, credit,
            special_ids=(8, 9, 10), executable_vocab=11, chunk_size=3)
        reference, _, _ = dense_reference(copies, actions, direct, credit, (8, 9, 10), 11)
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward(); reference.backward()
        self.assertGreater(float(heads[0].grad.abs().sum()), .01)
        for a, b in zip(heads[:3], copies[:3]):
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-12, atol=2e-12)

    def test_no_grad_scoring_performs_no_gradient_gemms(self):
        heads, actions, direct, credit = self.fixture()
        with torch.no_grad(), mock.patch('finetune.pose_opd_credit.text_head._mm_accum',
                                          side_effect=AssertionError('gradient GEMM during no-grad scoring')):
            loss, kl, lp = chunked_text_objective(*heads, actions, direct, credit,
                special_ids=(8, 9, 10), executable_vocab=11)
        self.assertFalse(loss.requires_grad)
        self.assertTrue(bool(torch.isfinite(kl).all() & torch.isfinite(lp).all()))

    def test_individual_gradient_ownership(self):
        for trained in range(3):
            with self.subTest(trained=trained):
                heads, actions, direct, credit = self.fixture()
                heads = [t.detach().requires_grad_(i == trained or i >= 3) for i, t in enumerate(heads)]
                reference_heads = [t.detach().clone().requires_grad_() for t in heads]
                loss, _, _ = chunked_text_objective(*heads, actions, direct, credit,
                    special_ids=(8, 9, 10), executable_vocab=11)
                expected, _, _ = dense_reference(reference_heads, actions, direct, credit, (8, 9, 10), 11)
                loss.backward(); expected.backward()
                torch.testing.assert_close(heads[trained].grad, reference_heads[trained].grad,
                                           rtol=2e-12, atol=2e-12)
                self.assertTrue(all(t.grad is None for i, t in enumerate(heads) if i != trained))

    def test_dtype_fp32_and_bfloat16(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                heads, actions, direct, credit = self.fixture(dtype)
                copies = [t.detach().clone().requires_grad_() for t in heads]
                actual, kl, lp = chunked_text_objective(*heads, actions, direct, credit,
                    special_ids=(8, 9, 10), executable_vocab=11, chunk_size=3)
                expected, _, _ = dense_reference(copies, actions, direct, credit, (8, 9, 10), 11)
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(kl.dtype, torch.float32)
                self.assertEqual(lp.dtype, torch.float32)
                actual.backward(); expected.backward()
                tolerance = 1e-6 if dtype == torch.float32 else .004
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                for a, b in zip(heads[:3], copies[:3]):
                    self.assertEqual(a.grad.dtype, dtype)
                    self.assertTrue(bool(torch.isfinite(a.grad).all()))
                    torch.testing.assert_close(a.grad, b.grad, rtol=tolerance, atol=tolerance)

    def test_empty_special_rows_and_empty_positions(self):
        heads, actions, direct, credit = self.fixture(special_count=0)
        clones = [t.detach().clone().requires_grad_() for t in heads]
        loss, _, _ = chunked_text_objective(*heads, actions, direct, credit,
            special_ids=(), executable_vocab=11, chunk_size=3)
        ref, _, _ = dense_reference(clones, actions, direct, credit, (), 11)
        loss.backward(); ref.backward()
        for a, b in zip(heads[:2], clones[:2]):
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-12, atol=2e-12)
        heads[0] = heads[0].detach()[:0].requires_grad_()
        heads[3] = heads[3].detach()[:0]
        empty, kl, lp = chunked_text_objective(*heads, actions[:0], direct[:0], credit[:0],
            special_ids=(), executable_vocab=11)
        empty.backward()
        self.assertEqual(float(empty.detach()), 0.)
        self.assertEqual(kl.numel(), 0)
        self.assertEqual(lp.numel(), 0)

    def test_first_order_only(self):
        heads, actions, direct, credit = self.fixture()
        loss, _, _ = chunked_text_objective(*heads, actions, direct, credit,
            special_ids=(8, 9, 10), executable_vocab=11)
        grad = torch.autograd.grad(loss, heads[0], create_graph=True)[0]
        self.assertFalse(grad.requires_grad)
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(grad.sum(), heads[0])

    def test_validation_rejects_invalid_support(self):
        heads, actions, direct, credit = self.fixture()
        for ids in ((8, 8, 10), (-1, 9, 10), (8, 9, 11)):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                chunked_text_objective(*heads, actions, direct, credit,
                                      special_ids=ids, executable_vocab=11)
        with self.assertRaises(ValueError):
            chunked_text_objective(*heads, actions + 11, direct, credit,
                                  special_ids=(8, 9, 10), executable_vocab=11)

    def test_dense_analytic_logit_gradient(self):
        student = torch.randn(4, 7, dtype=torch.float64, requires_grad=True)
        teacher = torch.randn(4, 7, dtype=torch.float64, requires_grad=True)
        direct = torch.tensor([.2, .0, .5, .1], dtype=torch.float64)
        credit = torch.tensor([.4, .3, .0, .2], dtype=torch.float64)
        actions = torch.tensor([2, 3, 0, 6])
        loss, _, _, gradient = logits_objective(student, teacher, actions, direct, credit)
        actual = torch.autograd.grad(loss, student)[0]
        torch.testing.assert_close(actual, gradient, rtol=2e-12, atol=2e-12)


if __name__ == '__main__':
    unittest.main()
