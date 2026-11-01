"""Finite-credit mathematical checks, independent of CUDA/model implementation."""
import itertools
import unittest

import torch

from finetune.pose_opd_credit.objectives import (
    AUDIO, POSE, TEXT, DecisionLayout, categorical_kl, causal_credit_weights,
    finite_credit_loss, sampled_log_probs,
)


def layout(entries):
    """entries are (row, native model frame, stream, codebook)."""
    columns = list(zip(*entries)) if entries else [[], [], [], []]
    return DecisionLayout(*(torch.tensor(x, dtype=torch.int64) for x in columns))


def reference_credit(costs, losses, actions, previous, valid=None, sampled=None):
    """Independent explicit ancestry predicate, used only on small fixtures."""
    valid = torch.ones(len(losses), dtype=torch.bool) if valid is None else valid
    sampled = torch.ones(len(actions), dtype=torch.bool) if sampled is None else sampled
    result = torch.zeros(len(actions), dtype=torch.float64)
    for a in range(len(actions)):
        for k in range(len(losses)):
            if not sampled[a] or not valid[k] or actions.rows[a] != losses.rows[k]:
                continue
            gap = int(losses.frames[k] - actions.frames[a])
            same_frame = gap == 0 and (
                (actions.streams[a] == TEXT and losses.streams[k] != TEXT)
                or (actions.streams[a] == losses.streams[k] and actions.streams[a] != TEXT
                    and actions.codebooks[a] < losses.codebooks[k]))
            if 1 <= gap <= previous or same_frame:
                result[a] += costs[k].detach().double()
    return result


class CategoricalTests(unittest.TestCase):
    def test_reverse_direction_and_teacher_detached(self):
        student = torch.tensor([[1.7, -0.8, 0.4]], dtype=torch.float64, requires_grad=True)
        teacher = torch.tensor([[-0.5, 1.5, 0.2]], dtype=torch.float64, requires_grad=True)
        actual = categorical_kl(student, teacher)
        p, q = student.softmax(-1), teacher.softmax(-1)
        expected = (p * (p.log() - q.log())).sum(-1)
        opposite = (q * (q.log() - p.log())).sum(-1)
        torch.testing.assert_close(actual, expected)
        self.assertGreater(float((actual - opposite).detach().abs().max()), 0.01)
        actual.sum().backward()
        self.assertIsNone(teacher.grad)
        torch.testing.assert_close(student.grad, p.detach() * (
            student.detach().log_softmax(-1) - teacher.detach().log_softmax(-1) - actual.detach()[:, None]))

    def test_full_distribution_and_exact_sample_logprob(self):
        logits = torch.tensor([[0.1, -0.4, 0.8, -2.0]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[0.0, 0.2, -0.4, 1.0]], dtype=torch.float64)
        loss = categorical_kl(logits, target).sum()
        loss.backward()
        # Every category participates, including ones not selected by a draw.
        self.assertTrue(bool((logits.grad.abs() > 1e-6).all()))
        ids = torch.tensor([2])
        actual = sampled_log_probs(logits, ids)
        torch.testing.assert_close(actual, logits.log_softmax(-1)[:, 2])

    def test_joint_zero_probability_support(self):
        student = torch.tensor([[0.2, -torch.inf, 0.7]], dtype=torch.float64, requires_grad=True)
        teacher = torch.tensor([[0.5, -torch.inf, -0.2]], dtype=torch.float64)
        loss = categorical_kl(student, teacher)
        loss.sum().backward()
        self.assertTrue(bool(torch.isfinite(loss).all()))
        self.assertTrue(bool(torch.isfinite(student.grad).all()))
        self.assertEqual(float(student.grad[0, 1]), 0.)


class CausalCreditTests(unittest.TestCase):
    def test_codebook14_actual_same_frame_ancestors(self):
        entries = [(0, 5, TEXT, -1)] + [(0, 5, s, c) for s in (AUDIO, POSE) for c in range(32)]
        actions = layout(entries)
        loss = layout([(0, 5, POSE, 14)])
        weights = causal_credit_weights(torch.tensor([2.], dtype=torch.float64), loss, actions)
        expected = torch.tensor([2.] + [0.] * 32 + [2.] * 14 + [0.] * 18, dtype=torch.float64)
        torch.testing.assert_close(weights, expected)
        # Own draw, later codebooks, and current audio receive no pose credit.
        self.assertEqual(float(weights[1 + 32 + 14]), 0.)

    def test_future_text_credits_both_media_branches(self):
        actions = layout([(0, 3, AUDIO, 31), (0, 3, POSE, 31), (0, 4, TEXT, -1)])
        losses = layout([(0, 4, TEXT, -1)])
        result = causal_credit_weights(torch.tensor([3.]), losses, actions, lookback_frames=1)
        torch.testing.assert_close(result, torch.tensor([3., 3., 0.], dtype=torch.float64))

    def test_default_is_eight_preceding_frames_plus_current_ancestors(self):
        actions = layout([(0, 0, TEXT, -1), (0, 1, POSE, 31),
                          (0, 8, AUDIO, 31), (0, 9, TEXT, -1),
                          (0, 9, POSE, 0), (0, 9, POSE, 1), (0, 9, AUDIO, 0)])
        losses = layout([(0, 9, POSE, 1)])
        result = causal_credit_weights(torch.tensor([2.]), losses, actions)
        torch.testing.assert_close(result,
            torch.tensor([0., 2., 2., 2., 2., 0., 0.], dtype=torch.float64))

    def test_window_boundaries_and_sparse_rows(self):
        actions = layout([(9, -100, TEXT, -1), (9, 0, POSE, 0), (9, 1, AUDIO, 1),
                          (9, 9, TEXT, -1), (99999, 9, TEXT, -1), (9, 10, TEXT, -1)])
        losses = layout([(9, 10, POSE, 0), (99999, 10, AUDIO, 0), (9, 1_000_000, TEXT, -1)])
        costs = torch.tensor([2., 7., 50.], dtype=torch.float64)
        prev = causal_credit_weights(costs, losses, actions, lookback_frames=10)
        incl = causal_credit_weights(costs, losses, actions, lookback_frames=10,
                                     window_mode='including_current')
        torch.testing.assert_close(prev, torch.tensor([0., 2., 2., 2., 7., 2.], dtype=torch.float64))
        torch.testing.assert_close(incl, torch.tensor([0., 0., 2., 2., 7., 2.], dtype=torch.float64))

    def test_vectorized_matches_independent_ancestry_on_random_order(self):
        generator = torch.Generator().manual_seed(81)
        entries = [(r, f, s, c) for r in (0, 14) for f in (0, 2, 3, 10)
                   for s, codes in ((TEXT, [-1]), (AUDIO, range(4)), (POSE, range(4))) for c in codes]
        order = torch.randperm(len(entries), generator=generator).tolist()
        actions = layout([entries[i] for i in order])
        losses = layout(entries + entries[5:10])  # also test repeated loss coordinates
        costs = torch.rand(len(losses), generator=generator, dtype=torch.float64)
        valid = torch.rand(len(losses), generator=generator) > .2
        sampled = torch.rand(len(actions), generator=generator) > .2
        for previous in (0, 1, 3, 10):
            actual = causal_credit_weights(costs, losses, actions, lookback_frames=previous,
                                            loss_mask=valid, sampled_mask=sampled)
            expected = reference_credit(costs, losses, actions, previous, valid, sampled)
            torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)

    def test_prefill_masks_account_for_two_frame_media_delay(self):
        entries = [(0, f, s, c) for f in range(24, 29)
                   for s, c in ((TEXT, -1), (AUDIO, 0), (POSE, 0))]
        coords = layout(entries)
        # At 12.5Hz, the initial 25 physical frames are forced. Text samples at
        # model frame25; the two-frame-delayed media first samples at frame27.
        sampled = torch.tensor([f >= (25 if s == TEXT else 27) for _, f, s, _ in entries])
        valid = sampled.clone()
        costs = torch.ones(len(coords), dtype=torch.float64)
        result = causal_credit_weights(costs, coords, sampled_mask=sampled, loss_mask=valid)
        expected = reference_credit(costs, coords, coords, 10, valid, sampled)
        torch.testing.assert_close(result, expected)
        self.assertTrue(bool((result[~sampled] == 0).all()))

    def test_surrogate_zero_detached_cost_and_masks(self):
        coords = layout([(0, 0, TEXT, -1), (0, 0, POSE, 0), (0, 1, TEXT, -1)])
        kl = torch.tensor([2., 3., 5.], dtype=torch.float64, requires_grad=True)
        logp = torch.tensor([-torch.inf, -.7, -.2], dtype=torch.float64, requires_grad=True)
        weight = torch.tensor([1., .25, .5], dtype=torch.float64, requires_grad=True)
        result = finite_credit_loss(kl, logp, coords, position_weights=weight,
                                   loss_mask=torch.tensor([False, True, True]),
                                   sampled_mask=torch.tensor([False, True, True]))
        self.assertEqual(float(result.credit_surrogate.detach()), 0.)
        self.assertEqual(float(result.loss.detach()), 3.25)
        result.loss.backward()
        torch.testing.assert_close(kl.grad, torch.tensor([0., .25, .5], dtype=torch.float64))
        torch.testing.assert_close(logp.grad, torch.tensor([0., 2.5, 0.], dtype=torch.float64))
        self.assertIsNone(weight.grad)
        self.assertFalse(result.credit_weights.requires_grad)

    def test_empty_layouts(self):
        no = layout([])
        one = layout([(0, 0, TEXT, -1)])
        self.assertEqual(len(causal_credit_weights(torch.zeros(0), no)), 0)
        torch.testing.assert_close(causal_credit_weights(torch.zeros(0), no, one),
                                   torch.zeros(1, dtype=torch.float64))

    def test_no_credit_across_finite_cutoff_but_direct_gradient_retained(self):
        coords = layout([(0, 0, TEXT, -1), (0, 11, TEXT, -1)])
        theta = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
        # The direct KL may depend on a shared parameter also used in old
        # frames. Cutting score credit does not detach this direct gradient.
        kl = torch.stack((theta * 0, theta.square()))
        logp = torch.tensor([-.2, -.5], dtype=torch.float64, requires_grad=True)
        result = finite_credit_loss(kl, logp, coords, lookback_frames=10)
        result.loss.backward()
        torch.testing.assert_close(logp.grad, torch.zeros(2, dtype=torch.float64))
        self.assertAlmostEqual(float(theta.grad), .6)

    def test_exact_enumerated_expected_gradient_of_sampling_dag(self):
        # One frame has text and two independent two-codebook branches; the
        # next text decision depends on all previous outcomes. Exhaustive
        # enumeration supplies an independent exact expected-gradient oracle.
        coords = layout([(0, 0, TEXT, -1), (0, 0, AUDIO, 0), (0, 0, AUDIO, 1),
                         (0, 0, POSE, 0), (0, 0, POSE, 1), (0, 1, TEXT, -1)])
        theta = torch.tensor([[.2, -.3], [.4, .1], [-.4, .5], [.1, -.7],
                              [.6, -.2], [-.1, .4]], dtype=torch.float64, requires_grad=True)
        exact, estimated = theta.new_zeros(()), theta.new_zeros(())
        for path in itertools.product(range(2), repeat=6):
            text, a0, a1, p0, p1, _ = path
            context = theta.new_tensor([0., .7 * text, .4 * text + .6 * a0,
                                       -.2 * text, .3 * text + .5 * p0,
                                       .2 * text + .4 * (a0 + a1) + .3 * (p0 + p1)])
            logits = theta + torch.stack((context, -context), -1)
            # Teacher targets also depend on the sampled history, but teachers
            # themselves are constants in the differentiable replay.
            teacher = torch.stack((.2 - context * .3, -.3 + context * .4), -1)
            kl = categorical_kl(logits, teacher)
            lp = sampled_log_probs(logits, torch.tensor(path))
            probability = lp.sum().exp()
            exact = exact + probability * kl.sum()
            surrogate = finite_credit_loss(kl, lp, coords, lookback_frames=1)
            estimated = estimated + probability.detach() * surrogate.loss
        expected_gradient = torch.autograd.grad(exact, theta, retain_graph=True)[0]
        actual_gradient = torch.autograd.grad(estimated, theta)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-12, atol=2e-12)


if __name__ == '__main__':
    unittest.main()
