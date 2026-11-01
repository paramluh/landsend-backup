"""Packed physical batches and accumulation equal independent row objectives."""
import copy
from dataclasses import replace
import unittest

import torch

from finetune.pose_opd_credit import runtime
from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.train import make_plan
from finetune.pose_opd_credit.tests.test_runtime import ToyPPM, dense_expected, fixture


class MicrobatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(812)

    def test_packed_two_rows_reset_positions_and_preserve_audio_alignment(self):
        examples = [fixture(frames=6), fixture(frames=9)]
        bundles, latents = zip(*examples)
        prepared, weights, masks = runtime.prepare(bundles, latents, TrainingConfig(), torch.device('cpu'), world=1)
        student = prepared['student']
        self.assertEqual(student['segs'], [10, 13])
        torch.testing.assert_close(student['pid'][0], torch.cat((torch.arange(10), torch.arange(13))))
        torch.testing.assert_close(student['vpos'], torch.cat((torch.arange(2, 8), torch.arange(12, 21))))
        self.assertEqual(tuple(student['vlat'].shape), (1, 60, 1280))
        layout = runtime.event_layout(23, torch.device('cpu'), student['segs'])
        self.assertEqual(layout.rows[10 * 65].item(), 1)
        self.assertEqual(layout.frames[10 * 65].item(), 0)
        self.assertEqual(weights.shape, masks.shape)
        for column, expected in ((0, 1 / 3), (1, 1 / 6), (33, 1 / 6)):
            self.assertAlmostEqual(float(weights[:, column].sum()), expected, places=6)

    def test_physical_batches_one_two_four_match_dense_effective_batch(self):
        examples = [fixture(frames=6), fixture(frames=9), fixture(frames=7), fixture(frames=8, dummy=True)]
        bundles, latents = zip(*examples)
        original = [ToyPPM(), ToyPPM().requires_grad_(False), ToyPPM().requires_grad_(False)]
        for reduction in ('row_mean', 'position_mean'):
            config = TrainingConfig(reduction=reduction, decoder_chunk_frames=3, effective_batch_per_rank=4)
            normalization = runtime.update_normalization(bundles, torch.device('cpu'))
            reference = copy.deepcopy(original)
            dense = sum(dense_expected(*reference, bundle, lat, config, normalization)[0]
                        for bundle, lat in examples)
            dense.backward()
            for size in (1, 2, 4):
                with self.subTest(reduction=reduction, physical_microbatch=size):
                    models = copy.deepcopy(original)
                    actual_config = replace(config, physical_microbatch=size)
                    observed_loss = 0.
                    for start in range(0, 4, size):
                        metrics, _, _ = runtime.replay_update(*models, bundles[start:start + size], latents[start:start + size],
                            actual_config, None, torch.device('cpu'), normalization=normalization)
                        observed_loss += metrics['loss/direct']
                    self.assertAlmostEqual(observed_loss, float(dense.detach()), places=6)
                    for (name, actual), (_, expected) in zip(models[0].named_parameters(), reference[0].named_parameters()):
                        torch.testing.assert_close(actual.grad, expected.grad, atol=3e-6, rtol=4e-5,
                            msg=lambda message: f'{name}: {message}')
                    # Each model forward actually receives all physical rows
                    # together; this assertion rules out a serial row loop.
                    self.assertTrue(all(len(segments) == size for segments in models[0].segment_history))
                    self.assertEqual(len(models[0].segment_history), 2 * (4 // size))
                    self.assertTrue(all(parameter.grad is None for teacher in models[1:] for parameter in teacher.parameters()))

    def test_large_batch_plan_preserves_coverage_and_effective_batch(self):
        items = [(0, i, i % 2) for i in range(173)]
        plan = make_plan(items, world=40, epochs=1, seed=2, rows_per_rank=4)
        self.assertEqual(plan['effective_batch'], 160)
        self.assertEqual([len(update) for update in plan['updates']], [160, 160])
        self.assertCountEqual([tuple(row) for update in plan['updates'] for row in update if row is not None], items)

    def test_invalid_batch_sizes_fail_configuration(self):
        for options in ({'physical_microbatch': 0}, {'effective_batch_per_rank': 0},
                        {'physical_microbatch': 3, 'effective_batch_per_rank': 4}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TrainingConfig(**options).validate()


if __name__ == '__main__':
    unittest.main()
