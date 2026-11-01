"""Fixed-trajectory learner benchmarks and atomic CPU rollout cache receipts."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from finetune.pose_opd_credit.benchmark import (cache_identity, save_rollout_rank, commit_rollout_cache,
    load_rollout_rank, validate_rollout_cache, gradient_statistics, replay_candidates)
from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.train import args_parser, validate_replay_microbatches, warm_optimizer_state
from finetune.pose_opd_credit.tests.test_rollout import seed_row, FakeCachedEngine
from finetune.pose_opd_credit.rollout import rollout_row


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def identity(self, rank=0):
        return cache_identity({'method': TrainingConfig().to_dict(), 'world': 2, 'source_sha256': 'source',
                'plan_digest': 'plan', 'models': {'student': 'weights'}, 'dataset': {'sha256': 'dataset'}},
                update=0, rank=rank)

    def payload(self):
        seed = seed_row(frames=30)
        seed['dummy'] = True
        rollout = rollout_row(FakeCachedEngine(), seed, seed=81)
        return [(seed, torch.randn(1, 120, 1280, dtype=torch.bfloat16))], [rollout]

    def test_identity_relaxes_learner_scheduling_and_keeps_actual_actor_batch(self):
        source = {'method': TrainingConfig(effective_batch_per_rank=4, rollout_batch_per_rank=4).to_dict(),
                  'world': 2, 'source_sha256': 'source'}
        reference = cache_identity(source, update=0, rank=0)
        source['method']['physical_microbatch'] = 4
        self.assertEqual(reference, cache_identity(source, update=0, rank=0))
        source['method'].update(decoder_gradient_sync='per_replay', decoder_chunk_frames=256, text_chunk_frames=128)
        self.assertEqual(reference, cache_identity(source, update=0, rank=0))
        for key, value in (('actor_attention_precision', 'native_bf16'), ('seed', 42),
                           ('effective_batch_per_rank', 8), ('rollout_batch_per_rank', 2)):
            changed = copy.deepcopy(source)
            changed['method'][key] = value
            self.assertNotEqual(reference, cache_identity(changed, update=0, rank=0))
        source['method']['rollout_batch_per_rank'] = None
        reference = cache_identity(source, update=0, rank=0)
        source['method']['physical_microbatch'] = 2
        self.assertNotEqual(reference, cache_identity(source, update=0, rank=0))

    def test_complete_cache_roundtrip_preserves_exact_rows_masks_latents_and_rollouts(self):
        with tempfile.TemporaryDirectory() as directory:
            payload, rollouts = self.payload()
            identities = [self.identity(rank) for rank in range(2)]
            receipts = [save_rollout_rank(directory, ident, payload, rollouts) for ident in identities]
            with self.assertRaisesRegex(ValueError, 'globally agreed'):
                load_rollout_rank(directory, identities[0])
            with self.assertRaisesRegex(ValueError, 'every rank'):
                commit_rollout_cache(directory, identities[0], receipts[:1])
            for ident in identities:
                commit_rollout_cache(directory, ident, receipts)
            loaded, results = load_rollout_rank(directory, identities[1])
            self.assertTrue(loaded[0][0]['dummy'])
            for key in ('ids', 'depth', 'pdepth', 'vin', 'present'):
                np.testing.assert_array_equal(loaded[0][0][key], payload[0][0][key])
            torch.testing.assert_close(loaded[0][1], payload[0][1], rtol=0, atol=0)
            for stream in ('text', 'audio', 'pose'):
                np.testing.assert_array_equal(getattr(results[0], stream), getattr(rollouts[0], stream))
                np.testing.assert_array_equal(results[0].sampled_masks[stream], rollouts[0].sampled_masks[stream])
                np.testing.assert_array_equal(results[0].behavior_logprobs[stream], rollouts[0].behavior_logprobs[stream])
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                save_rollout_rank(directory, identities[0], payload, rollouts)

    def test_cache_rejects_changed_source_missing_receipt_and_corrupted_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            payload, rollouts = self.payload()
            identities = [self.identity(rank) for rank in range(2)]
            receipts = [save_rollout_rank(directory, ident, payload, rollouts) for ident in identities]
            commit_rollout_cache(directory, identities[0], receipts)
            changed = {**identities[0], 'source_sha256': 'other'}
            with self.assertRaisesRegex(ValueError, 'changes source'):
                load_rollout_rank(directory, changed)
            marker = Path(directory) / 'rank-00000.complete.json'
            original = marker.read_text()
            marker.unlink()
            with self.assertRaisesRegex(ValueError, 'receipt'):
                load_rollout_rank(directory, identities[0])
            marker.write_text(original)
            shard = Path(directory) / 'rank-00000.pt'
            data = bytearray(shard.read_bytes()); data[len(data) // 2] ^= 1
            shard.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                load_rollout_rank(directory, identities[0])

    def test_mismatched_rank_identity_never_gets_global_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            payload, rollouts = self.payload()
            a, b = self.identity(0), self.identity(1)
            b['plan_digest'] = 'different'
            receipts = [save_rollout_rank(directory, ident, payload, rollouts) for ident in (a, b)]
            with self.assertRaisesRegex(ValueError, 'disagree'):
                commit_rollout_cache(directory, a, receipts)
            self.assertFalse((Path(directory) / 'COMPLETE').exists())

    def test_replay_candidates_preserve_warm_adam_weights_and_state_until_final_step(self):
        from finetune.pose_opd_credit import runtime
        from finetune.pose_opd_credit.tests.test_runtime import ToyPPM, fixture
        torch.manual_seed(988)
        models = [ToyPPM(), ToyPPM().requires_grad_(False), ToyPPM().requires_grad_(False)]
        optimizer = torch.optim.AdamW(models[0].parameters(), lr=1e-3, weight_decay=0.)
        warm_optimizer_state(optimizer)
        initial = [parameter.detach().clone() for parameter in models[0].parameters()]
        examples = [fixture(frames=n) for n in (6, 8, 7, 9)]
        bundles, latents = zip(*examples)
        config = TrainingConfig(effective_batch_per_rank=4, decoder_chunk_frames=3)
        normalization = runtime.update_normalization(bundles, torch.device('cpu'))
        gradients = []
        for candidate in replay_candidates(config, (1, 2, 4), optimizer):
            self.assertTrue(all(parameter.grad is None for parameter in models[0].parameters()))
            size = candidate.physical_microbatch
            for begin in range(0, 4, size):
                runtime.replay_update(*models, bundles[begin:begin + size], latents[begin:begin + size],
                    candidate, None, torch.device('cpu'), normalization=normalization)
            gradients.append(torch.cat([parameter.grad.flatten() for parameter in models[0].parameters()]))
            stats = gradient_statistics(models[0].parameters(), torch.device('cpu'))
            self.assertAlmostEqual(stats['l2'], float(gradients[-1].double().norm()), places=7)
            self.assertEqual(stats['elements'], gradients[-1].numel())
            for before, parameter in zip(initial, models[0].parameters()):
                torch.testing.assert_close(before, parameter, rtol=0, atol=0)
                self.assertEqual(float(optimizer.state[parameter]['step']), 1.)
        for gradient in gradients[1:]:
            torch.testing.assert_close(gradient, gradients[0], rtol=3e-5, atol=3e-6)
        optimizer.step()
        self.assertTrue(all(float(state['step']) == 2. for state in optimizer.state.values()))
        self.assertTrue(any(not torch.equal(before, parameter) for before, parameter in zip(initial, models[0].parameters())))

    def test_cli_reuse_is_explicit_single_update_unsaved_and_nonresumable(self):
        base = ['--config', 'config.json', '--run-dir', '/tmp/benchmark']
        bounded = ['--no-save', '--max-updates', '1']
        args = args_parser(base + bounded + ['--benchmark-replay-microbatches', '1,2,4',
                           '--benchmark-save-rollouts', '/tmp/cache', '--warm-optimizer'])
        self.assertEqual(args.benchmark_replay_microbatches, [1, 2, 4])
        validate_replay_microbatches(TrainingConfig(effective_batch_per_rank=4), [1, 2, 4])
        with self.assertRaisesRegex(ValueError, 'divide'):
            validate_replay_microbatches(TrainingConfig(effective_batch_per_rank=4), [3])
        for options in (['--benchmark-replay-microbatches', '1,2'],
                        ['--benchmark-load-rollouts', '/tmp/cache'],
                        bounded + ['--benchmark-replay-microbatches', '0'],
                        bounded + ['--benchmark-replay-microbatches', 'x'],
                        bounded + ['--benchmark-replay-microbatches', '1,2', '--benchmark-microbatches', '1'],
                        bounded + ['--benchmark-save-rollouts', '/tmp/cache', '--resume', '/tmp/checkpoint'],
                        ['--no-save', '--max-updates', '2', '--benchmark-load-rollouts', '/tmp/cache'],
                        bounded + ['--benchmark-save-rollouts', '/tmp/a', '--benchmark-load-rollouts', '/tmp/b']):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                args_parser(base + options)


if __name__ == '__main__':
    unittest.main()
