"""Packed CFG and teacher/student batch independence on causal toy PPMs."""
import contextlib
import copy
import datetime
from dataclasses import replace
import os
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from finetune.pose_opd_credit import runtime
from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.data import teacher_views
from finetune.pose_opd_credit.teacher_batch import pack_cfg_batch, split_pose_outputs
from finetune.pose_opd_credit.tests.test_runtime import ToyPPM, fixture


class TeacherBatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(418)
        self.models = [ToyPPM(), ToyPPM().requires_grad_(False), ToyPPM().requires_grad_(False)]
        self.config = TrainingConfig(decoder_chunk_frames=3, text_chunk_frames=2,
            effective_batch_per_rank=4, teacher_batch_per_rank=4, teacher_depth_cache='cpu',
            pose_cfg_batch=True, pose_teacher_cfg_scale=3,
            pose_teacher_silence_path='fixture.npy', pose_teacher_silence_sha256='0' * 64)
        self.rows, self.latents = map(list, zip(*(fixture(frames=n) for n in (6, 8, 5, 7))))
        self.device = torch.device('cpu')
        self.normalization = runtime.update_normalization(self.rows, self.device)
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(mock.patch.object(runtime, 'resident_decoders',
                                  side_effect=lambda *_: contextlib.nullcontext()))
        self.stack.enter_context(mock.patch.object(runtime, 'load_silence_codes',
                                  return_value=np.zeros((32, 24), np.int64)))
        self.addCleanup(self.stack.close)

    def test_packed_cfg_preserves_row_boundaries_positions_and_outputs(self):
        batch, _, _ = runtime.prepare(self.rows, self.latents, self.config, self.device,
                                      world=1, normalization=self.normalization)
        conditional = batch['pose_teacher']
        unconditional = runtime.silent_pose_view(conditional, np.zeros((32, 24), np.int64))
        packed = pack_cfg_batch(conditional, unconditional)
        total = sum(conditional['segs'])
        self.assertEqual(len(packed['segs']), 8)
        self.assertEqual(packed['segs'], conditional['segs'] * 2)
        torch.testing.assert_close(packed['pid'][:, :total], packed['pid'][:, total:], rtol=0, atol=0)
        for key in ('ids', 'present'):
            torch.testing.assert_close(packed[key][:, :total], packed[key][:, total:], rtol=0, atol=0)
        torch.testing.assert_close(packed['pdepth'][:total], packed['pdepth'][total:], rtol=0, atol=0)
        torch.testing.assert_close(packed['sel'][len(conditional['sel']):], conditional['sel'] + total)
        with torch.no_grad():
            both = self.models[2](packed, None, role='pose_teacher')
            actual = split_pose_outputs(both, total)
            expected = [self.models[2](view, None, role='pose_teacher') for view in (conditional, unconditional)]
        for left, right in zip(actual, expected):
            for key in ('hv_p', 'logits_p', 'zb_p'):
                torch.testing.assert_close(left[key], right[key], atol=2e-7, rtol=2e-6)

    def test_prepared_full4_teacher_matches_separate_cfg_for_student_batches_1_2_4(self):
        expected = copy.deepcopy(self.models)
        reference_config = replace(self.config, pose_cfg_batch=False)
        metrics, _, _ = runtime.replay_update(*expected, self.rows, self.latents, reference_config,
            None, self.device, normalization=self.normalization)
        for physical in (1, 2, 4):
            with self.subTest(physical_microbatch=physical):
                models = copy.deepcopy(self.models)
                config = replace(self.config, physical_microbatch=physical)
                prepared = runtime.prepare_teacher_update(*models[1:], self.rows, self.latents, config,
                    None, self.device, normalization=self.normalization)
                self.assertEqual(models[1].calls, [('audio_teacher', False)])
                self.assertEqual(models[2].calls, [('pose_teacher', False)])
                self.assertEqual(len(models[2].segment_history[0]), 8)
                calls = (len(models[1].dec_s.calls), len(models[2].dec_p.calls))
                self.assertEqual(calls, (6, 6))
                self.assertTrue(all(len(flat) == 6 and not grad for flat, grad in models[2].dec_p.calls))
                self.assertNotIn('hv_p', prepared.pose)
                self.assertIsNone(prepared.unconditional)
                self.assertEqual(prepared.cache['p'].dtype, torch.float32)
                with self.assertRaises(TypeError):
                    prepared.pose['logits_p'] = None
                actual_loss = 0.
                for begin in range(0, 4, physical):
                    result, _, _ = runtime.replay_update(*models, self.rows[begin:begin + physical],
                        self.latents[begin:begin + physical], config, None, self.device,
                        normalization=self.normalization,
                        prepared_teachers=prepared.slice_rows(begin, begin + physical))
                    actual_loss += result['loss/direct']
                self.assertAlmostEqual(actual_loss, metrics['loss/direct'], places=6)
                self.assertEqual(calls, (len(models[1].dec_s.calls), len(models[2].dec_p.calls)))
                self.assertEqual(len(models[1].calls), 1)
                self.assertEqual(len(models[2].calls), 1)
                for (name, actual), (_, target) in zip(models[0].named_parameters(), expected[0].named_parameters()):
                    with self.subTest(parameter=name):
                        torch.testing.assert_close(actual.grad, target.grad, rtol=4e-5, atol=4e-6)

    def test_teacher_batch2_and_no_cache_still_independent_of_student_batch(self):
        reference = None
        for teacher_batch, cache in ((4, 'cpu'), (2, 'cpu'), (2, 'none')):
            models = copy.deepcopy(self.models)
            config = replace(self.config, teacher_batch_per_rank=teacher_batch, teacher_depth_cache=cache,
                             physical_microbatch=1)
            prepared = runtime.prepare_teacher_update(*models[1:], self.rows, self.latents, config,
                None, self.device, normalization=self.normalization)
            self.assertEqual(len(models[1].calls), 4 // teacher_batch)
            self.assertEqual(len(models[2].calls), 4 // teacher_batch)
            # This student call crosses a teacher-batch boundary when batch=2.
            for begin, end in ((0, 1), (1, 3), (3, 4)):
                runtime.replay_update(*models, self.rows[begin:end], self.latents[begin:end], config,
                    None, self.device, normalization=self.normalization,
                    prepared_teachers=prepared.slice_rows(begin, end))
            gradients = [parameter.grad for parameter in models[0].parameters()]
            if reference is None:
                reference = gradients
            else:
                for actual, expected in zip(gradients, reference):
                    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-6)

    def test_cached_padding_uses_first_valid_frame_and_rejects_bad_slices(self):
        prepared = runtime.prepare_teacher_update(*self.models[1:], self.rows, self.latents, self.config,
            None, self.device, normalization=self.normalization)
        view = prepared.slice_rows(1, 2)  # six generated frames
        full_padding = view.depth_chunk('p', 3, 3, self.device)
        expected = prepared.cache['p'][4:5].expand(3, -1, -1)
        torch.testing.assert_close(full_padding, expected, rtol=0, atol=0)
        partial = view.depth_chunk('p', 1, 4, self.device)
        torch.testing.assert_close(partial[:2], prepared.cache['p'][8:10], rtol=0, atol=0)
        torch.testing.assert_close(partial[2:], prepared.cache['p'][4:5].expand(2, -1, -1), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            prepared.slice_rows(2, 2)
        with self.assertRaises(ValueError):
            replace(self.config, teacher_batch_per_rank=3).validate()
        with self.assertRaises(ValueError):
            replace(self.config, pose_cfg_batch=1).validate()

    def test_eight_teacher_rows_cfg16_match_student_batches2_4_8_with_dummy_tail(self):
        rows, latents = map(list, zip(*(fixture(frames=n, dummy=i == 7)
            for i, n in enumerate((6, 8, 5, 7, 9, 4, 6, 10)))))
        config = replace(self.config, effective_batch_per_rank=8, teacher_batch_per_rank=8)
        normalization = runtime.update_normalization(rows, self.device)
        reference = copy.deepcopy(self.models)
        expected, _, _ = runtime.replay_update(*reference, rows, latents,
            replace(config, pose_cfg_batch=False), None, self.device, normalization=normalization)
        for physical in (2, 4, 8):
            with self.subTest(physical_microbatch=physical):
                models = copy.deepcopy(self.models)
                candidate = replace(config, physical_microbatch=physical).validate()
                prepared = runtime.prepare_teacher_update(*models[1:], rows, latents, candidate,
                    None, self.device, normalization=normalization)
                self.assertEqual(models[1].segment_history, [[len(row['student']['pid']) for row in rows]])
                self.assertEqual(models[2].segment_history, [[len(row['student']['pid']) for row in rows] * 2])
                self.assertEqual(len(models[1].calls), 1)
                self.assertEqual(len(models[2].calls), 1)
                loss = 0.
                for begin in range(0, 8, physical):
                    metrics, _, _ = runtime.replay_update(*models, rows[begin:begin + physical],
                        latents[begin:begin + physical], candidate, None, self.device,
                        normalization=normalization,
                        prepared_teachers=prepared.slice_rows(begin, begin + physical))
                    loss += metrics['loss/direct']
                self.assertAlmostEqual(loss, expected['loss/direct'], places=6)
                for actual, wanted in zip(models[0].parameters(), reference[0].parameters()):
                    torch.testing.assert_close(actual.grad, wanted.grad, rtol=5e-5, atol=5e-6)
                self.assertEqual(len(models[1].calls), 1)
                self.assertEqual(len(models[2].calls), 1)

    def test_static_padding_preserves_variable_prefix_gradients_and_probe_positions_cfg16(self):
        # The small fixture vocabulary needs absence mapped to a valid index
        # before masking, just as the production code embedding does.
        self.stack.enter_context(mock.patch.object(ToyPPM, 'media_embedding', staticmethod(
            lambda module, codes: (module(codes.clamp_min(0).remainder(2048)) *
                                   (codes >= 0)[..., None]).mean(1))))
        rows, features = [], []
        for index, frames in enumerate((6, 8, 5, 7, 9, 4, 6, 10)):
            bundle, latent = fixture(frames=frames, dummy=index == 7)
            raw = copy.deepcopy(bundle['student'])
            extra = index % 5
            for key, fill in (('ids', 0), ('depth', -1), ('present', 0)):
                widths = [(0, 0), (extra, 0)] + [(0, 0)] * (raw[key].ndim - 2)
                raw[key] = np.pad(raw[key], widths, constant_values=fill)
            raw['pdepth'] = np.pad(raw['pdepth'], ((extra, 0), (0, 0)), constant_values=-1)
            raw['pid'] = np.arange(len(raw['pid']) + extra)
            raw['segs'] = [len(raw['pid'])]
            raw['vin'][0] += extra
            raw['pad_id'] = 0
            for key in ('sel', 'text_positions', 'light', 'plight'):
                raw[key] += extra
            raw['sampled_masks'] = {key: np.pad(value, (extra, 0)) for key, value in raw['sampled_masks'].items()}
            rows.append(teacher_views(raw, 0))
            features.append(latent)
        normalization = runtime.update_normalization(rows, self.device)
        fixed = max(len(row['student']['pid']) for row in rows) + 7
        config = replace(self.config, effective_batch_per_rank=8, teacher_batch_per_rank=8,
                         physical_microbatch=8)
        observations = []

        class Probes:
            def __init__(self):
                self.values = {}
            def text(self, *_):
                pass
            def cb0(self, stream, logits):
                self.values[(stream, 0)] = logits.clone()
            def depth(self, stream, positions, logits):
                for position, value in zip(positions.tolist(), logits):
                    self.values[(stream, position, 'depth')] = value.clone()

        for target in (None, fixed):
            candidate = replace(config, fixed_sequence_frames=target)
            models = copy.deepcopy(self.models)
            prepared = runtime.prepare_teacher_update(*models[1:], rows, features, candidate,
                None, self.device, normalization=normalization)
            probes = Probes()
            metrics, lp, mask = runtime.replay_update(*models, rows, features, candidate,
                None, self.device, normalization=normalization, policy_probes=probes,
                prepared_teachers=prepared.slice_rows(0, 8))
            self.assertEqual(len(lp), sum(len(row['student']['pid']) for row in rows))
            if target is not None:
                self.assertEqual(models[2].segment_history, [[fixed] * 16])
                self.assertEqual(models[0].segment_history, [[fixed] * 8] * 2)
                for row in rows:
                    padded = runtime.pad_replay_bundle(row, fixed)
                    length = len(row['student']['pid'])
                    self.assertEqual(padded['student']['segs'], [fixed])
                    np.testing.assert_array_equal(padded['student']['vin'], row['student']['vin'])
                    np.testing.assert_array_equal(padded['student']['pid'][:length], row['student']['pid'])
                    self.assertTrue(all(not value[length:].any() for value in padded['student']['sampled_masks'].values()))
                    self.assertTrue((padded['student']['depth'][:, length:] == -1).all())
                    self.assertTrue((padded['student']['pdepth'][length:] == -1).all())
                    self.assertTrue((padded['student']['ids'][:, length:] == 0).all())
                    self.assertTrue((padded['student']['present'][:, length:] == 0).all())
            observations.append((models, metrics, lp, mask, probes.values))
        before, after = observations
        self.assertAlmostEqual(before[1]['loss/direct'], after[1]['loss/direct'], places=6)
        torch.testing.assert_close(before[2][before[3]], after[2][after[3]], rtol=3e-6, atol=3e-6)
        self.assertEqual(before[4].keys(), after[4].keys())
        for key in before[4]:
            torch.testing.assert_close(before[4][key], after[4][key], rtol=3e-6, atol=3e-6)
        for actual, expected in zip(after[0][0].parameters(), before[0][0].parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=5e-5, atol=5e-6)
        with self.assertRaises(ValueError):
            runtime.pad_replay_bundle(rows[0], 1)
        shifted = copy.deepcopy(rows[0])
        for view in shifted.values():
            view['pid'] = view['pid'] + 100
        padded = runtime.pad_replay_bundle(shifted, fixed)
        np.testing.assert_array_equal(padded['student']['pid'], np.arange(100, 100 + fixed))


def _distributed_prepared_worker(rank, rendezvous):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FSDPModule, fully_shard
    from finetune.pose_opd_credit.tests.test_cfg import DenseGuidedTeacher
    from finetune.pose_opd_credit.tests.test_runtime import dense_expected
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=20))
    try:
        torch.manual_seed(117)
        models = [ToyPPM(), ToyPPM().requires_grad_(False), ToyPPM().requires_grad_(False)]
        reference = copy.deepcopy(models)
        reference[2] = DenseGuidedTeacher(reference[2], 3)
        mesh = init_device_mesh('cpu', (2,))
        for model in models:
            for child in model.children():
                fully_shard(child, mesh=mesh, reshard_after_forward=True)
            fully_shard(model, mesh=mesh, reshard_after_forward=True)
            for child in model.modules():
                if isinstance(child, FSDPModule):
                    child.set_force_sum_reduction_for_comms(True)
        torch.manual_seed(319 + rank)
        rows, latents = map(list, zip(*(fixture(frames=6 + rank + n) for n in range(4))))
        config = TrainingConfig(decoder_chunk_frames=3, effective_batch_per_rank=4,
            teacher_batch_per_rank=4, teacher_depth_cache='cpu', pose_cfg_batch=True,
            pose_teacher_cfg_scale=3, decoder_gradient_sync='per_replay',
            pose_teacher_silence_path='fixture.npy', pose_teacher_silence_sha256='0' * 64)
        normalization = runtime.update_normalization(rows, torch.device('cpu'))
        with mock.patch.object(runtime, 'load_silence_codes', return_value=np.zeros((32, 24), np.int64)):
            prepared = runtime.prepare_teacher_update(*models[1:], rows, latents, config,
                None, torch.device('cpu'), world=2, normalization=normalization)
        for i in range(4):
            runtime.replay_update(*models, rows[i:i + 1], latents[i:i + 1], config,
                None, torch.device('cpu'), world=2, normalization=normalization,
                prepared_teachers=prepared.slice_rows(i, i + 1))
        expected, _, _ = dense_expected(*reference, rows, latents, config, normalization=normalization)
        expected.backward()
        for (name, actual), (_, wanted) in zip(models[0].named_parameters(), reference[0].named_parameters()):
            dist.all_reduce(wanted.grad)
            torch.testing.assert_close(actual.grad.full_tensor(), wanted.grad, rtol=5e-5, atol=5e-6,
                                       msg=lambda message: f'{name}: {message}')
        if len(models[1].calls) != 1 or len(models[2].calls) != 1:
            raise AssertionError('Teacher temporal batches were recomputed in student accumulation')
        if any(parameter.grad is not None for model in models[1:] for parameter in model.parameters()):
            raise AssertionError('Frozen teacher received a gradient')
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'CPU Gloo is unavailable')
class DistributedTeacherBatchTests(unittest.TestCase):
    def test_real_fsdp_teacher4_cfg8_and_student1_match_dense_global_gradient(self):
        with tempfile.TemporaryDirectory(prefix='pose-opd-teacher-batch-') as directory, \
             mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}):
            context = mp.start_processes(_distributed_prepared_worker, args=(directory + '/rendezvous',),
                                         nprocs=2, start_method='spawn', join=False)
            deadline = time.monotonic() + 30
            try:
                while not context.join(timeout=1):
                    if time.monotonic() >= deadline:
                        self.fail('Teacher batch FSDP workers exceeded timeout')
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                for process in context.processes:
                    process.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
