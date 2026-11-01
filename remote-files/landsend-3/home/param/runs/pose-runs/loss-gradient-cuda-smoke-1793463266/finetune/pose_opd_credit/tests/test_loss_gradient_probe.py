"""Cached loss decomposition leaves the optimizer update and RNG unchanged."""
import copy
import contextlib
import datetime
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.gradient_norms import COMPONENTS
from finetune.pose_opd_credit.loss_gradient_probe import (
    BACKWARD_TERMS, RESIDUAL_TERM, loss_gradient_probe_due, measure_loss_gradient_norms,
)
from finetune.pose_opd_credit.schedule import apply_learning_rate, resolve_schedule
from finetune.pose_opd_credit.train import optimizer_parameter_groups


TERMS = (*BACKWARD_TERMS, RESIDUAL_TERM)


class SmallStudent(nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.layers = nn.Linear(2, 3, dtype=dtype)
        self.dec_p = nn.Linear(3, 2, dtype=dtype)
        self.tower_p = nn.Linear(2, 2, dtype=dtype)
        self.register_parameter('unused', nn.Parameter(torch.ones(1, dtype=dtype)))


def _loss(model, term, microbatch=0):
    index = TERMS.index(term) + 1
    return sum(((parameter + microbatch * .03).square() * (index * .015)
                + parameter.sin() * ((-1) ** index) * .02).sum()
               for name, parameter in model.named_parameters() if name != 'unused')


def _dense_gradients(model, terms):
    parameters = list(model.parameters())
    gradients = torch.autograd.grad(sum(_loss(model, term, microbatch)
        for term in terms for microbatch in range(2)), parameters, allow_unused=True)
    return {name: value for (name, _), value in zip(model.named_parameters(), gradients, strict=True)}


def _distributed_probe_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard, Replicate

    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=30))
    try:
        mesh = init_device_mesh('cpu', (2,))
        model = nn.Module()
        originals = {}
        for name, values, placements in (
            ('layers', torch.arange(1., 8., dtype=torch.float64), [Shard(0)]),
            ('dec_p', torch.tensor([2., -3.], dtype=torch.float64), [Replicate()]),
        ):
            module = nn.Module()
            module.register_parameter('weight', nn.Parameter(distribute_tensor(values, mesh, placements)))
            model.add_module(name, module)
            originals[name] = values
        coefficients = {term: 2 ** index for index, term in enumerate(TERMS)}

        def backward(term):
            # Real DTensor autograd, with distinct shard/replica placements.
            torch.autograd.backward([p.square().sum() * coefficients[term] for p in model.parameters()])

        for term in TERMS:
            backward(term)
        before = [(p.grad.to_local().clone(), p.grad.placements, p.grad.shape, p.grad.stride())
                  for p in model.parameters()]
        calls = []
        def callback(term):
            calls.append(term)
            backward(term)
        metrics = measure_loss_gradient_norms(model, callback, 'cpu')
        assert tuple(calls) == BACKWARD_TERMS
        for parameter, (local, placements, shape, stride) in zip(model.parameters(), before, strict=True):
            assert parameter.grad.placements == placements and parameter.grad.shape == shape
            assert parameter.grad.stride() == stride
            torch.testing.assert_close(parameter.grad.to_local(), local, rtol=0., atol=0.)
        for term in TERMS:
            total = 0.
            for component, values in originals.items():
                squared = float((2 * coefficients[term] * values).square().sum())
                total += squared
                assert math.isclose(metrics[f'grad_norm/{term}/{component}'], math.sqrt(squared), rel_tol=1e-12)
            assert math.isclose(metrics[f'grad_norm/{term}/all'], math.sqrt(total), rel_tol=1e-12)
    finally:
        dist.destroy_process_group()


class LossGradientProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        torch.set_num_threads(1)

    def test_all_six_loss_norms_and_aggregates_match_dense_autograd(self):
        model = SmallStudent()
        reference = {term: _dense_gradients(model, [term]) for term in TERMS}
        reference.update(direct=_dense_gradients(model, TERMS[:3]), credit=_dense_gradients(model, TERMS[3:]))
        for microbatch in range(2):
            sum(_loss(model, term, microbatch) for term in TERMS).backward()
        calls = []
        def backward(term):
            calls.append(term)
            for microbatch in range(2):
                _loss(model, term, microbatch).backward()
        actual = measure_loss_gradient_norms(model, backward, 'cpu')
        self.assertEqual(tuple(calls), BACKWARD_TERMS)
        for term, gradients in reference.items():
            squares = {component: 0. for component in COMPONENTS}
            for name, gradient in gradients.items():
                if gradient is not None:
                    component = name.split('.')[0]
                    squares[component if component in squares else 'other'] += float(gradient.square().sum())
            for component, square in squares.items():
                self.assertAlmostEqual(actual[f'grad_norm/{term}/{component}'], math.sqrt(square), places=12)
            self.assertAlmostEqual(actual[f'grad_norm/{term}/all'], math.sqrt(sum(squares.values())), places=12)

    def test_probe_preserves_gradients_rng_weights_and_exact_next_adam_update(self):
        config = TrainingConfig(pose_light_learning_rate=8e-6, learning_rate_schedule='amu_opd')
        model = SmallStudent(torch.float32)
        optimizer = torch.optim.AdamW(optimizer_parameter_groups(model, config), betas=(.9, .95), weight_decay=0.)
        schedule = resolve_schedule(config, 123)
        sum(_loss(model, term) for term in TERMS).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        control = copy.deepcopy(model)
        control_optimizer = torch.optim.AdamW(optimizer_parameter_groups(control, config))
        control_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for target, opt in ((model, optimizer), (control, control_optimizer)):
            apply_learning_rate(opt, config.learning_rate, schedule, 9)
            for microbatch in range(2):
                sum(_loss(target, term, microbatch) for term in TERMS).backward()
        before = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
        weights = {key: value.clone() for key, value in model.state_dict().items()}
        state_before = copy.deepcopy(optimizer.state_dict())
        random.seed(12); np.random.seed(13); torch.manual_seed(14)
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        def callback(term):
            random.random(); np.random.rand(); torch.rand(2)
            for microbatch in range(2):
                _loss(model, term, microbatch).backward()
        metrics = measure_loss_gradient_norms(model, callback, 'cpu')
        self.assertEqual(metrics['loss_gradient_diagnostics_extra_backwards'], 5)
        self.assertEqual(random.getstate(), rng[0])
        np.testing.assert_array_equal(np.random.get_state()[1], rng[1][1])
        torch.testing.assert_close(torch.get_rng_state(), rng[2], rtol=0., atol=0.)
        for previous, parameter in zip(before, model.parameters(), strict=True):
            if previous is None:
                self.assertIsNone(parameter.grad)
            else:
                torch.testing.assert_close(parameter.grad, previous, rtol=0., atol=0.)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[name], rtol=0., atol=0.)
        self.assertEqual(optimizer.state_dict()['param_groups'], state_before['param_groups'])
        for key, values in optimizer.state_dict()['state'].items():
            for name, value in values.items():
                torch.testing.assert_close(value, state_before['state'][key][name], rtol=0., atol=0.)
        for opt in (optimizer, control_optimizer):
            opt.step()
        for actual, expected in zip(model.parameters(), control.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0., atol=0.)

    def test_failure_also_restores_none_gradients_and_rng(self):
        model = SmallStudent(torch.float32)
        sum(_loss(model, term) for term in TERMS).backward()
        gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
        state = torch.get_rng_state()
        def callback(term):
            _loss(model, term).backward()
            torch.rand(10)
            raise RuntimeError('injected probe failure')
        with self.assertRaisesRegex(RuntimeError, 'injected probe failure'):
            measure_loss_gradient_norms(model, callback, 'cpu')
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0., atol=0.)
        for parameter, expected in zip(model.parameters(), gradients, strict=True):
            if expected is None:
                self.assertIsNone(parameter.grad)
            else:
                torch.testing.assert_close(parameter.grad, expected, rtol=0., atol=0.)

    def test_cached_cfg_replay_probe_preserves_the_real_training_update(self):
        from finetune.pose_opd_credit import runtime
        from finetune.pose_opd_credit.tests.test_runtime import ToyPPM, fixture

        student, audio, pose = ToyPPM(), ToyPPM().requires_grad_(False), ToyPPM().requires_grad_(False)
        config = TrainingConfig(decoder_chunk_frames=3, text_chunk_frames=2,
            effective_batch_per_rank=2, physical_microbatch=1, teacher_batch_per_rank=2,
            teacher_depth_cache='cpu', pose_cfg_batch=True, pose_teacher_cfg_scale=3,
            pose_teacher_silence_path='fixture.npy', pose_teacher_silence_sha256='0' * 64,
            pose_light_learning_rate=8e-6)
        rows, latents = map(list, zip(*(fixture(frames=frames) for frames in (6, 8))))
        device = torch.device('cpu')
        normalization = runtime.update_normalization(rows, device)
        with mock.patch.object(runtime, 'resident_decoders', side_effect=lambda *_: contextlib.nullcontext()), \
                mock.patch.object(runtime, 'load_silence_codes', return_value=np.zeros((32, 24), np.int64)):
            prepared = runtime.prepare_teacher_update(audio, pose, rows, latents, config, None, device,
                                                      normalization=normalization)
            caches = []
            for index in range(2):
                cache = {}
                runtime.replay_update(student, audio, pose, rows[index:index + 1], latents[index:index + 1],
                    config, None, device, normalization=normalization,
                    prepared_teachers=prepared.slice_rows(index, index + 1), loss_gradient_cache=cache)
                caches.append(cache)
            control = copy.deepcopy(student)
            for actual, expected in zip(control.parameters(), student.parameters(), strict=True):
                actual.grad = None if expected.grad is None else expected.grad.clone()
            initial_calls = len(student.calls)

            def callback(term):
                for cache in caches:
                    runtime.replay_loss_gradient(student, cache, term, None, device)

            with mock.patch.object(audio, 'forward', side_effect=AssertionError('Teacher forward')), \
                    mock.patch.object(pose, 'forward', side_effect=AssertionError('Teacher forward')), \
                    mock.patch.object(runtime, 'prepare_teacher_update', side_effect=AssertionError('Teacher refresh')), \
                    mock.patch.object(runtime, 'scored_categorical_statistics', side_effect=AssertionError('Student rescoring')), \
                    mock.patch.object(runtime, 'causal_credit_weights', side_effect=AssertionError('New credit')):
                metrics = measure_loss_gradient_norms(student, callback, device)
            self.assertEqual(len(student.calls) - initial_calls, 10)  # five terms × two cached microbatches
            self.assertGreater(metrics['grad_norm/direct/pose/dec_p'], 0.)
            for actual, expected in zip(student.parameters(), control.parameters(), strict=True):
                if expected.grad is None:
                    self.assertIsNone(actual.grad)
                else:
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=0., atol=0.)
            for target in (student, control):
                optimizer = torch.optim.AdamW(optimizer_parameter_groups(target, config),
                                             betas=(.9, .95), eps=1e-8, weight_decay=0.)
                torch.nn.utils.clip_grad_norm_(target.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            for actual, expected in zip(student.parameters(), control.parameters(), strict=True):
                torch.testing.assert_close(actual, expected, rtol=0., atol=0.)

    def test_absolute_update_cadence_does_not_reset_at_resume_or_epoch(self):
        config = TrainingConfig(loss_gradient_interval_steps=10)
        self.assertEqual([step for step in range(1, 124) if loss_gradient_probe_due(config, step)],
                         list(range(10, 121, 10)))
        for resume in (9, 40, 60, 81):
            self.assertEqual([step for step in range(resume + 1, 124) if loss_gradient_probe_due(config, step)],
                             [step for step in range(10, 121, 10) if step > resume])
        self.assertFalse(loss_gradient_probe_due(TrainingConfig(), 10))

    def test_dtensor_shard_and_replica_gradients_are_restored_and_counted_once(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_distributed_probe_worker, args=(str(Path(directory, 'rendezvous')),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
