"""Global accumulated gradient norms: shards, replicas and component coverage."""
import datetime
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from finetune.pose_opd_credit.gradient_norms import COMPONENTS, combined_gradient_norms


def _parameter(model, component, values):
    module = nn.Module()
    parameter = nn.Parameter(torch.zeros_like(values))
    module.register_parameter('weight', parameter)
    model.add_module(component, module)
    parameter.grad = values
    return parameter


def _distributed_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, DTensor, Partial, Replicate, Shard

    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank,
                            world_size=4, timeout=datetime.timedelta(seconds=30))
    try:
        mesh = init_device_mesh('cpu', (4,))
        hybrid = init_device_mesh('cpu', (2, 2))
        model = nn.Module()
        references = {}
        for component, value, layout_mesh, placements in (
            ('layers', torch.arange(1., 12.), mesh, [Shard(0)]),
            ('head', torch.tensor([3., -4.]), mesh, [Replicate()]),
            ('dec_p', torch.arange(-4., 5.), hybrid, [Replicate(), Shard(0)]),
            ('tower_p', torch.arange(1., 49.).reshape(8, 6) / 10, hybrid, [Shard(0), Shard(1)]),
            ('dec_s', torch.tensor([1., -2.]), mesh, [Shard(0)]),
        ):
            # These are the accumulated gradients after two microbatches,
            # including sign cancellation, not averages of microbatch norms.
            gradient = value + (-.25 * value)
            references[component] = gradient
            distributed = distribute_tensor(gradient, layout_mesh, placements)
            _parameter(model, component, distributed)
        references['vgain'] = torch.tensor(2.5)
        references['other'] = torch.tensor([.5, -.25])
        _parameter(model, 'vgain', references['vgain'])
        _parameter(model, 'unclassified', references['other'])
        before = [p.grad.to_local().clone() if hasattr(p.grad, 'to_local') else p.grad.clone()
                  for p in model.parameters()]
        with mock.patch.object(dist, 'all_reduce', wraps=dist.all_reduce) as reduce:
            actual = combined_gradient_norms(model, 'cpu')
            assert reduce.call_count == 1, reduce.call_count
        expected_squared = 0.
        for component in COMPONENTS:
            values = references.get(component, torch.zeros(1))
            squared = float(values.double().square().sum())
            expected_squared += squared
            assert math.isclose(actual[f'grad_norm/combined/{component}'], math.sqrt(squared),
                                abs_tol=1e-12, rel_tol=1e-12), (component, actual, squared)
        assert math.isclose(actual['gradient_norm'], math.sqrt(expected_squared), rel_tol=1e-12)
        for previous, parameter in zip(before, model.parameters(), strict=True):
            current = parameter.grad.to_local() if hasattr(parameter.grad, 'to_local') else parameter.grad
            torch.testing.assert_close(current, previous, rtol=0., atol=0.)

        # Reject an unsynchronized gradient coherently on every rank using the
        # same one reduction, even if only one rank encounters that layout.
        bad = nn.Module()
        if rank == 0:
            gradient = DTensor.from_local(torch.ones(3), mesh, [Partial()],
                                         shape=torch.Size([3]), stride=(1,))
        else:
            gradient = DTensor.from_local(torch.ones(3), mesh, [Replicate()],
                                         shape=torch.Size([3]), stride=(1,))
        _parameter(bad, 'layers', gradient)
        try:
            combined_gradient_norms(bad, 'cpu')
        except ValueError:
            pass
        else:
            raise AssertionError('Unreduced Partial gradient must not be reported as a norm')
    finally:
        dist.destroy_process_group()


class GradientNormTests(unittest.TestCase):
    def test_accumulated_component_norms_cover_unique_parameters_and_do_not_modify_grads(self):
        model = nn.Module()
        expected_squared = 0.
        for index, component in enumerate(COMPONENTS[:-1], start=1):
            first = torch.tensor([float(index), 2. * index], dtype=torch.float64)
            second = torch.tensor([-float(index), 3. * index], dtype=torch.float64)
            _parameter(model, component, first + second)
            expected_squared += (5. * index) ** 2
        _parameter(model, 'unclassified', torch.tensor([3., 4.], dtype=torch.float64))
        expected_squared += 25
        model.alias = model.dec_p  # Shared module/parameters must not be counted twice.
        before = [p.grad.clone() for p in model.parameters()]
        with mock.patch.object(torch.autograd, 'backward', side_effect=AssertionError('extra backward')):
            actual = combined_gradient_norms(model, 'cpu')
        self.assertEqual(actual['grad_norm/combined/layers'], 10.)
        self.assertEqual(actual['grad_norm/combined/other'], 5.)
        self.assertAlmostEqual(actual['gradient_norm'], math.sqrt(expected_squared), places=12)
        for previous, parameter in zip(before, model.parameters(), strict=True):
            torch.testing.assert_close(parameter.grad, previous, rtol=0., atol=0.)

    def test_missing_and_empty_gradients_are_zero_and_nonfinite_gradients_fail(self):
        model = nn.Module()
        parameter = _parameter(model, 'dec_p', torch.empty(0))
        model.register_parameter('unused', nn.Parameter(torch.ones(1)))
        self.assertTrue(all(value == 0. for value in combined_gradient_norms(model, 'cpu').values()))
        for value in (float('nan'), float('inf')):
            parameter = _parameter(nn.Module(), 'layers', torch.tensor([value]))
            model = nn.Module()
            model.register_parameter('bad', parameter)
            with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
                combined_gradient_norms(model, 'cpu')

    def test_four_rank_shards_hybrid_replicas_and_plain_replicas_count_each_element_once(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_distributed_worker, args=(str(Path(directory, 'rendezvous')),), nprocs=4, join=True)


if __name__ == '__main__':
    unittest.main()
