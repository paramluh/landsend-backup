import os
import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from distillation.lora import LoRALinear, inject_lora
from distillation.update_scale import capture, low_rank_delta_squared, measure
from utils import saver
from utils.harness import Zero1FP32MasterAdamW


def effective_weights(net):
    return [layer.weight.detach().double() + (layer.lora_B.detach().double() @ layer.lora_A.detach().double())
            * layer.scaling if isinstance(layer, LoRALinear) else layer.weight.detach().double().clone()
            for layer in net]


def distributed_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{directory}/rendezvous', rank=rank, world_size=4,
                            timeout=timedelta(seconds=90))
    saver.init_gloo()
    for lora in (False, True):
        torch.manual_seed(12)
        net = nn.Sequential(*[nn.Linear(8, 8) for _ in range(4)])
        if lora:
            inject_lora(net, rank=2, alpha=4, targets=[r'[0-3]'])
            for layer in net:
                nn.init.normal_(layer.lora_B, std=.1)
        optimizer = Zero1FP32MasterAdamW([{'params': [p for p in net.parameters() if p.requires_grad]}],
                                        lambda groups: torch.optim.AdamW(groups, lr=.005))
        before = capture(net, optimizer, [r'[0-3]'], lora)
        previous = effective_weights(net)
        base_sq = sum(layer.weight.detach().double().square().sum() for layer in net)
        for parameter in net.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.full_like(parameter, .1)
        optimizer.step()
        after = capture(net, optimizer, [r'[0-3]'], lora)
        report = measure(before, after)
        if rank == 0:
            expected_sq = sum((actual - old).square().sum() for actual, old in zip(effective_weights(net), previous))
            torch.testing.assert_close(torch.tensor(report['update_l2'] ** 2, dtype=torch.float64), expected_sq,
                                       rtol=1e-10, atol=1e-14)
            torch.testing.assert_close(torch.tensor(report['base_weight_l2'] ** 2, dtype=torch.float64), base_sq)
            assert report['matrix_count'] == 4 and report['weight_numel'] == 256
        else:
            assert report is None
    dist.destroy_process_group()


class UpdateScaleTests(unittest.TestCase):
    def test_low_rank_delta_matches_materialization(self):
        torch.manual_seed(14)
        a0, b0 = torch.randn(4, 13), torch.randn(9, 4)
        for magnitude in (1.0, 1e-5, 0.0):
            a1, b1 = a0 + magnitude * torch.randn_like(a0), b0 + magnitude * torch.randn_like(b0)
            expected = ((b1.double() @ a1.double() - b0.double() @ a0.double()) * 3).square().sum()
            actual = low_rank_delta_squared(a0, b0, a1, b1, 3)
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-14)

    def test_distributed_dense_and_lora_master_updates(self):
        with tempfile.TemporaryDirectory(prefix='update-scale-', dir=os.environ['BIMODAL_TEST_OUTPUT']) as directory:
            mp.spawn(distributed_worker, args=(directory,), nprocs=4, join=True)


if __name__ == '__main__':
    unittest.main()
