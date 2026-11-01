import math
import re

import torch
import torch.distributed as dist
from torch import nn

from distillation.lora import LoRALinear
from utils import saver


def low_rank_delta_squared(a0, b0, a1, b1, scale):
    a0, b0, a1, b1 = (tensor.double() for tensor in (a0, b0, a1, b1))
    # B1 A1 - B0 A0 = (B1 - B0) A1 + B0 (A1 - A0), without forming a dense weight.
    left = torch.cat((b1 - b0, b0), dim=1)
    right = torch.cat((a1, a1 - a0), dim=0)
    return ((left.T @ left) * (right @ right.T)).sum().clamp_min(0) * scale ** 2


@torch.no_grad()
def capture(net, optimizer, targets, lora):
    if saver.gloo is None:
        raise ValueError('Update calibration requires saver.init_gloo()')
    rank, world = dist.get_rank(), dist.get_world_size()
    patterns = [re.compile(pattern) for pattern in targets]
    layers = {name: layer for name, layer in net.named_modules()
              if isinstance(layer, nn.Linear) and any(pattern.fullmatch(name) for pattern in patterns)}
    if not layers or any(not any(pattern.fullmatch(name) for name in layers) for pattern in patterns):
        raise ValueError('Every update-calibration target must match a Linear')
    if any(isinstance(layer, LoRALinear) != bool(lora) for layer in layers.values()):
        raise ValueError('Update-calibration mode does not match the targeted layers')
    selected = {}
    for name, layer in layers.items():
        if lora:
            selected[layer.lora_A], selected[layer.lora_B] = f'{name}.lora_A', f'{name}.lora_B'
        else:
            selected[layer.weight] = f'{name}.weight'
    state = {selected[p]: master.detach().to('cpu', copy=True) for p, master in optimizer.pairs if p in selected}
    count = torch.tensor(len(state), dtype=torch.int64)
    dist.all_reduce(count, group=saver.gloo)
    if count.item() != len(selected):
        raise ValueError('Optimizer must own all targeted weights or LoRA factors exactly once')
    device = next(iter(layers.values())).weight.device
    norm_sq = torch.zeros((), device=device, dtype=torch.float64)
    for index, layer in enumerate(layers.values()):
        if index % world == rank:
            norm_sq += torch.linalg.vector_norm(layer.weight.detach(), dtype=torch.float64).square()
    if lora:
        gathered = {}
        for source in range(world):
            shard = saver._to_rank0(state if rank == source else None, source)
            if rank == 0:
                gathered.update(shard)
        state = gathered
    return {'lora': bool(lora), 'state': state, 'base_norm_sq': norm_sq.item(), 'device': device,
            'scales': {name: layer.scaling for name, layer in layers.items()} if lora else {},
            'matrix_count': len(layers), 'weight_numel': sum(layer.weight.numel() for layer in layers.values())}


@torch.no_grad()
def measure(before, after):
    for key in ('lora', 'scales', 'matrix_count', 'weight_numel'):
        if before[key] != after[key]:
            raise ValueError(f'Update-calibration snapshots differ in {key}')
    if before['state'].keys() != after['state'].keys():
        raise ValueError('Update-calibration snapshot parameters differ')
    delta_sq = torch.zeros((), dtype=torch.float64, device=before['device'])
    if before['lora']:
        if dist.get_rank() == 0:
            for name, scale in before['scales'].items():
                factors = [snapshot['state'][f'{name}.{factor}'].to(before['device'])
                           for snapshot in (before, after) for factor in ('lora_A', 'lora_B')]
                delta_sq += low_rank_delta_squared(*factors, scale)
    else:
        for name, value in before['state'].items():
            difference = after['state'][name].double() - value.double()
            delta_sq += torch.linalg.vector_norm(difference, dtype=torch.float64).square().to(before['device'])
    totals = torch.tensor([delta_sq.item(), before['base_norm_sq']], dtype=torch.float64)
    dist.all_reduce(totals, group=saver.gloo)
    if dist.get_rank() != 0:
        return None
    delta, base = (math.sqrt(value) for value in totals.tolist())
    return {'update_l2': delta, 'update_rms': delta / math.sqrt(before['weight_numel']),
            'base_weight_l2': base, 'relative_update_l2': delta / base if base else None,
            'matrix_count': before['matrix_count'], 'weight_numel': before['weight_numel'],
            'parameterization': 'lora_effective_fp32' if before['lora'] else 'dense_fp32',
            'reference': 'stored_base_weights_before_update'}
