"""Norms of the accumulated, synchronized parameter gradient before clipping.

These are combined direct-KL plus score-credit gradients. Decomposing their
parameter norms by loss would require additional backward passes; these metrics
do not claim that decomposition. Local squared norms count each global element
once, including hybrid-sharded or replicated DTensors, before one all-reduce.
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist


COMPONENTS = ('inp', 'layers', 'head', 'vadapter', 'vgain', 'tower_s', 'dec_s',
              'tower_p', 'dec_p', 'decp_bb', 'pose_head', 'other')


def _replica_factor(gradient, world):
    if not hasattr(gradient, 'to_local'):
        # Plain gradients left outside FSDP are synchronized replicas.
        return world
    mesh = gradient.device_mesh
    if world % mesh.size():
        raise ValueError('Gradient device mesh must divide the reduction world')
    replicas = world // mesh.size()
    for axis, placement in enumerate(gradient.placements):
        if placement.is_replicate():
            replicas *= mesh.size(axis)
        elif not placement.is_shard():
            raise ValueError('Gradient norm requires synchronized Shard/Replicate gradients, not Partial')
    return replicas


@torch.no_grad()
def combined_gradient_norms(model, device):
    """Return component/global L2 norms without modifying grads or running VJPs.

    Call after all microbatches and gradient synchronization, before clipping.
    Ordinary tensors must already contain identical gradients on every rank.
    DTensor shards may be uneven or empty; replicated mesh dimensions are
    compensated before the global sum. Every unique parameter is counted once.
    """
    device = torch.device(device)
    world = dist.get_world_size() if dist.is_initialized() else 1
    squared = torch.zeros(len(COMPONENTS) + 1, dtype=torch.float64, device=device)
    gradients, components, factors = [], [], []
    invalid = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        local = gradient.to_local() if hasattr(gradient, 'to_local') else gradient
        try:
            factor = _replica_factor(gradient, world)
            if local.layout != torch.strided or local.device != device:
                raise ValueError('Gradient norm requires dense local gradients on the rank device')
        except ValueError as exc:
            invalid.append(f'{name}: {exc}')
            continue
        canonical = name.replace('_checkpoint_wrapped_module.', '')
        component = canonical.split('.', 1)[0]
        components.append(COMPONENTS.index(component) if component in COMPONENTS else len(COMPONENTS) - 1)
        factors.append(1. / factor)
        gradients.append(local.detach())
    if gradients:
        # The dtype argument accumulates in FP64 without materializing FP64
        # copies of every model gradient. Only scalar norms are stacked.
        norms = torch.stack(torch._foreach_norm(gradients, 2., dtype=torch.float64))
        contributions = norms.square() * torch.tensor(factors, device=device, dtype=torch.float64)
        squared.scatter_add_(0, torch.tensor(components, device=device), contributions)
    squared[-1] = bool(invalid)
    if dist.is_initialized():
        dist.all_reduce(squared, op=dist.ReduceOp.SUM)
    values = squared.tolist()
    if values[-1]:
        detail = '; '.join(invalid) if invalid else 'unsupported gradient layout on another rank'
        raise ValueError(detail)
    if any(not math.isfinite(value) or value < 0 for value in values[:-1]):
        raise RuntimeError('Nonfinite combined parameter gradients before clipping')
    result = {f'grad_norm/combined/{name}': math.sqrt(value)
              for name, value in zip(COMPONENTS, values[:-1], strict=True)}
    result['gradient_norm'] = math.sqrt(math.fsum(values[:-1]))
    return result
