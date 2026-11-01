"""Occasional loss-wise parameter norms with an unchanged training update.

Five cached backward passes measure three direct terms and text/audio credit.
Pose credit is the residual of the saved total minus those five terms. Residual
and aggregate subtraction use FP64 host accumulators, but independently rounded
backward passes mean the derived sixth term can differ by floating-point error.
The original gradient is restored bit for bit, including None and DTensor layout,
even on failure. This module never calls an optimizer or samples a trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from .gradient_norms import COMPONENTS, _replica_factor


BACKWARD_TERMS = ('direct/text', 'direct/audio', 'direct/pose', 'credit/text', 'credit/audio')
RESIDUAL_TERM = 'credit/pose'
SERIES = (*BACKWARD_TERMS, RESIDUAL_TERM, 'direct', 'credit')


def loss_gradient_probe_due(config, completed_update):
    if type(completed_update) is not int or completed_update < 1:
        raise ValueError('Loss-gradient cadence uses a positive completed update')
    interval = config.loss_gradient_interval_steps
    return interval is not None and completed_update % interval == 0


@dataclass
class _Gradient:
    parameter: torch.Tensor
    component: int
    original: torch.Tensor | None
    device: torch.device
    layout: dict | None
    shape: tuple
    replicas: int
    accumulated: torch.Tensor | None = None


def _snapshot(model, world):
    result = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        template = parameter if gradient is None else gradient
        local = template.to_local() if hasattr(template, 'to_local') else template
        if local.layout != torch.strided:
            raise ValueError('Loss-gradient probing requires dense gradients')
        component = name.replace('_checkpoint_wrapped_module.', '').split('.', 1)[0]
        layout = None if not hasattr(template, 'to_local') else {
            'device_mesh': template.device_mesh, 'placements': template.placements,
            'shape': template.shape, 'stride': template.stride(), 'run_check': False}
        result.append(_Gradient(parameter,
            COMPONENTS.index(component) if component in COMPONENTS else len(COMPONENTS) - 1,
            None if gradient is None else local.detach().to('cpu', copy=True), local.device,
            layout, tuple(local.shape), _replica_factor(template, world)))
    return result


def _restore(records):
    # Release the last probe's device gradients before copying originals back.
    for record in records:
        record.parameter.grad = None
    for record in records:
        if record.original is None:
            continue
        value = record.original.to(record.device, copy=True)
        if record.layout is not None:
            from torch.distributed.tensor import DTensor
            value = DTensor.from_local(value, **record.layout)
        record.parameter.grad = value


def _add_square(target, record, value):
    if value is not None:
        target[record.component] += torch.linalg.vector_norm(value, dtype=torch.float64).square() / record.replicas


def _collect_term(records, target, world):
    for record in records:
        gradient = record.parameter.grad
        if gradient is None:
            continue
        local = gradient.to_local() if hasattr(gradient, 'to_local') else gradient
        if tuple(local.shape) != record.shape or _replica_factor(gradient, world) != record.replicas:
            raise ValueError('Probe gradient layout differs from the synchronized training gradient')
        value = local.detach().to('cpu', copy=True)
        _add_square(target, record, value)
        if record.accumulated is None:
            record.accumulated = value.double()
        else:
            record.accumulated.add_(value)


def _residual(record):
    if record.accumulated is None:
        return record.original
    if record.original is None:
        return -record.accumulated
    return record.original.double() - record.accumulated


def measure_loss_gradient_norms(model, backward_term, device):
    """Call backward_term(term) five times on cached rows, then restore grads.

    The callback must accumulate that one term across all original microbatches
    without modifying model weights, buffers, optimizer state or input data.
    It must use the original globally normalized objective coefficients.
    Only CPU copies of gradient shards are retained between backward passes.
    """
    started = time.monotonic()
    world = dist.get_world_size() if dist.is_initialized() else 1
    records = _snapshot(model, world)
    cpu_rng = torch.get_rng_state()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = sorted({record.device.index for record in records if record.device.type == 'cuda'})
    cuda_rng = {index: torch.cuda.get_rng_state(index) for index in devices}
    squares = torch.zeros(len(SERIES), len(COMPONENTS), dtype=torch.float64)
    try:
        for index, term in enumerate(BACKWARD_TERMS):
            model.zero_grad(set_to_none=True)
            backward_term(term)
            _collect_term(records, squares[index], world)
            if index == 2:
                # Once the direct sum is available, the aggregate credit is
                # total minus direct. Then reuse this same FP64 host buffer.
                for record in records:
                    _add_square(squares[SERIES.index('direct')], record, record.accumulated)
                    _add_square(squares[SERIES.index('credit')], record, _residual(record))
        for record in records:
            _add_square(squares[SERIES.index(RESIDUAL_TERM)], record, _residual(record))
        # All eight series share one reduction of squared local shard norms.
        reduced = squares.to(device)
        if dist.is_initialized():
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        values = reduced.cpu().tolist()
        if any(not math.isfinite(value) or value < 0 for row in values for value in row):
            raise RuntimeError('Nonfinite loss-wise parameter gradients')
        metrics = {}
        for series, row in zip(SERIES, values, strict=True):
            for component, value in zip(COMPONENTS, row, strict=True):
                metrics[f'grad_norm/{series}/{component}'] = math.sqrt(value)
            metrics[f'grad_norm/{series}/all'] = math.sqrt(math.fsum(row))
    finally:
        try:
            _restore(records)
        finally:
            torch.set_rng_state(cpu_rng)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            for index, state in cuda_rng.items():
                torch.cuda.set_rng_state(state, index)
    metrics.update(loss_gradient_diagnostics_seconds=time.monotonic() - started,
                   loss_gradient_diagnostics_extra_backwards=len(BACKWARD_TERMS))
    return metrics
