"""Temporarily move Adam moments to host memory while inference owns the GPUs.

Park only between optimizer updates, after clearing gradients. Restore before
learner replay/backward or checkpointing. This changes state placement, never its
values or the optimizer schedule. A parked optimizer cannot step, save, or load
state. The context-manager form restores state even when rollout raises.

DTensor metadata survives separately from its local CPU shard. In particular,
keeping the original DTensor as a restoration template would retain its CUDA
storage and defeat parking; this module never keeps such a template.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
from torch.distributed.tensor import DTensor


@dataclass(frozen=True)
class _Placement:
    device: torch.device
    mesh: Any = None
    placements: tuple | None = None
    shape: torch.Size | None = None
    stride: tuple[int, ...] | None = None


class OptimizerStateParking:
    """Lossless, synchronous-at-boundaries host parking for Adam/AdamW state.

    ``pin_memory=True`` enables asynchronous copies into pinned host buffers;
    park/restore synchronize affected CUDA devices before returning. It needs
    sufficient host pinned memory for all local Adam moments. The default uses
    ordinary host memory and synchronous copies. Parameters and gradients are
    never moved or retained separately.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, *, pin_memory: bool = False):
        if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            raise TypeError('Optimizer state parking supports Adam and AdamW')
        self.optimizer = optimizer
        self.pin_memory = bool(pin_memory)
        self.parked = False
        self._entries: list[tuple[dict, Any, _Placement]] = []
        self._hooks = []
        self.stats = {'parked_tensors': 0, 'cuda_bytes_released': 0,
                      'host_bytes': 0, 'park_seconds': 0., 'restore_seconds': 0.}

    def _forbid_state_use(self, *_args, **_kwargs):
        raise RuntimeError('Adam state is parked on CPU; restore it before step/save/load')

    @staticmethod
    def _sync(devices):
        for device in devices:
            torch.cuda.synchronize(device)

    @torch.no_grad()
    def park(self):
        if self.parked or getattr(self.optimizer, '_pose_opd_state_parking', None) is not None:
            raise RuntimeError('Optimizer state is already parked')
        started = time.monotonic()
        self.parked = True
        self.optimizer._pose_opd_state_parking = self
        self._hooks = [self.optimizer.register_step_pre_hook(self._forbid_state_use),
                       self.optimizer.register_state_dict_pre_hook(self._forbid_state_use),
                       self.optimizer.register_load_state_dict_pre_hook(self._forbid_state_use)]
        self.stats = {'parked_tensors': 0, 'cuda_bytes_released': 0,
                      'host_bytes': 0, 'park_seconds': 0., 'restore_seconds': 0.}
        devices = set()
        try:
            # Adam uses a flat state dictionary. Reject nested containers rather
            # than silently leave unknown tensor state resident on the device.
            for state in self.optimizer.state.values():
                for key, value in tuple(state.items()):
                    if isinstance(value, (dict, list, tuple)):
                        raise TypeError(f'Unsupported nested Adam state: {key}')
                    if not isinstance(value, torch.Tensor):
                        continue
                    is_dtensor = isinstance(value, DTensor)
                    local = value.to_local() if is_dtensor else value
                    if local.device.type != 'cuda' and not is_dtensor:
                        continue  # Preserve ordinary CPU scalar steps in place.
                    if local.layout != torch.strided or local.requires_grad:
                        raise ValueError(f'Adam state must be a detached strided tensor: {key}')
                    placement = _Placement(local.device, value.device_mesh if is_dtensor else None,
                                           tuple(value.placements) if is_dtensor else None,
                                           value.shape if is_dtensor else None,
                                           value.stride() if is_dtensor else None)
                    pinned = self.pin_memory and local.device.type == 'cuda'
                    host = torch.empty_like(local, device='cpu', pin_memory=pinned)
                    host.copy_(local, non_blocking=pinned)
                    state[key] = host
                    self._entries.append((state, key, placement))
                    size = local.numel() * local.element_size()
                    self.stats['parked_tensors'] += 1
                    self.stats['host_bytes'] += size
                    if local.device.type == 'cuda':
                        devices.add(local.device)
                        self.stats['cuda_bytes_released'] += size
            self._sync(devices)
        except BaseException:
            # Complete outstanding D2H writes before reconstructing any state.
            self._sync(devices)
            self.restore()
            raise
        self.stats['park_seconds'] = time.monotonic() - started
        return dict(self.stats)

    @torch.no_grad()
    def restore(self):
        if not self.parked:
            return dict(self.stats)
        started = time.monotonic()
        devices = set()
        try:
            # Pop only successfully restored entries. If an allocation fails,
            # guards remain installed and callers can free memory and retry.
            while self._entries:
                state, key, placement = self._entries[-1]
                host = state[key]
                if not isinstance(host, torch.Tensor) or isinstance(host, DTensor) or host.device.type != 'cpu':
                    raise RuntimeError(f'Parked Adam state was modified: {key}')
                local = host.to(device=placement.device,
                                non_blocking=host.is_pinned() and placement.device.type == 'cuda')
                if placement.mesh is not None:
                    restored = DTensor.from_local(local, device_mesh=placement.mesh,
                                                  placements=placement.placements, run_check=False,
                                                  shape=placement.shape, stride=placement.stride)
                else:
                    restored = local
                state[key] = restored
                self._entries.pop()
                if placement.device.type == 'cuda':
                    devices.add(placement.device)
        finally:
            self._sync(devices)
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        del self.optimizer._pose_opd_state_parking
        self.parked = False
        self.stats['restore_seconds'] = time.monotonic() - started
        return dict(self.stats)

    def __enter__(self):
        self.park()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.restore()
        return False
