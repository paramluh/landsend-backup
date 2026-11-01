"""Bounded CUDA-event stage profiling; never synchronizes individual stages.

GPU times are elapsed intervals on the current stream, including stream waits
and CPU launch gaps. Parent/child intervals overlap and must not be summed.
CPU seconds measure host dispatch, not a second additive component of GPU time.
Only end-of-step synchronization resolves the events.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, nullcontext
import time


class NullProfile:
    def span(self, name):
        return nullcontext()


class PhaseProfiler:
    def __init__(self, device):
        import torch
        self.torch = torch
        self.device = device
        self.active = False
        self.events = []

    def begin(self, step):
        self.step = step
        self.events = []
        self.started = time.perf_counter()
        self.first = self.torch.cuda.Event(enable_timing=True)
        self.first.record()
        self.active = True

    @contextmanager
    def span(self, name):
        if not self.active:
            yield
            return
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        cpu = time.perf_counter()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append((name, start, end, time.perf_counter() - cpu))

    def finish(self, **metadata):
        last = self.torch.cuda.Event(enable_timing=True)
        last.record()
        self.torch.cuda.synchronize(self.device)
        wall = time.perf_counter() - self.started
        self.active = False
        stages = defaultdict(lambda: {'gpu_seconds': 0., 'cpu_dispatch_seconds': 0., 'calls': 0})
        for name, start, end, cpu in self.events:
            stage = stages[name]
            stage['gpu_seconds'] += start.elapsed_time(end) / 1000
            stage['cpu_dispatch_seconds'] += cpu
            stage['calls'] += 1
        result = {'step': self.step, 'wall_seconds': wall,
                  'gpu_timeline_seconds': self.first.elapsed_time(last) / 1000,
                  'stages': dict(stages), **metadata}
        self.events = []
        return result
