"""Opt-in replay stage timings, without adding per-chunk synchronization."""
from __future__ import annotations

from contextlib import contextmanager
import time


class ReplayStages:
    def __init__(self, device):
        import torch
        self.device = torch.device(device)
        self.records = []
        self.result = None

    @contextmanager
    def span(self, name):
        import torch
        if self.result is not None:
            raise RuntimeError('Replay stage profile is already collected')
        events = None
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                events[0].record()
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            if events is not None:
                with torch.cuda.device(self.device):
                    events[1].record()
            self.records.append((name, elapsed, events))

    def collect(self):
        import torch
        if self.result is not None:
            return self.result
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        stages = {}
        for name, elapsed, events in self.records:
            record = stages.setdefault(name, {'calls': 0, 'host_submission_seconds': 0.,
                                             'cuda_current_stream_ms': 0. if events is not None else None})
            record['calls'] += 1
            record['host_submission_seconds'] += elapsed
            if events is not None:
                record['cuda_current_stream_ms'] += events[0].elapsed_time(events[1])
        self.result = {'stages': stages,
            'scope': 'host submission spans and current-stream CUDA event intervals; overlapping FSDP communication is not isolated',
            'synchronization': 'one profiler collection sync after replay; none added per chunk'}
        self.records.clear()
        return self.result
