"""One batch ahead, in an isolated spawned CPU process.

ProductionRows temporarily replaces native modules and environment variables,
so it must never execute concurrently with the model in a Python thread.
Only numpy-backed row inputs cross the process boundary; GPU encoding remains
on the training rank. No sampling or plan construction happens here.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os

_loader = None


def _production_loader(config):
    from .config import TrainingConfig
    from .data import ProductionRows
    return ProductionRows(TrainingConfig.from_dict(config))


def _initialize_worker(config, loader_factory):
    global _loader
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)
    _loader = loader_factory(config)


def _load_batch(items, fallback_item, rank, update, partner_mode=False):
    return [_loader.load(fallback_item if item is None else item, rank,
                         **({'update': update, 'slot': slot} if partner_mode else {}))
            for slot, item in enumerate(items)]


class RowPrefetch:
    """A persistent worker with exactly one pending batch and explicit cursor."""

    def __init__(self, config, *, rank, fallback_item, loader_factory=_production_loader):
        self.capacity = config.effective_batch_per_rank
        self.rank, self.fallback_item = int(rank), tuple(fallback_item)
        self.partner_mode = config.rollout_mode == 'teacher_partner'
        self._pending = None
        self._closed = False
        self._pool = ProcessPoolExecutor(max_workers=1,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_initialize_worker, initargs=(config.to_dict(), loader_factory))

    def submit(self, update, items):
        if self._closed:
            raise RuntimeError('Row prefetch is closed')
        if self._pending is not None:
            raise RuntimeError('Row prefetch already has one pending batch')
        if len(items) != self.capacity:
            raise ValueError('Prefetch batch must match effective_batch_per_rank')
        frozen = tuple(None if item is None else tuple(item) for item in items)
        future = self._pool.submit(_load_batch, frozen, self.fallback_item, self.rank, update, self.partner_mode)
        self._pending = (update, future)

    def take(self, update, *, timeout=None):
        if self._pending is None or self._pending[0] != update:
            raise RuntimeError('Row prefetch cursor differs from the training update')
        _, future = self._pending
        try:
            return future.result(timeout=timeout)
        finally:
            if future.done():
                self._pending = None

    def close(self):
        if not self._closed:
            self._closed = True
            if self._pending is not None:
                self._pending[1].cancel()
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pending = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
