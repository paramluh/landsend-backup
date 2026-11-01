"""CPU distributed checks for parallel, durable optimizer-shard checkpoint saves."""
import copy
from datetime import timedelta
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distillation import checkpoints
from utils import saver
from utils.harness import EMA, Zero1FP32MasterAdamW


def _init(rank, world, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(),
                            rank=rank, world_size=world, timeout=timedelta(seconds=30))
    saver.gloo = dist.group.WORLD
    torch.manual_seed(100 + rank)
    random.seed(200 + rank)
    np.random.seed(300 + rank)


def _training_state(rank):
    optimizers, schedulers = {}, {}
    for role in ('student', 'critic'):
        parameters = [torch.nn.Parameter(torch.arange(size, dtype=torch.float32) + 1)
                      for size in (3, 4, 5)]
        for index, parameter in enumerate(parameters):
            parameter.original_name = f'{role}.parameter_{index}'
        optimizer = Zero1FP32MasterAdamW([{'params': parameters}],
            lambda groups: torch.optim.AdamW(groups, lr=0.01 * (rank + 1)))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer.inner, step_size=1, gamma=0.75)
        sum(parameter.square().sum() for parameter in parameters).backward()
        optimizer.step()
        scheduler.step()
        optimizers[role], schedulers[role] = optimizer, scheduler
    ema = EMA(0.9, optimizers['student'].pairs)
    with torch.no_grad():
        for _, _, shadow in ema.triples:
            shadow.add_(rank + 0.5)
    return optimizers, schedulers, ema


def _assert_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected), (actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_equal(left, right)
    else:
        assert actual == expected, (actual, expected)


def _success_worker(rank, world, root, rendezvous):
    _init(rank, world, rendezvous)
    try:
        directory = Path(root)
        optimizers, schedulers, ema = _training_state(rank)
        counters = {'rounds': 17, 'student_updates': 8, 'critic_updates': 16, 'rank': rank}
        expected = copy.deepcopy({'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
                                  'schedulers': {k: v.state_dict() for k, v in schedulers.items()},
                                  'ema': ema.state_dict()})
        real_save, writes = torch.save, []

        def owner_save(state, destination, *args, **kwargs):
            # A serial rank-0 gather/write cannot enter this rendezvous on every rank.
            name = Path(getattr(destination, 'name', destination)).name
            assert f'rank-{rank:05d}.pt' in name, name
            assert state['counters']['rank'] == rank
            writes.append(name)
            dist.monitored_barrier(timeout=timedelta(seconds=15))
            return real_save(state, destination, *args, **kwargs)

        with patch.object(checkpoints.torch, 'save', owner_save):
            checkpoints.save_rank_states(directory, optimizers, schedulers, ema, counters, False)
        assert len(writes) == 1
        assert len(list(directory.glob('rank-*.pt'))) == world
        assert not list(directory.glob('rank-*.pt.tmp'))
        if rank == 0:
            checkpoints.complete_checkpoint(directory, {'world_size': world, 'optimizer_state_retained': True})
        expected_rng = (torch.rand(5), random.random(), np.random.random(5))

        for optimizer in optimizers.values():
            optimizer.inner.state.clear()
            optimizer.inner.param_groups[0]['lr'] = 99.
            for parameter, master in optimizer.pairs:
                parameter.data.fill_(99.)
                master.fill_(99.)
        for scheduler in schedulers.values():
            scheduler.last_epoch = 99
        for _, _, shadow in ema.triples:
            shadow.zero_()
        torch.manual_seed(999)
        random.seed(999)
        np.random.seed(999)

        # Existing rank-0-mediated readers must accept the new owner-written format.
        actual_counters = checkpoints.load_rank_state(directory, optimizers, schedulers, ema,
                                                       node_sharded=True, critic_only=False)
        assert actual_counters == counters
        _assert_equal({k: v.state_dict() for k, v in optimizers.items()}, expected['optimizers'])
        _assert_equal({k: v.state_dict() for k, v in schedulers.items()}, expected['schedulers'])
        _assert_equal(ema.state_dict(), expected['ema'])
        assert torch.equal(torch.rand(5), expected_rng[0])
        assert random.random() == expected_rng[1]
        assert np.array_equal(np.random.random(5), expected_rng[2])
    finally:
        dist.destroy_process_group()


def _failure_worker(rank, world, root, rendezvous):
    _init(rank, world, rendezvous)
    try:
        optimizers, schedulers, ema = _training_state(rank)
        for failure in ('write', 'fsync', 'nonshared'):
            directory = Path(root) / failure
            directory.mkdir(exist_ok=True)
            mine = directory
            if failure == 'nonshared' and rank == 1:
                mine = directory / 'private-rank-1'
                mine.mkdir()
            real_save, real_fsync = torch.save, checkpoints.os.fsync

            def fail_write(*args, **kwargs):
                if failure == 'write' and rank == 1:
                    raise OSError('injected rank-1 write failure')
                return real_save(*args, **kwargs)

            def fail_fsync(*args, **kwargs):
                if failure == 'fsync' and rank == 1:
                    raise OSError('injected rank-1 fsync failure')
                return real_fsync(*args, **kwargs)

            error = None
            with patch.object(checkpoints.torch, 'save', fail_write), \
                    patch.object(checkpoints.os, 'fsync', fail_fsync):
                try:
                    checkpoints.save_rank_states(mine, optimizers, schedulers, ema, {'rounds': 17}, False)
                except RuntimeError as exc:
                    error = str(exc)
                else:
                    if rank == 0:
                        checkpoints.complete_checkpoint(directory, {'world_size': world})
            errors = [None] * world
            dist.all_gather_object(errors, error)
            assert all(errors), (failure, errors)
            assert len(set(errors)) == 1, errors
            if failure != 'nonshared':
                assert f'injected rank-1 {failure} failure' in error, error
                assert not (directory / 'rank-00001.pt').exists()
            else:
                assert 'rank-00001.pt' in error, error
            assert not (directory / 'checkpoint.complete.json').exists()
    finally:
        dist.destroy_process_group()


class ParallelCheckpointTests(unittest.TestCase):
    def test_owner_writes_and_exact_legacy_resume(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / 'checkpoint'
            directory.mkdir()
            mp.spawn(_success_worker, args=(3, str(directory), str(Path(root) / 'gloo')), nprocs=3)
            manifest = json.loads((directory / 'checkpoint.complete.json').read_text())
            self.assertTrue({f'rank-{rank:05d}.pt' for rank in range(3)} <=
                            {item['path'] for item in manifest['files']})
            for item in manifest['files']:
                self.assertEqual((directory / item['path']).stat().st_size, item['size'])

    def test_rank_local_failures_prevent_completion_everywhere(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(_failure_worker, args=(3, root, str(Path(root) / 'gloo')), nprocs=3)


if __name__ == '__main__':
    unittest.main()
