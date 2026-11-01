import ast
import copy
import inspect
import json
import os
import random
import tempfile
import tomllib
import unittest
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from distillation.checkpoints import (
    compatibility, complete_checkpoint, distribution_fingerprint, load_model_weights, load_rank_state, prune_optimizer_state,
    read_checkpoint_metadata,
    save_ema, save_rank_states, validate_resume,
)
from distillation.lora import inject_lora
from learn_dmd_bimodal import SeededWindows
from utils import saver
from utils.harness import EMA, Zero1FP32MasterAdamW


class SmallLoRAModel(torch.nn.Sequential):
    def __init__(self):
        super().__init__(torch.nn.Linear(5, 8), torch.nn.GELU(), torch.nn.Linear(8, 3))

    def original_named_parameters(self):
        return self.named_parameters()


def lora_checkpoint_roundtrip(rank, directory):
    from safetensors.torch import save_file

    def optimizer_for(net):
        parameters = [p for p in net.parameters() if p.requires_grad]
        optimizer = Zero1FP32MasterAdamW([{'params': parameters}], lambda groups: torch.optim.AdamW(groups, lr=.01))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer.inner, lambda step: 1 / (step + 1))
        return optimizer, scheduler

    def update(net, optimizer, scheduler):
        for p in net.parameters():
            p.grad = None
        x = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 20
        net(x).square().mean().backward()
        assert all(p.grad is None for p in net.parameters() if not p.requires_grad)
        optimizer.step()
        scheduler.step()

    nets, optimizers, schedulers = {}, {}, {}
    for role in ('student', 'critic'):
        torch.manual_seed(73)
        net = SmallLoRAModel()
        if role == 'critic':
            with torch.no_grad():
                for p in net.parameters():
                    p.add_(.25)
        torch.manual_seed(100 + rank)
        rng = torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            inject_lora(net, rank=2, alpha=4, targets=[r'0|2'])
        assert torch.equal(torch.get_rng_state(), rng)
        nets[role] = net
        optimizers[role], schedulers[role] = optimizer_for(net)
        for _ in range(2):
            update(net, optimizers[role], schedulers[role])

    destination = Path(directory) / f'node-{rank}' / 'lora'
    if rank == 0:
        destination.mkdir(parents=True)
        for role, net in nets.items():
            save_file({name: p.detach().contiguous() for name, p in net.original_named_parameters()},
                      str(destination / f'{role}.safetensors'))
    dist.barrier()
    counters = dict(warmup_step=150, critic_updates=602, student_updates=2, outer_step=2, data_steps=2)
    save_rank_states(destination, optimizers, schedulers, None, counters, True)
    for role, net in nets.items():
        update(net, optimizers[role], schedulers[role])

    restored_nets, restored_optimizers, restored_schedulers = {}, {}, {}
    for role in nets:
        torch.manual_seed(99 + rank)
        restored = SmallLoRAModel()
        inject_lora(restored, rank=2, alpha=4, targets=[r'0|2'])
        load_model_weights(destination, role, restored, True)
        restored_nets[role] = restored
        restored_optimizers[role], restored_schedulers[role] = optimizer_for(restored)
    assert load_rank_state(destination, restored_optimizers, restored_schedulers, None, True, False) == counters
    for role, restored in restored_nets.items():
        update(restored, restored_optimizers[role], restored_schedulers[role])
        for (name, parameter), (_, expected) in zip(restored.named_parameters(), nets[role].named_parameters()):
            assert torch.equal(parameter, expected), f'{role}.{name}'
        expected_states = optimizers[role].inner.state_dict()['state']
        for index, state in restored_optimizers[role].inner.state_dict()['state'].items():
            assert all(torch.equal(value, expected_states[index][key]) for key, value in state.items())
        assert restored_schedulers[role].state_dict() == schedulers[role].state_dict()


def distributed_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{directory}/rendezvous', rank=rank, world_size=4,
                            timeout=timedelta(seconds=90))
    saver.init_gloo()
    source = SimpleNamespace(lengths=np.array([44, 61] if rank < 2 else [70, 90]), clips=[0, 1])
    data = SeededWindows(source, rank % 2, 2, 2, saver.gloo)
    assert len(data) == 6, len(data)
    selected = []
    for micro in range(len(data)):
        start = (micro * 2 + rank % 2) * 2
        ids = data._order(0)[start:start + 2]
        selected.extend(int(i) for i in ids)
        for index in ids:
            clip, ofi = data._window(0, int(index))
            assert len(ofi) == 8 and ofi[0] == 0 and ofi[-1] <= source.lengths[clip]
    all_selected = [None] * 4
    dist.all_gather_object(all_selected, selected)
    assert set(all_selected[rank // 2 * 2]).isdisjoint(all_selected[rank // 2 * 2 + 1])
    assert len(set(all_selected[rank // 2 * 2] + all_selected[rank // 2 * 2 + 1])) == 24
    assert len(data._order(1)) == 24
    nonsharded = SeededWindows(source, rank, 4, 2)
    assert len(nonsharded) == (3 if rank < 2 else 4)

    optimizers, schedulers, params = {}, {}, {}
    for role, scale in [('student', 1.0), ('critic', 2.0)]:
        params[role] = [torch.nn.Parameter(torch.full((i + 1,), scale * (i + 1))) for i in range(8)]
        for i, parameter in enumerate(params[role]):
            parameter.original_name = f'weight{i}'
        optimizers[role] = Zero1FP32MasterAdamW([{'params': params[role]}], lambda groups: torch.optim.AdamW(groups, lr=.01))
        schedulers[role] = torch.optim.lr_scheduler.LambdaLR(optimizers[role].inner, lambda step: 1 / (step + 1))
        for _ in range(2):
            for parameter in params[role]:
                parameter.grad = torch.full_like(parameter, .1)
            optimizers[role].step()
            schedulers[role].step()
    expected = {role: [p.detach().clone() for p in parameters] for role, parameters in params.items()}
    expected_optim = {role: copy.deepcopy(opt.inner.state_dict()) for role, opt in optimizers.items()}
    counters = dict(warmup_step=2, critic_updates=8, student_updates=0, outer_step=0, data_steps=2)
    local_directory = Path(directory) / f'node-{rank}' / 'checkpoint'
    if rank == 0:
        local_directory.mkdir(parents=True)
        from safetensors.torch import save_file
        save_file({f'weight{i}': p.detach() for i, p in enumerate(params['critic'])},
                  str(local_directory / 'critic.safetensors'))
    dist.barrier()
    warm_parameters = {f'weight{i}': torch.nn.Parameter(torch.zeros_like(p)) for i, p in enumerate(params['critic'])}
    warm_net = SimpleNamespace(original_named_parameters=warm_parameters.items)
    load_model_weights(local_directory, 'critic', warm_net, True)
    assert all(torch.equal(p, saved) for p, saved in zip(warm_parameters.values(), expected['critic']))
    broken_net = SimpleNamespace(original_named_parameters=lambda: [('missing', torch.nn.Parameter(torch.zeros(1)))])
    try:
        load_model_weights(local_directory, 'critic', broken_net, True)
    except ValueError:
        pass
    else:
        raise AssertionError('mismatched weight names accepted')
    torch.manual_seed(100 + rank)
    random.seed(100 + rank)
    np.random.seed(100 + rank)
    save_ema(local_directory, None, True)
    save_rank_states(local_directory, optimizers, schedulers, None, counters, True)
    expected_random = (torch.rand(3), random.random(), np.random.random())
    for role in params:
        for parameter, master in optimizers[role].pairs:
            master.fill_(99)
            parameter.data.fill_(99)
        optimizers[role]._gather_shards()
        optimizers[role].inner.state.clear()
        schedulers[role].step()
    restored = load_rank_state(local_directory, optimizers, schedulers, None, True, False)
    assert restored == counters
    for role, parameters in params.items():
        assert all(torch.equal(p, saved) for p, saved in zip(parameters, expected[role]))
        assert schedulers[role].last_epoch == 2
        assert optimizers[role].inner.state
        for index, state in optimizers[role].inner.state_dict()['state'].items():
            assert all(torch.equal(value, expected_optim[role]['state'][index][key]) for key, value in state.items())
    assert torch.equal(torch.rand(3), expected_random[0])
    assert random.random() == expected_random[1] and np.random.random() == expected_random[2]

    for parameter, master in optimizers['student'].pairs:
        master.fill_(77)
        parameter.data.fill_(77)
    optimizers['student']._gather_shards()
    load_rank_state(local_directory, optimizers, schedulers, None, True, True)
    assert all(torch.all(p == 77) for p in params['student'])
    assert all(torch.equal(p, saved) for p, saved in zip(params['critic'], expected['critic']))
    if rank == 0:
        assert len(list(local_directory.glob('rank-*.pt'))) == 4
        assert not (local_directory / 'model_ema.safetensors').exists()
        assert not any((Path(directory) / f'node-{r}').exists() for r in range(1, 4))
    ema = EMA(.99, optimizers['student'].pairs)
    save_ema(local_directory, ema, True)
    if rank == 0:
        from safetensors.torch import load_file
        merged = load_file(local_directory / 'model_ema.safetensors')
        assert len(merged) == 8 and all(torch.all(t == 77) for t in merged.values())
    lora_checkpoint_roundtrip(rank, directory)
    dist.destroy_process_group()


class InfraTests(unittest.TestCase):
    def test_zero_resolver_weight_skips_generation_and_scoring(self):
        import learn_dmd_bimodal as trainer
        tree = ast.parse(inspect.getsource(trainer.main)).body[0]
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('critic_phase', 'separate_critic_losses')]
        called = []
        weight = torch.nn.Parameter(torch.ones(1))
        net = SimpleNamespace(audio_tokens=lambda x: x, frame_audio=lambda x, *args: x)
        rows = {'latents': torch.zeros(1, 1, 8, 1, 1), 'audio': torch.zeros(1, 1, 1),
                'ofi': torch.arange(8)[None], 'drop': torch.zeros(1, dtype=torch.bool)}

        def rollout(*args, stab_only):
            self.assertTrue(stab_only)
            called.append('stabilizer generation')
            return torch.zeros(1, 1, 4, 1, 1), None

        def forbidden(*args, **kwargs):
            self.fail('resolver generation or critic scoring was called')

        noop = lambda *args, **kwargs: None
        scope = vars(trainer).copy()
        scope.update(config={'critic_rows': 1, 'critic_updates': 1, 'pool_micro_batch': 1,
                             'critic_combined': True, 'gradient_clipping': 2},
                     nets={'student': net, 'critic': net}, prepare=lambda *args: rows,
                     codebook_count=lambda: 16, timed=lambda *args: nullcontext(),
                     generation_noise=lambda x: (torch.zeros_like(x['latents']), torch.zeros_like(x['latents'])),
                     pool_rollout=rollout, fm_weights={'stab': 1, 'res': 0}, memory=noop,
                     grads={'critic': SimpleNamespace(allocate=noop, reduce_clip=lambda *args: 0, release=noop)},
                     optimizers={'critic': SimpleNamespace(step=noop)}, schedulers={'critic': SimpleNamespace(step=noop)},
                     sums=SimpleNamespace(add=noop, frames=noop, buckets=noop), counters={'critic_updates': 0},
                     device=torch.device('cpu'), t_bounds={'stab': (.22, .98), 'res': (.02, .4)}, h=.2,
                     stab_times=lambda *args: torch.full((1, 4), .5), noised=lambda y, *args: (y, torch.zeros_like(y)),
                     stab_history=lambda *args: None, stab_pass=lambda net, rows, x, *args: weight.expand_as(x),
                     resolve_times=forbidden, resolve_pass=forbidden, both_pass=forbidden)
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<critic-driver>', 'exec'), scope)
        scope['optimizer_step'] = lambda role, calibrate=True: scope['optimizers'][role].step()
        scope['probe_critic'] = lambda raw: None
        scope['critic_phase'](None)
        self.assertEqual(called, ['stabilizer generation'])
        self.assertEqual(scope['counters']['critic_updates'], 1)
        self.assertIsNotNone(weight.grad)

    def test_shards_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory(prefix='infra-', dir=os.environ['BIMODAL_TEST_OUTPUT']) as directory:
            mp.spawn(distributed_worker, args=(directory,), nprocs=4, join=True)

    def test_resume_compatibility(self):
        with open('experiments/distillation/aloe-studio-stabilizer-dmd-richmond.toml', 'rb') as handle:
            config = tomllib.load(handle)
        signature = compatibility(config, 'initial-checkpoint-hash', [16, 30], 'data-corpus-hash')
        from distillation.checkpoints import RESUME_KEYS
        meta = {'world_size': 32, 'compatibility': copy.deepcopy(signature),
                'training': {key: config[key] for key in RESUME_KEYS}, 'data_sources': ['shard-a', 'shard-b'],
                'dataset_length': 123, 'counters': {'student_updates': 0}, 'critic_warmup_complete': True}
        validate_resume(meta, config, signature, 32, ['shard-a', 'shard-b'], 123, False)
        validate_resume(meta, config, signature, 32, ['different-shard'], 45, True)
        for changed in [dict(world_size=40), dict(data_sources=['changed']), dict(dataset_length=124)]:
            with self.assertRaises(ValueError):
                validate_resume(meta | changed, config, signature, 32, ['shard-a', 'shard-b'], 123, False)
        for changed in [dict(counters={'student_updates': 1}), dict(critic_warmup_complete=False)]:
            with self.assertRaises(ValueError):
                validate_resume(meta | changed, config, signature, 32, [], 0, True)
        for field, value in [('initial_checkpoint_sha256', 'another-teacher'), ('latent_hw', [30, 16]),
                             ('data_distribution_sha256', 'different-corpus')]:
            with self.assertRaises(ValueError):
                validate_resume(meta, config, signature | {field: value}, 32, [], 0, True)
        changed = copy.deepcopy(config)
        changed['band_steps'] = [1, 1, 2, 4]
        with self.assertRaises(ValueError):
            validate_resume(meta, changed, compatibility(changed, 'initial-checkpoint-hash', [16, 30], 'data-corpus-hash'), 32, [], 0, True)

        lora = {'rank': 32, 'alpha': 32, 'targets': ['transformer.blocks.*.self_attn.*']}
        handoff = config | {'lora': lora, 'critic_resume_optimizer': 'reset', 'critic_resume_allow_data_change': True}
        new_corpus = signature | {'data_distribution_sha256': 'new-disjoint-corpus'}
        validate_resume(meta, handoff, new_corpus, 40, ['new-shard'], 45, True)
        for changed in [dict(critic_resume_optimizer='restore'), dict(critic_resume_optimizer='invalid'),
                        dict(critic_resume_allow_data_change=False)]:
            with self.assertRaises(ValueError):
                validate_resume(meta, handoff | changed, new_corpus, 40, ['new-shard'], 45, True)
        with self.assertRaises(ValueError):
            validate_resume(meta, handoff | {'critic_resume_optimizer': 'restore'}, signature, 32, [], 0, True)
        for field, value in [('initial_checkpoint_sha256', 'another-teacher'), ('latent_hw', [30, 16])]:
            with self.assertRaises(ValueError):
                validate_resume(meta, handoff, new_corpus | {field: value}, 40, [], 0, True)
        with self.assertRaises(ValueError):
            validate_resume(meta, handoff, new_corpus, 32, ['shard-a', 'shard-b'], 123, False)
        lora_meta = meta | {'lora': lora}
        validate_resume(lora_meta, config | {'lora': lora}, signature, 32, ['shard-a', 'shard-b'], 123, False)
        with self.assertRaises(ValueError):
            validate_resume(lora_meta, config | {'lora': lora | {'alpha': 16}}, signature, 32,
                            ['shard-a', 'shard-b'], 123, False)

    def test_corpus_identity_and_overlap(self):
        a, b, c = ('a', 1875, False), ('b', 125, True), ('c', 18, False)
        self.assertEqual(distribution_fingerprint([[a, b], [c]]), distribution_fingerprint([[c, a], [b]]))
        self.assertNotEqual(distribution_fingerprint([[a, b]]), distribution_fingerprint([[a, c]]))
        with self.assertRaises(ValueError):
            distribution_fingerprint([[a, b], [a, c]])

    def test_completion_and_partial_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix='completion-', dir=os.environ['BIMODAL_TEST_OUTPUT']) as directory:
            path = Path(directory)
            shard = path / 'rank-00000.pt'
            shard.write_bytes(b'optimizer state')
            complete_checkpoint(path, {'world_size': 1, 'optimizer_state_retained': True})
            first = read_checkpoint_metadata(path)
            shard.write_bytes(b'partial')
            with self.assertRaises(ValueError):
                read_checkpoint_metadata(path)
            complete_checkpoint(path, {'world_size': 1, 'optimizer_state_retained': True})
            self.assertNotEqual(first['checkpoint_id'], read_checkpoint_metadata(path)['checkpoint_id'])
            shard.unlink()
            complete_checkpoint(path, {'world_size': 1, 'optimizer_state_retained': False})
            with self.assertRaises(ValueError):
                read_checkpoint_metadata(path)

    def test_warm_optimizer_prune_preserves_weights_and_requires_explicit_reset(self):
        with tempfile.TemporaryDirectory(prefix='warm-prune-', dir=os.environ['BIMODAL_TEST_OUTPUT']) as directory:
            path = Path(directory) / 'critic-warmup'
            path.mkdir()
            weights = {'student.safetensors': b'student weights', 'critic.safetensors': b'warmed critic weights'}
            for name, value in weights.items():
                (path / name).write_bytes(value)
            for rank in range(2):
                (path / f'rank-{rank:05d}.pt').write_bytes(b'optimizer and fp32 masters')
            original = complete_checkpoint(path, {'world_size': 2, 'optimizer_state_retained': True,
                                                  'critic_warmup_complete': True, 'counters': {'student_updates': 0}})
            archive_receipt = path.with_name('critic-warmup.archive.json')
            archive_receipt.write_text(json.dumps({'checkpoint_id': original['checkpoint_id']}))
            with self.assertRaises(ValueError):
                prune_optimizer_state(path, 'wrong-generation')
            self.assertEqual(len(list(path.glob('rank-*.pt'))), 2)
            receipt = prune_optimizer_state(path, original['checkpoint_id'])
            self.assertFalse(list(path.glob('rank-*.pt')))
            for name, value in weights.items():
                self.assertEqual((path / name).read_bytes(), value)
                from distillation.checkpoints import file_sha256
                self.assertEqual(receipt['weights'][name]['sha256'], file_sha256(path / name))
            current = read_checkpoint_metadata(path, critic_only=True, optimizer_policy='reset')
            self.assertNotEqual(current['checkpoint_id'], original['checkpoint_id'])
            self.assertEqual(current['optimizer_pruned_from_checkpoint_id'], original['checkpoint_id'])
            self.assertEqual(receipt['checkpoint_id'], current['checkpoint_id'])
            self.assertTrue(receipt['weights_unchanged'])
            self.assertEqual(json.loads(archive_receipt.read_text())['checkpoint_id'], original['checkpoint_id'])
            self.assertEqual(json.loads(path.with_name('critic-warmup.optimizer-prune.json').read_text()), receipt)
            for options in ({}, {'critic_only': True}, {'optimizer_policy': 'reset'}):
                with self.assertRaises(ValueError):
                    read_checkpoint_metadata(path, **options)
            complete_checkpoint(path, current | {'counters': {'student_updates': 1}})
            with self.assertRaises(ValueError):
                read_checkpoint_metadata(path, critic_only=True, optimizer_policy='reset')

    def test_critic_reset_rejects_incomplete_or_corrupt_weights(self):
        with tempfile.TemporaryDirectory(prefix='warm-invalid-', dir=os.environ['BIMODAL_TEST_OUTPUT']) as directory:
            path = Path(directory)
            critic = path / 'critic.safetensors'
            critic.write_bytes(b'critic')
            original = complete_checkpoint(path, {'world_size': 1, 'optimizer_state_retained': True,
                                                  'critic_warmup_complete': True, 'counters': {'student_updates': 0}})
            with self.assertRaises(ValueError):
                read_checkpoint_metadata(path, critic_only=True, optimizer_policy='reset')
            with self.assertRaises(ValueError):
                prune_optimizer_state(path, original['checkpoint_id'])
            complete_checkpoint(path, original | {'optimizer_state_retained': False})
            read_checkpoint_metadata(path, critic_only=True, optimizer_policy='reset')
            critic.write_bytes(b'partial critic')
            with self.assertRaises(ValueError):
                read_checkpoint_metadata(path, critic_only=True, optimizer_policy='reset')


if __name__ == '__main__':
    unittest.main()
