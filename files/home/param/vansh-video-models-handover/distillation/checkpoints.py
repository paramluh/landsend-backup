import hashlib
import json
import os
import random
import uuid
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open

from utils import saver


ROLLOUT_KEYS = ('band_steps', 'resolve_steps', 'history_t', 't_start', 'pose_latents', 'mimi_codes', 'bad_audio',
                'stab_t_min', 'stab_t_max', 'res_t_min', 'res_t_max', 'fm_weight_stab', 'fm_weight_res')
RESUME_KEYS = ('student_kinds', 'critic_updates', 'critic_rows', 'optimizer', 'lr_warmup_steps',
               'critic_lr_warmup_steps', 'critic_warmup_steps', 'ema_decay_steps', 'seed', 'gradient_clipping',
               'normalizer_epsilon')
CRITIC_RESUME_OPTIMIZER = 'restore'


def file_sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def distribution_fingerprint(shards):
    rows = [tuple(row) for shard in shards for row in shard]
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError('Node-local training CSVs overlap; choose disjoint shards before training')
    return hashlib.sha256(json.dumps(sorted(rows), separators=(',', ':')).encode()).hexdigest()


def write_json_atomic(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def complete_checkpoint(directory, meta):
    meta = dict(meta, checkpoint_id=uuid.uuid4().hex)
    write_json_atomic(directory / 'metadata.json', meta)
    files = [{'path': path.name, 'size': path.stat().st_size} for path in sorted(directory.iterdir())
             if path.is_file() and not path.name.endswith('.tmp') and path.name != 'checkpoint.complete.json']
    write_json_atomic(directory / 'checkpoint.complete.json',
                      {'schema_version': 1, 'checkpoint_id': meta['checkpoint_id'], 'files': files})
    return meta


def compatibility(config, checkpoint_sha256, latent_hw, data_distribution_sha256):
    return {'initial_checkpoint_sha256': checkpoint_sha256, 'latent_hw': list(latent_hw),
            'data_distribution_sha256': data_distribution_sha256,
            'audio': config['model']['audio'], 'rollout': {key: config.get(key) for key in ROLLOUT_KEYS}}


def validate_resume(meta, config, expected, world_size, data_sources, dataset_length, critic_only):
    policy = config.get('critic_resume_optimizer', CRITIC_RESUME_OPTIMIZER)
    if policy not in ('restore', 'reset'):
        raise ValueError('critic_resume_optimizer must be restore or reset')
    reset = critic_only and policy == 'reset'
    if meta['world_size'] != world_size and not reset:
        raise ValueError(f"checkpoint world_size={meta['world_size']}, requested {world_size}; optimizer resharding required")
    saved_signature, expected_signature = meta['compatibility'], expected
    if critic_only and config.get('critic_resume_allow_data_change') is True:
        saved_signature = {k: v for k, v in saved_signature.items() if k != 'data_distribution_sha256'}
        expected_signature = {k: v for k, v in expected_signature.items() if k != 'data_distribution_sha256'}
    if saved_signature != expected_signature:
        raise ValueError('checkpoint teacher, initial student, rollout, scoring, conditioning, or data corpus differs')
    if not reset and meta.get('lora') != config.get('lora'):
        raise ValueError('checkpoint LoRA settings differ; dense-to-LoRA critic handoff requires critic_resume_optimizer=reset')
    if critic_only:
        if meta['counters']['student_updates'] or not meta['critic_warmup_complete']:
            raise ValueError('critic_resume_from requires a completed warmup checkpoint before any student update')
    else:
        if meta['training'] != {key: config[key] for key in RESUME_KEYS}:
            raise ValueError('full resume training settings differ from checkpoint')
        if meta['data_sources'] != data_sources or meta['dataset_length'] != dataset_length:
            raise ValueError('full resume data shards or row batching differ from checkpoint')


@torch.no_grad()
def load_model_weights(directory, role, net, node_sharded):
    rank = dist.get_rank()
    parameters = dict(net.original_named_parameters())
    reader = safe_open(str(Path(directory) / f'{role}.safetensors'), framework='pt', device='cpu') \
        if rank == 0 or not node_sharded else nullcontext(None)
    with reader as handle:
        error = [None]
        if handle is not None:
            if set(handle.keys()) != set(parameters):
                error[0] = f'{role} checkpoint parameter names differ from model'
            elif any(tuple(handle.get_slice(name).get_shape()) != tuple(p.shape) for name, p in parameters.items()):
                error[0] = f'{role} checkpoint parameter shapes differ from model'
        if node_sharded:
            dist.broadcast_object_list(error, src=0, group=saver.gloo)
        if error[0]:
            raise ValueError(error[0])
        for name, parameter in parameters.items():
            if handle is not None:
                parameter.copy_(handle.get_tensor(name))
            if node_sharded:
                dist.broadcast(parameter.data, src=0)


def rng_state():
    numpy = np.random.get_state()
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
            'python': random.getstate(), 'numpy': (numpy[0], numpy[1].tolist(), *numpy[2:])}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))


def save_rank_states(directory, optimizers, schedulers, ema, counters, node_sharded):
    rank, world = dist.get_rank(), dist.get_world_size()

    def state():
        return {'optimizers': {role: opt.state_dict() for role, opt in optimizers.items()},
                'schedulers': {role: scheduler.state_dict() for role, scheduler in schedulers.items()},
                'ema': ema.state_dict() if ema is not None else None, 'counters': dict(counters), 'rng': rng_state()}

    if node_sharded:
        for src in range(world):
            shard = saver._to_rank0(state() if rank == src else None, src)
            if rank == 0:
                torch.save(shard, directory / f'rank-{src:05d}.pt')
            del shard
        dist.barrier(group=saver.gloo)
    else:
        torch.save(state(), directory / f'rank-{rank:05d}.pt')
        dist.barrier()


def load_rank_state(directory, optimizers, schedulers, ema, node_sharded, critic_only):
    rank, world = dist.get_rank(), dist.get_world_size()
    if node_sharded:
        mine = None
        for dst in range(world):
            shard = torch.load(directory / f'rank-{dst:05d}.pt', map_location='cpu', weights_only=True) if rank == 0 else None
            shard = saver._from_rank0(shard, dst)
            if rank == dst:
                mine = shard
            del shard
    else:
        mine = torch.load(directory / f'rank-{rank:05d}.pt', map_location='cpu', weights_only=True)
    for role in ('critic',) if critic_only else ('student', 'critic'):
        optimizers[role].load_state_dict(mine['optimizers'][role])
        schedulers[role].load_state_dict(mine['schedulers'][role])
    if not critic_only:
        if ema is not None:
            ema.load_state_dict(mine['ema'])
        restore_rng(mine['rng'])
    return mine['counters']


def save_ema(directory, ema, node_sharded):
    if ema is None:
        return
    if not node_sharded:
        ema.save(directory, torch.bfloat16)
        return
    rank = dist.get_rank()
    state = {}
    for src in range(dist.get_world_size()):
        shard = {name: shadow.to('cpu', torch.bfloat16) for name, _, shadow in ema.triples} if rank == src else None
        shard = saver._to_rank0(shard, src)
        if rank == 0:
            state.update(shard)
    if rank == 0:
        from safetensors.torch import save_file
        save_file(state, str(directory / 'model_ema.safetensors'), metadata={'format': 'pt'})
    dist.barrier(group=saver.gloo)


def read_checkpoint_metadata(directory, *, critic_only=False, optimizer_policy=CRITIC_RESUME_OPTIMIZER):
    if optimizer_policy not in ('restore', 'reset'):
        raise ValueError('optimizer_policy must be restore or reset')
    weights_only = critic_only and optimizer_policy == 'reset'
    path = Path(directory)
    marker = json.loads((path / 'checkpoint.complete.json').read_text())
    meta = json.loads((path / 'metadata.json').read_text())
    if marker['checkpoint_id'] != meta['checkpoint_id']:
        raise ValueError('Checkpoint completion marker does not match metadata')
    if any(not (path / item['path']).is_file() or (path / item['path']).stat().st_size != item['size'] for item in marker['files']):
        raise ValueError('Checkpoint is incomplete or a saved file size changed')
    names = {item['path'] for item in marker['files']}
    if weights_only and ('critic.safetensors' not in names or not meta.get('critic_warmup_complete')
                         or meta.get('counters', {}).get('student_updates') != 0):
        raise ValueError('Weights-only critic reset requires a completed warmup checkpoint before any student update')
    if meta['optimizer_state_retained']:
        if not all(f'rank-{rank:05d}.pt' in names for rank in range(meta['world_size'])):
            raise ValueError('Checkpoint claims optimizer state but is missing rank shards')
    elif not weights_only:
        raise ValueError('Optimizer shards were pruned; resume from the latest checkpoint or its archived copy')
    return meta


def prune_optimizer_state(directory, checkpoint_id, *, verify_weight_hashes=True):
    """Retire one completed generation's optimizer shards, retaining its weights and lineage."""
    path = Path(directory)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError('Checkpoint must be an absolute, real directory')
    meta = read_checkpoint_metadata(path)
    if meta['checkpoint_id'] != checkpoint_id:
        raise ValueError('Checkpoint generation differs from requested optimizer prune')
    shards = sorted(path.glob('rank-*.pt'))
    if {p.name for p in shards} != {f'rank-{rank:05d}.pt' for rank in range(meta['world_size'])}:
        raise ValueError('Unexpected optimizer shard files; checkpoint must remain untouched')
    weights = sorted(path.glob('*.safetensors'))
    if not weights or any(p.is_symlink() for p in weights + shards):
        raise ValueError('Checkpoint weights/shards must be real files')

    def generation(file):
        stat = file.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    generations = {file.name: generation(file) for file in weights}
    audited = {file.name: {'size': file.stat().st_size,
                          **({'sha256': file_sha256(file)} if verify_weight_hashes else {})} for file in weights}
    if read_checkpoint_metadata(path)['checkpoint_id'] != checkpoint_id or any(
            generation(file) != generations[file.name] for file in weights):
        raise ValueError('Checkpoint changed before optimizer pruning')
    removed_bytes = sum(file.stat().st_size for file in shards)
    (path / 'checkpoint.complete.json').unlink()
    for file in shards:
        file.unlink()
    for file in weights:
        if generation(file) != generations[file.name] or (
                verify_weight_hashes and file_sha256(file) != audited[file.name]['sha256']):
            raise ValueError('Checkpoint weights changed during optimizer pruning; completion marker withheld')
    updated = complete_checkpoint(path, dict(meta, optimizer_state_retained=False,
                                            optimizer_pruned_from_checkpoint_id=checkpoint_id,
                                            optimizer_pruned_weights=audited))
    receipt = {'path': str(path), 'previous_checkpoint_id': checkpoint_id,
               'checkpoint_id': updated['checkpoint_id'], 'optimizer_state_retained': False,
               'removed_shards': [file.name for file in shards], 'removed_bytes': removed_bytes,
               'weights_unchanged': True, 'weights': audited}
    write_json_atomic(path.with_name(path.name + '.optimizer-prune.json'), receipt)
    return receipt


def read_metadata(directory, *, critic_only=False, optimizer_policy=CRITIC_RESUME_OPTIMIZER):
    box = [read_checkpoint_metadata(directory, critic_only=critic_only, optimizer_policy=optimizer_policy)
           if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(box, src=0)
    return box[0]
