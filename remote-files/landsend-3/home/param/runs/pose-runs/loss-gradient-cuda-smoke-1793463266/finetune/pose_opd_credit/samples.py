"""Atomic, exact snapshots for the dashboard's four samples per update."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np


def sample_origins(update_items, rows_per_rank, count):
    """Prefer rank zero, then subsequent ranks, without training randomness."""
    if rows_per_rank < 1 or count < 0:
        raise ValueError('Sample selection requires positive rank capacity and nonnegative count')
    origins = [divmod(slot, rows_per_rank) for slot, item in enumerate(update_items)
               if item is not None][:count]
    if len(origins) != count:
        raise ValueError(f'Global update has fewer than {count} real rows for the sample archive')
    return origins


def sample_slots(update_items, rows_per_rank, count, rank=0):
    """Local slots selected from the global update, including partial tails."""
    return [slot for owner, slot in sample_origins(update_items, rows_per_rank, count)
            if owner == rank]


def sample_records(payload, rollouts, waveforms, slots, rank):
    """Stage only archive metadata and selected CPU trajectories for transfer."""
    records = {}
    for slot in slots:
        row, _ = payload[slot]
        if row.get('dummy', False):
            raise ValueError('Cannot export a dummy rollout as a real sample')
        metadata = {name: row[name] for name in ('item', 'row_key', 'aref', 'conditioning')}
        records[rank, slot] = (metadata, rollouts[slot], waveforms[slot])
    return records


def gather_samples(records, origins, *, rank, group):
    """Transfer selected remote rows to rank zero over the CPU control group.

    Full batches stay local. Point-to-point transfers avoid gather_object's
    world-size multiplication of the largest serialized waveform on rank zero.
    Every rank must finish staging its records before any rank enters here.
    """
    import torch.distributed as dist
    owners = sorted({owner for owner, _ in origins} - {0})
    if rank == 0:
        gathered = dict(records)
        for owner in owners:
            received = [None]
            dist.recv_object_list(received, src=owner, group=group)
            gathered.update(received[0])
        if set(gathered) != set(origins):
            raise ValueError('Gathered dashboard samples do not match the selected real rows')
        return gathered
    if rank in owners:
        dist.send_object_list([records], dst=0, group=group)
    return None


def save_sample_records(root, *, records, origins, **kwargs):
    """Keep source rank/slot identity after packing the gathered archive rows."""
    values = [records[origin] for origin in origins]
    return save_samples(root, payload=[(row, None) for row, _, _ in values],
        rollouts=[rollout for _, rollout, _ in values],
        waveforms={slot: wave for slot, (_, _, wave) in enumerate(values)},
        slots=range(len(values)), origins=origins, **kwargs)


def _json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def save_samples(root, *, update, payload, rollouts, waveforms, slots, fps, source_identity,
                 origins=None):
    """Publish only after all complete, validated numpy payloads exist.

    Media codes on the physical timeline include the forced prefill and final
    delayed frames. Model-timeline arrays and sampled masks are retained too.
    Only rank zero calls this function, after collecting any selected remote
    rows needed to fill the archive for a partial update.
    """
    raw_root = Path(root) / 'raw-rollouts'
    destination = raw_root / f'update-{update:06d}'
    folder = raw_root / f'.update-{update:06d}-{uuid.uuid4().hex}'
    folder.mkdir(parents=True)
    manifest = folder / 'manifest.json'
    entries = []
    offsets = np.arange(32, dtype=np.int64) * 2048
    for slot in slots:
        source_rank, source_slot = (0, slot) if origins is None else origins[slot]
        seed, _ = payload[slot]
        rollout = rollouts[slot]
        if seed.get('dummy', False):
            raise ValueError('Cannot export a dummy rollout as a real sample')
        frames, prefix = int(rollout.conversation_frames), int(rollout.prefix)
        wave = np.asarray(waveforms[slot], dtype=np.float32)
        if wave.shape != (frames * 1920,) or not np.isfinite(wave).all():
            raise ValueError('Saved input audio must match the entire physical row')
        arrays = {'input_audio': wave,
                  'text_ids': np.asarray(rollout.text[prefix:prefix + frames], dtype=np.int64),
                  'text_model': np.asarray(rollout.text),
                  'audio_model': np.asarray(rollout.audio),
                  'pose_model': np.asarray(rollout.pose)}
        for stream in ('audio', 'pose'):
            codes = np.asarray(getattr(rollout, stream)[prefix + 2:prefix + 2 + frames], dtype=np.int64) - offsets
            if codes.shape != (frames, 32) or np.any(codes < 0) or np.any(codes >= 2048):
                raise ValueError(f'Invalid or incomplete physical {stream} codes')
            arrays[f'{stream}_codes'] = codes.T.copy()
        for stream, delay in (('text', 0), ('audio', 2), ('pose', 2)):
            mask = np.asarray(rollout.sampled_masks[stream], dtype=bool)
            arrays[f'sampled_{stream}_mask'] = mask[prefix + delay:prefix + delay + frames]
            arrays[f'sampled_{stream}_model_mask'] = mask
            if stream in rollout.behavior_logprobs:
                arrays[f'behavior_{stream}_logprobs'] = np.asarray(rollout.behavior_logprobs[stream])
        name = f'sample-u{update:06d}-r{source_rank:03d}-s{source_slot:02d}.npz'
        target = folder / name
        temporary = target.with_suffix('.npz.tmp')
        with temporary.open('wb') as output:
            # Avoid CPU compression on the synchronous training path.
            np.savez(output, **arrays)
        os.replace(temporary, target)
        item = list(map(int, seed['item']))
        entries.append({'id': f'u{update}-r{source_rank}-s{source_slot}', 'file': name,
            'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size,
            'row_id': seed['row_key'], 'item': item, 'rank': source_rank, 'slot': source_slot,
            'orientation': 'assistant → user' if item[2] else 'user → assistant',
            'frames': frames, 'fps': fps, 'prefill_frames': int(rollout.prefill_frames),
            'media_delay_frames': 2, 'prefix_frames': prefix,
            'source_audio': json.loads(seed['aref']), 'conditioning': seed['conditioning'],
            'input_sample_rate': 24000,
            'incoming_audio_delay_samples': int(seed.get('incoming_audio_delay_samples', 0))})
    document = {'schema_version': 1, 'synthetic': False, 'update': update,
                'source_identity': source_identity, 'samples': entries}
    _json(manifest, document)
    if destination.exists():
        # A resume can repeat updates after the last durable optimizer save.
        # Preserve the previous attempt rather than blocking or deleting it.
        os.replace(destination, raw_root / f'history-update-{update:06d}-{uuid.uuid4().hex}')
    os.replace(folder, destination)
    return {'manifest': str(destination / 'manifest.json'), 'count': len(entries),
            'bytes': sum(row['bytes'] for row in entries)}
