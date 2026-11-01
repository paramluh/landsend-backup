"""Atomic, exact snapshots for the dashboard's four samples per update."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np


def sample_slots(update_items, rows_per_rank, count):
    """Choose real rank-zero rows without consuming any training randomness."""
    slots = [slot for slot, item in enumerate(update_items[:rows_per_rank]) if item is not None][:count]
    if len(slots) != count:
        raise ValueError(f'Rank zero has fewer than {count} real rows for the sample archive')
    return slots


def _json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def save_samples(root, *, update, payload, rollouts, waveforms, slots, fps, source_identity):
    """Publish only after all complete, validated numpy payloads exist.

    Media codes on the physical timeline include the forced prefill and final
    delayed frames. Model-timeline arrays and sampled masks are retained too.
    Only rank zero calls this function; its selected rows exist in every update
    of the approved plan, including the final partial update.
    """
    raw_root = Path(root) / 'raw-rollouts'
    destination = raw_root / f'update-{update:06d}'
    folder = raw_root / f'.update-{update:06d}-{uuid.uuid4().hex}'
    folder.mkdir(parents=True)
    manifest = folder / 'manifest.json'
    entries = []
    offsets = np.arange(32, dtype=np.int64) * 2048
    for slot in slots:
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
        name = f'sample-u{update:06d}-r000-s{slot:02d}.npz'
        target = folder / name
        temporary = target.with_suffix('.npz.tmp')
        with temporary.open('wb') as output:
            # Avoid CPU compression on the synchronous training path.
            np.savez(output, **arrays)
        os.replace(temporary, target)
        item = list(map(int, seed['item']))
        entries.append({'id': f'u{update}-r0-s{slot}', 'file': name,
            'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size,
            'row_id': seed['row_key'], 'item': item, 'rank': 0, 'slot': slot,
            'orientation': 'assistant → user' if item[2] else 'user → assistant',
            'frames': frames, 'fps': fps, 'prefill_frames': int(rollout.prefill_frames),
            'media_delay_frames': 2, 'prefix_frames': prefix,
            'source_audio': json.loads(seed['aref']), 'conditioning': seed['conditioning'],
            'input_sample_rate': 24000})
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
