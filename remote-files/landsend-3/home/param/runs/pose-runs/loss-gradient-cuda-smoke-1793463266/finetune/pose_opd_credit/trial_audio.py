"""Export real paired rollouts for listening, without redecoding or rendering pose."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def save_trial_audio(result, rows, seeds, config, destination, *, row_offset=0):
    """Write exact float32 received PCM: left student, right frozen partner.

    Both channels already contain the recorded inverse gain and the same
    delivery delay used during generation. Do not loudness-normalize them again.
    The initial real prefill remains audible and is explicitly identified.
    """
    from scipy.io.wavfile import write
    from .conversation_metrics import finalize_conversation_metrics
    if result['partner_sampling'] != config.partner_sampling:
        raise ValueError('Actual partner sampling policy differs from the listening recipe')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if not (len(rows) == len(seeds) == len(result['rollouts']) == len(result['student_waveforms'])
            == len(result['waveforms']) == len(result['partner_rollouts'])):
        raise ValueError('Listening export requires complete paired rollout rows')
    entries = []
    for index, (row, seed, rollout) in enumerate(zip(rows, seeds, result['rollouts'], strict=True)):
        frames = int(row['conversation_frames'])
        student = np.asarray(result['student_waveforms'][index], dtype=np.float32)
        partner = np.asarray(result['waveforms'][index], dtype=np.float32)
        if student.shape != (frames * 1920,) or partner.shape != student.shape:
            raise ValueError('Listening PCM must cover the complete physical conversation')
        stereo = np.stack((student, partner), axis=-1)
        if not np.isfinite(stereo).all():
            raise ValueError('Listening PCM must be finite')
        stem = f'conversation-{row_offset + index + 1:02d}'
        path = destination / (stem + '.wav')
        write(path, 24000, stereo)
        tokens = {'student_text': rollout.text, 'student_audio': rollout.audio, 'student_pose': rollout.pose,
                  **{f'student_sampled_{key}': value for key, value in rollout.sampled_masks.items()},
                  'partner_text': np.asarray(result['partner_rollouts'][index]['lane_b']),
                  'partner_audio': np.asarray(result['partner_rollouts'][index]['dep_b'])}
        np.savez_compressed(destination / (stem + '-tokens.npz'), **tokens)
        entries.append({'id': stem, 'row_key': str(row['row_key']), 'item': list(map(int, row['item'])),
            'student_seed': int(seed), 'partner_seed': int(seed) ^ 0x5A17B39D,
            'frames': frames, 'seconds': frames / 12.5,
            'prefill_frames': int(row['prefill_frames']), 'prefill_seconds': int(row['prefill_frames']) / 12.5,
            'student_conditioning': row['conditioning'], 'partner_conditioning': row['partner']['conditioning'],
            'audio': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'tokens': stem + '-tokens.npz', 'pose_generated': True,
            'conversation_metrics': finalize_conversation_metrics(result['conversation_metrics'][index])})
    manifest = {'format': 'pose-opd-conversation-audio-v1', 'synthetic': False,
        'channels': ['student', 'frozen_audio_duplex_partner'], 'sample_rate': 24000,
        'audio_format': 'IEEE float32 WAV, no extra gain or loudness normalization',
        'audio_clock': 'PCM delivered to the opposite model, including the configured delivery delay',
        'delivery_delay_samples': result['delivery_delay_samples'],
        'student': config.student, 'partner': config.resolved_partner_checkpoint,
        'student_sampling': 'untruncated temperature-one categorical on text/audio/pose',
        'partner_sampling': config.partner_sampling, 'samples': entries}
    temporary = destination / 'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    temporary.replace(destination / 'manifest.json')
    return manifest
