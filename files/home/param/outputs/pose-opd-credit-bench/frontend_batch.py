"""Standalone exact-input frontend batch comparison; never trains a model."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/param/outputs/pose-opd-credit-bench/frontend-batch')
REPO = Path('/home/param/pose-worktrees/opd-bench-b320-static-v7')
RUN = Path('/home/param/runs/pose-runs/pose-opd-bench-b320-static-v7')
VOX = '/home/param/models/imported-weights/aloe-ft-vox17127-nopersona/assets/voxtral-realtime'

def emit(value):
    print(json.dumps(value), flush=True)

def prepare():
    import numpy as np
    sys.path.insert(0, str(REPO))
    from finetune.pose_opd_credit.config import TrainingConfig
    from finetune.pose_opd_credit.data import ProductionRows
    config = TrainingConfig.from_dict(json.loads((REPO / 'benchmark-config.json').read_text()))
    items = json.loads((RUN / 'plan.json').read_text())['updates'][0][:8]
    assert all(item is not None for item in items)
    data = ProductionRows(config)
    rows, waveforms = [], []
    started = time.monotonic()
    for item in items:
        stamp = time.monotonic()
        row = data.load(item, 0)
        wave = row['wav']
        assert row['conversation_frames'] == 1500 and wave.shape == (1500 * 1920,)
        waveforms.append(wave)
        rows.append({'item': item, 'row_key': row['row_key'], 'audio_reference': json.loads(row['aref']),
                     'wave_sha256': hashlib.sha256(wave.tobytes()).hexdigest(),
                     'load_seconds': time.monotonic() - stamp})
        emit({'event': 'waveform', **rows[-1]})
    ROOT.mkdir(parents=True, exist_ok=True)
    np.save(ROOT / 'waves.npy', np.stack(waveforms), allow_pickle=False)
    document = {'source': str(REPO), 'plan': str(RUN / 'plan.json'), 'rows': rows,
                'frames': 1500, 'waveform_shape': [8, 1500 * 1920], 'dtype': 'float32',
                'data_load_seconds': time.monotonic() - started, 'rowcache_env': os.environ.get('ALOE_ROWCACHE')}
    (ROOT / 'inputs.json').write_text(json.dumps(document, indent=2))
    emit({'event': 'inputs_ready', **document})

def benchmark():
    lockpath = Path('/home/param/runs/pose-runs/.duplex-distill-gpus.lock')
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if apps:
            raise RuntimeError('GPU processes remain; refusing frontend benchmark: ' + apps)
        import numpy as np
        import torch
        torch.set_num_threads(1)
        waveforms = np.load(ROOT / 'waves.npy', allow_pickle=False)
        receipt = json.loads((ROOT / 'inputs.json').read_text())
        assert waveforms.shape == (8, 1500 * 1920) and waveforms.dtype == np.float32
        for wave, row in zip(waveforms, receipt['rows']):
            assert hashlib.sha256(wave.tobytes()).hexdigest() == row['wave_sha256']
        sys.path.insert(0, str(REPO / 'finetune/duplex_distill/production/lib'))
        import audio_s4q
        started = time.monotonic()
        front = audio_s4q.make_front({'s4_audio': 'flac', 's4_vox_model': VOX, 's4_enc_hidden': 1280}, 'cuda:0')
        torch.cuda.synchronize()
        emit({'event': 'frontend_ready', 'seconds': time.monotonic() - started})
        waves = torch.from_numpy(waveforms)
        refs = [{'row_key': row['row_key'], 'n': 1500} for row in receipt['rows']]
        import voxtral_front
        with torch.no_grad():
            wgpu = waves.cuda()
            single_resampled = torch.cat([voxtral_front.resample(wgpu[i:i+1], 24000, 16000) for i in range(8)])
            batch_resampled = voxtral_front.resample(wgpu, 24000, 16000)
            single_mel = torch.cat([front.fe.log_mel(single_resampled[i:i+1]) for i in range(8)])
            batch_mel = front.fe.log_mel(batch_resampled)
            preprocessing = {'resample_max_abs': float((single_resampled - batch_resampled).abs().max()),
                             'log_mel_max_abs': float((single_mel - batch_mel).abs().max()),
                             'resample_relative_rms': float((single_resampled - batch_resampled).square().mean().sqrt() / single_resampled.square().mean().sqrt().clamp_min(1e-30)),
                             'log_mel_relative_rms': float((single_mel - batch_mel).square().mean().sqrt() / single_mel.square().mean().sqrt().clamp_min(1e-30))}
            del wgpu, single_resampled, batch_resampled, single_mel, batch_mel
        emit({'event': 'preprocessing_parity', **preprocessing})
        def sequential():
            return torch.cat([front.encode([waves[i]], [refs[i]]).cpu() for i in range(8)], dim=0)
        def batched():
            return front.fe.encode(waves, 1500).cpu()
        def measure(name, function, repeats):
            timings, value, peaks = [], None, []
            for iteration in range(repeats):
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                stamp = time.monotonic()
                value = function()
                torch.cuda.synchronize()
                elapsed = time.monotonic() - stamp
                peak = {'allocated_gib': torch.cuda.max_memory_allocated() / 2 ** 30,
                        'reserved_gib': torch.cuda.max_memory_reserved() / 2 ** 30}
                timings.append(elapsed); peaks.append(peak)
                emit({'event': 'measurement', 'mode': name, 'iteration': iteration,
                      'seconds': elapsed, **peak})
            return value, {'seconds': timings, 'memory': peaks}
        reference, seq = measure('sequential8', sequential, 3)
        candidate, batch = measure('batched8', batched, 3)
        left, right = reference.float(), candidate.float()
        delta = right - left
        diff = {'max_abs': float(delta.abs().max()),
                'relative_rms': float(delta.square().mean().sqrt() / left.square().mean().sqrt().clamp_min(1e-30)),
                'different_fraction': float((reference != candidate).float().mean()),
                'per_row_relative_rms': [float(d.square().mean().sqrt() / r.square().mean().sqrt().clamp_min(1e-30)) for d, r in zip(delta, left)],
                'finite': bool(torch.isfinite(candidate).all())}
        result = {'input': receipt, 'sequential8': seq, 'batched8': batch, 'latent_shape': list(candidate.shape),
                  'dtype': str(candidate.dtype), 'difference': diff,
                  'preprocessing_parity': preprocessing,
                  'steady_speedup': sum(seq['seconds'][1:]) / sum(batch['seconds'][1:]),
                  'semantics': 'Identical gained waveforms; equal1500frames; native batched resample/STFT/encoder; no tail padding; no input cache changes',
                  'production_changed': False}
        (ROOT / 'result.json').write_text(json.dumps(result, indent=2))
        emit({'event': 'RESULT', **result})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-inputs', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare_inputs else benchmark()
