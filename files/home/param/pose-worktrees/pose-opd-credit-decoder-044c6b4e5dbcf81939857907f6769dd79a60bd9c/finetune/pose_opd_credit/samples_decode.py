"""Independent CPU decoder for the four atomically archived rollout samples.

Consumes physical-timeline arrays, including conversation prefill. Neither
codec is restarted inside a conversation. Pose is a normalized 2D skeleton;
its confidence channel is never presented as depth.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
POSE_CODE = REPO / 'infer/engine/posetok'
DEFAULT_MIMI = Path('/home/param/models/mimi')
DEFAULT_POSE = Path('/home/param/models/pose-tokenizer-32cb-200k-noshift')
DEFAULT_TEXT = Path('/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-32B')
SAFE_ID = re.compile(r'[A-Za-z0-9_-]+\Z')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_sample(directory, item):
    """Reject malformed or shifted data rather than repairing its timeline."""
    name = item['file']
    if Path(name).name != name or not SAFE_ID.fullmatch(item['id']):
        raise ValueError('Unsafe sample file or identifier')
    frames = int(item['frames'])
    if frames <= 0 or float(item['fps']) != 12.5 or not 0 <= int(item['prefill_frames']) < frames:
        raise ValueError('Invalid sample frame rate/length/prefill')
    source = Path(directory) / name
    if 'bytes' in item and source.stat().st_size != int(item['bytes']):
        raise ValueError('Sample file size does not match its committed manifest')
    if 'sha256' in item and hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
        raise ValueError('Sample SHA256 does not match its committed manifest')
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in (
            'input_audio', 'text_ids', 'audio_codes', 'pose_codes',
            'sampled_text_mask', 'sampled_audio_mask', 'sampled_pose_mask')}
    wav = arrays['input_audio']
    if wav.shape != (frames * 1920,) or wav.dtype.kind != 'f' or not np.isfinite(wav).all():
        raise ValueError('Input audio must cover the complete physical timeline')
    text = arrays['text_ids']
    if text.shape != (frames,) or text.dtype.kind not in 'iu' or np.any(text < 0) or np.any(text >= 151682):
        raise ValueError('Invalid executable text IDs')
    for stream in ('audio', 'pose'):
        codes = arrays[stream + '_codes']
        if (codes.shape != (32, frames) or codes.dtype.kind not in 'iu'
                or np.any(codes < 0) or np.any(codes >= 2048)):
            raise ValueError(f'{stream} codes must be local IDs [32, frames]')
    for stream in ('text', 'audio', 'pose'):
        mask = arrays['sampled_' + stream + '_mask']
        if mask.shape != (frames,) or mask.dtype != np.bool_:
            raise ValueError('Sample masks must use the physical timeline')
        if mask[:int(item['prefill_frames'])].any():
            raise ValueError('Conversation prefill must not be marked sampled')
    return arrays


def pose_plane(features, parents, fps=25.0):
    """Integrate (off_y, off_x) in shoulder widths; keep confidence separate."""
    f = np.asarray(features, np.float32)
    if f.ndim == 2 and f.shape[-1] == 165:
        f = f.reshape(-1, 55, 3)
    parents = np.asarray(parents, np.int64)
    if f.ndim != 3 or f.shape[1:] != (55, 3) or parents.shape != (55,) or not np.isfinite(f).all():
        raise ValueError('Expected finite decoded 55-joint offsets/confidence')
    positions = np.zeros((len(f), 55, 2), np.float32)
    done = set()

    def visit(index, visiting):
        if index in done:
            return
        if index in visiting or not -1 <= int(parents[index]) < 55:
            raise ValueError('Invalid pose parent graph')
        parent = int(parents[index])
        if parent >= 0:
            visit(parent, visiting | {index})
            positions[:, index] = positions[:, parent]
        positions[:, index] += f[:, index, :2]
        done.add(index)

    for index in range(55):
        visit(index, set())
    plane = np.zeros((len(f), 55, 3), np.float32)
    plane[..., 0] = positions[..., 1]
    plane[..., 1] = -positions[..., 0]
    names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
             'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
             'left_wrist', 'right_wrist', 'left_hip', 'right_hip']
    names += [f'{side}_hand_{i}' for side in ('left', 'right') for i in range(21)]
    return {'fps': float(fps), 'start_time_seconds': 0, 'up_axis': 'y',
            'units': 'shoulder_widths', 'dimensionality': 2,
            'coordinate_system': '2D canonical root-relative pose; z=0; not reconstructed 3D',
            'root_position': [0, 0], 'normalization': 'unit frame-zero shoulder width',
            'joints': names, 'parents': parents.tolist(), 'frames': plane.tolist(),
            'confidence': f[..., 2].tolist()}


def lane_transcript(ids, tokenizer):
    words, current = [], []
    for token in np.asarray(ids).tolist():
        if token >= 151643:
            if current:
                words.append(tokenizer.decode(current).strip())
                current = []
        else:
            piece = tokenizer.convert_ids_to_tokens(token)
            if piece is None:
                raise ValueError(f'Text tokenizer is missing executable token {token}')
            if current and piece.startswith('Ġ'):
                words.append(tokenizer.decode(current).strip())
                current = []
            current.append(token)
    if current:
        words.append(tokenizer.decode(current).strip())
    return ' '.join(word for word in words if word)


class CPUCodecs:
    def __init__(self, mimi=DEFAULT_MIMI, pose_tokenizer=DEFAULT_POSE,
                 text_tokenizer=DEFAULT_TEXT, threads=2):
        import torch
        from transformers import AutoTokenizer, MimiModel
        if threads < 1:
            raise ValueError('threads must be positive')
        torch.set_num_threads(threads)
        self.torch = torch
        self.text = AutoTokenizer.from_pretrained(str(text_tokenizer), local_files_only=True)
        if len(self.text) < 151643:
            raise ValueError('Complete Qwen3 tokenizer assets are required, not configuration files alone')
        self.mimi = MimiModel.from_pretrained(str(mimi), local_files_only=True).to('cpu').eval()
        self.mimi.config.use_cache = False
        if str(POSE_CODE) not in sys.path:
            sys.path.insert(0, str(POSE_CODE))
        from pose_tokenizer import PoseTokenizer
        self.pose = PoseTokenizer.from_pretrained(str(pose_tokenizer), device='cpu', local_files_only=True)
        spec = importlib.util.spec_from_file_location('_sample_pose_kinematic', POSE_CODE / 'kinematic.py')
        geometry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(geometry)
        self.parents = geometry.PARENT_INDEX
        self.metadata = {'mimi': str(Path(mimi).resolve()), 'pose_tokenizer': str(Path(pose_tokenizer).resolve()),
                         'text_tokenizer': str(Path(text_tokenizer).resolve()), 'device': 'cpu',
                         'audio_decode': 'one complete conversation per call, including prefill',
                         'pose_decode': '2D normalized offsets, confidence separate, planar z=0'}

    def decode(self, arrays):
        torch = self.torch
        frames = arrays['audio_codes'].shape[1]
        with torch.inference_mode():
            # A single full-row invocation preserves all causal convolution and
            # attention history. Passing only Transformer KV between chunks
            # would reset the convolution state and is deliberately avoided.
            codes = torch.from_numpy(arrays['audio_codes'].astype(np.int64, copy=False))[None]
            audio = self.mimi.decode(codes).audio_values[0, 0].float().numpy()
            if len(audio) < frames * 1920 or not np.isfinite(audio).all():
                raise ValueError('Mimi output does not cover the complete conversation')
            audio = audio[:frames * 1920].copy()
            pose = self.pose.decode([torch.from_numpy(row.astype(np.int64, copy=False))[None]
                                     for row in arrays['pose_codes']]).float()[0].numpy()
            if pose.shape != (2 * frames, 165):
                raise ValueError('Pose codec output does not cover the 25fps timeline')
        return audio, pose_plane(pose, self.parents), lane_transcript(arrays['text_ids'], self.text)


def publish_update(manifest_path, output_dir, codecs):
    import soundfile as sf
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    raw_bytes = manifest_path.read_bytes()
    source_sha = hashlib.sha256(raw_bytes).hexdigest()
    doc = json.loads(raw_bytes)
    if doc.get('schema_version') != 1 or doc.get('synthetic') is not False or len(doc.get('samples', [])) != 4:
        raise ValueError('A live rollout update must contain exactly four real samples')
    update = int(doc['update'])
    if update < 1 or len({item['id'] for item in doc['samples']}) != 4:
        raise ValueError('Invalid update or duplicate sample identifiers')
    public_path = output_dir / 'manifest.json'
    public = json.loads(public_path.read_text()) if public_path.exists() else {
        'schema_version': 1, 'synthetic': False, 'rollouts': []}
    existing = next((entry for entry in public['rollouts'] if entry['update'] == update), None)
    if existing and existing.get('source_manifest_sha256') == source_sha:
        return existing
    asset_root = output_dir / 'assets'
    asset_root.mkdir(parents=True, exist_ok=True)
    # Resume may legitimately regenerate this update. Keep old assets immutable
    # until the replacement manifest points to the new content-addressed set.
    final = asset_root / f'update-{update:06d}-{source_sha[:16]}'
    temporary = asset_root / f'.{final.name}.tmp'
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    samples = []
    try:
        for item in doc['samples']:
            arrays = read_sample(manifest_path.parent, item)
            audio, pose, text = codecs.decode(arrays)
            sample_id = item['id']
            user_name, generated_name, pose_name = (sample_id + suffix for suffix in ('-user.wav', '-generated.wav', '-pose.json'))
            sf.write(temporary / user_name, arrays['input_audio'], 24000, subtype='FLOAT')
            sf.write(temporary / generated_name, audio, 24000, subtype='FLOAT')
            atomic_json(temporary / pose_name, pose)
            base = f'assets/{final.name}/'
            samples.append({'id': sample_id, 'row_id': item['row_id'], 'orientation': item.get('orientation', ''),
                            'duration_seconds': item['frames'] / item['fps'], 'fps': item['fps'],
                            'prefill_seconds': item['prefill_frames'] / item['fps'],
                            'input_audio_url': base + user_name, 'output_audio_url': base + generated_name,
                            'pose_url': base + pose_name, 'video_url': None, 'text': text,
                            'pose_representation': '2D normalized skeleton (z=0), not 3D motion',
                            'input_audio_offset_seconds': 0, 'output_audio_offset_seconds': 0,
                            'source_audio': item.get('source_audio'), 'conditioning': item.get('conditioning')})
        entry = {'id': f'update-{update}', 'update': update, 'samples': samples,
                 'source_manifest_sha256': source_sha, 'codecs': codecs.metadata}
        atomic_json(temporary / 'decoded.json', entry)
        if final.exists():
            shutil.rmtree(final)  # Unpublished crash residue only; existing entries returned above.
        temporary.replace(final)
        public['rollouts'] = [row for row in public['rollouts'] if row['update'] != update]
        public['rollouts'].append(entry)
        public['rollouts'].sort(key=lambda value: value['update'], reverse=True)
        atomic_json(public_path, public)
        return entry
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mimi', type=Path, default=DEFAULT_MIMI)
    parser.add_argument('--pose-tokenizer', type=Path, default=DEFAULT_POSE)
    parser.add_argument('--text-tokenizer', type=Path, default=DEFAULT_TEXT)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=10)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error('--poll-seconds must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / '.decoder.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        codecs, finished = None, {}
        while True:
            for path in sorted(args.raw_root.glob('update-*/manifest.json')):
                fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
                if finished.get(path) == fingerprint:
                    continue
                started = time.monotonic()
                print(json.dumps({'event': 'decode_begin', 'manifest': str(path)}), flush=True)
                try:
                    if codecs is None:
                        codecs = CPUCodecs(args.mimi, args.pose_tokenizer, args.text_tokenizer, args.threads)
                    entry = publish_update(path, args.output_dir, codecs)
                    finished[path] = fingerprint
                    print(json.dumps({'event': 'decode_complete', 'update': entry['update'], 'samples': len(entry['samples']),
                                      'seconds': time.monotonic() - started}), flush=True)
                except Exception as error:
                    print(json.dumps({'event': 'decode_error', 'manifest': str(path),
                                      'error': f'{type(error).__name__}: {error}'}), flush=True)
                    if not args.watch:
                        raise
            if not args.watch:
                break
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
