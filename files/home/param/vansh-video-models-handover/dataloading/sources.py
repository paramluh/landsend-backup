import csv
import os
from typing import NamedTuple

import numpy as np
import torch
from safetensors import safe_open

from utils.common import is_main_process


EXPECTED_LATENT_CHANNELS = 48
MIN_LATENT_FRAMES = 19
AUDIO_CODE_QUANTUM = 750
NUM_REFS = 9


class ClipRecord(NamedTuple):
    latent_path: str
    pose_path: str
    mimi_path: str
    length: int


class ClipTensors(NamedTuple):
    latents: torch.Tensor
    pose_latents: torch.Tensor
    codes: torch.Tensor


def _clip_paths(root_dir, clip_rel, pose_suffix):
    clip_dir = os.path.join(root_dir, clip_rel)
    clip_id = os.path.basename(clip_rel.rstrip('/'))
    return (
        os.path.join(clip_dir, clip_id + '.safetensors'),
        os.path.join(clip_dir, clip_id + f'_pose_latents{pose_suffix}.safetensors'),
        os.path.join(clip_dir, clip_id + '_mimi.npy'),
    )


def _read_clip_rows(csv_path, pose_column):
    with open(csv_path, newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'paths', 'latent_t'}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f'Expected CSV with {sorted(required)} columns: {csv_path}')
        check_pose = pose_column in reader.fieldnames
        check_mimi = 'has_mimi' in reader.fieldnames
        return [
            (row['paths'].strip(), int(row['latent_t']))
            for row in reader
            if row.get('paths', '').strip()
            and (not check_pose or row[pose_column].strip().lower() in ('1', 'true'))
            and (not check_mimi or row['has_mimi'].strip().lower() in ('1', 'true'))
        ]


class PretrainClipSource:
    def __init__(self, root_dir, buckets, clipnames_root, pose_column, pose_version, shuffle_seed):
        pose_suffix = f'_{pose_version}' if pose_version else ''
        clips = []
        seen = set()
        for bucket, subbuckets in buckets.items():
            for subbucket in subbuckets:
                csv_path = os.path.join(clipnames_root, bucket, subbucket + '.csv')
                added = 0
                for rel, latent_t in _read_clip_rows(csv_path, pose_column):
                    # Mimi emits 2T-1 or 2T-2 codes, covering T-1 complete latent frames.
                    clip_len = latent_t - 1
                    paths = _clip_paths(root_dir, rel, pose_suffix)
                    key = os.path.normpath(paths[0])
                    if key in seen or clip_len < MIN_LATENT_FRAMES or 2 * clip_len > AUDIO_CODE_QUANTUM:
                        continue
                    seen.add(key)
                    clips.append(ClipRecord(*paths, clip_len))
                    added += 1
                if is_main_process():
                    print(f'PretrainClipSource: {bucket}/{subbucket}: {added} unique clips')

        rng = np.random.default_rng(shuffle_seed)
        self.clips = [clips[index] for index in rng.permutation(len(clips))]
        self.lengths = np.fromiter((clip.length for clip in self.clips), dtype=np.int64)
        self._cached = None

    def load(self, clip_index):
        if self._cached is not None and self._cached[0] == clip_index:
            return self._cached[1]
        clip = self.clips[clip_index]
        with safe_open(clip.latent_path, framework='pt') as handle:
            latents = handle.get_tensor('latents')[:, :clip.length].contiguous()
        if latents.shape[:2] != (EXPECTED_LATENT_CHANNELS, clip.length):
            raise RuntimeError(f'latent shape {tuple(latents.shape)} does not match manifest clip_len={clip.length}')
        with safe_open(clip.pose_path, framework='pt') as handle:
            pose_latents = handle.get_tensor('pose_latents')[:, :clip.length].contiguous()
        if pose_latents.shape[:2] != (EXPECTED_LATENT_CHANNELS, clip.length):
            raise RuntimeError(f'pose shape {tuple(pose_latents.shape)} does not match manifest clip_len={clip.length}')
        codes = torch.from_numpy(np.array(np.load(clip.mimi_path, mmap_mode='r'), copy=True))
        if codes.shape[1] < 2 * clip.length:
            raise RuntimeError(f'codes shape {tuple(codes.shape)} does not cover clip_len={clip.length}')
        source = ClipTensors(latents, pose_latents, codes)
        self._cached = clip_index, source
        return source
