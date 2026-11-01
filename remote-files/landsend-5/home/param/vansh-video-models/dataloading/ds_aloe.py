import csv
import os
import zlib
from typing import NamedTuple

import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open

from dataloading.ds_dense import SHUFFLE_SEED, DenseWindows
from dataloading.sources import AUDIO_CODE_QUANTUM, EXPECTED_LATENT_CHANNELS, NUM_REFS
from utils.common import is_main_process


REF_DROP_STREAM = 3
REF_PICK_STREAM = 4
EVAL_REF_SEED = 1234


class AloeRecord(NamedTuple):
    video_path: str
    pose_path: str
    mimi_path: str
    length: int
    uuid: str


def _csv_path(clipnames_root, name):
    path = os.path.join(clipnames_root, name)
    return path if path.endswith('.csv') else path + '.csv'


def _stable_id(text):
    return zlib.crc32(text.encode()) & 0x7FFFFFFF


def load_actor_refs(bank, uuid, like, rng):
    """-> [C, NUM_REFS, h, w], bool [NUM_REFS]; one uniform image per present property."""
    video = like.new_zeros(like.shape[0], NUM_REFS, *like.shape[2:])
    present = torch.zeros(NUM_REFS, dtype=torch.bool)
    for slot in range(NUM_REFS):
        path = os.path.join(bank, uuid, f'property_{slot + 1}', 'latents.safetensors')
        if not os.path.exists(path):
            continue
        with safe_open(path, framework='pt') as handle:
            images = handle.get_slice('video')
            index = int(rng.integers(images.get_shape()[0]))
            video[:, slot] = images[index:index + 1][0]
        present[slot] = True
    return video, present


class AloeChunkSource:
    """bank: actor bank dir; ref slot N of a chunk is one image of <bank>/<uuid>/property_<N>/latents.safetensors
    `video` [n, C, h, w], absent if the actor has no such file.
    chunk_ids / person_ids are hashes of the CSV paths / uuid, so they are comparable across node shards.
    bad_audio: chunks with crossbleed != 'no' or suspected_crossbleed == 'yes' get bad_audio True (their audio is
    always dropped in training)."""

    def __init__(self, root_dir, clipnames_root, csv_name, bank=None, *, pose_file, mimi_file, bad_audio=False):
        csv_path = _csv_path(clipnames_root, csv_name)
        clips = []
        chunk_ids = []
        bad = []
        self.bank = bank
        with open(csv_path, newline='') as handle:
            reader = csv.DictReader(handle)
            required = {'paths', 'latent_t', 'uuid'}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f'Expected CSV with {sorted(required)} columns: {csv_path}')
            for row in reader:
                rel = row['paths'].strip()
                if not rel:
                    continue
                clip_dir = os.path.join(root_dir, rel)
                clips.append(AloeRecord(
                    os.path.join(clip_dir, 'video_latents.safetensors'),
                    os.path.join(clip_dir, pose_file),
                    os.path.join(clip_dir, mimi_file),
                    int(row['latent_t']) - 1,
                    row['uuid'].strip(),
                ))
                chunk_ids.append(_stable_id(rel))
                if bad_audio:
                    bad.append(row['crossbleed'].strip() != 'no' or row['suspected_crossbleed'].strip() == 'yes')
        self.clips = clips
        self.bad_audio = np.array(bad, dtype=bool) if bad_audio else None
        self.lengths = np.fromiter((clip.length for clip in clips), dtype=np.int64)
        people = {clip.uuid for clip in clips}
        self.chunk_ids = np.array(chunk_ids, dtype=np.int32)
        self.person_ids = np.fromiter((_stable_id(clip.uuid) for clip in clips), dtype=np.int32, count=len(clips))
        if is_main_process():
            print(f'AloeChunkSource: {csv_name}: {len(clips)} chunks, {len(people)} people')

    def load_refs(self, clip_index, like, rng):
        """-> video [C, NUM_REFS, h, w], present bool [NUM_REFS], one uniform pick (by rng) per property; `like` [C, 1, h, w]."""
        return load_actor_refs(self.bank, self.clips[clip_index].uuid, like, rng)

    def load_eval_refs(self, clip_index, like):
        return self.load_refs(clip_index, like, np.random.default_rng([EVAL_REF_SEED, int(self.chunk_ids[clip_index])]))

    def _slice_latents(self, path, start, end):
        with safe_open(path, framework='pt') as handle:
            frames = handle.get_slice('latents')[start:end]
        if frames.ndim != 4 or frames.shape[0] != end - start or frames.shape[1] != EXPECTED_LATENT_CHANNELS:
            raise RuntimeError(f'latent slice {tuple(frames.shape)} from {path} [{start}:{end}]')
        return frames.permute(1, 0, 2, 3).contiguous()

    def load_window(self, clip_index, ofi):
        clip = self.clips[clip_index]
        real = ofi[ofi >= 1]
        body_start = int(real.min())
        body_end = int(real.max()) + 1
        frame0 = self._slice_latents(clip.video_path, 0, 1)
        pose0 = self._slice_latents(clip.pose_path, 0, 1)
        body = self._slice_latents(clip.video_path, body_start, body_end)
        pose_body = self._slice_latents(clip.pose_path, body_start, body_end)
        with safe_open(clip.mimi_path, framework='pt') as handle:
            codes = handle.get_tensor('codes').long()
        f_last = int(real.max())
        # Shifted frames must stay <= 374: frame f reads tokens 4(f-1)..4f-1 of 1500 - drop = 1498.
        a = max(0, f_last - (AUDIO_CODE_QUANTUM // 2 - 1))
        codes = codes[:, 2 * a:2 * a + AUDIO_CODE_QUANTUM]
        shifted = ofi.clone()
        shifted[ofi >= 1] = ofi[ofi >= 1] - a
        return frame0, body, pose0, pose_body, body_start, codes, shifted, a

    def load_prefix(self, clip_index, frames):
        clip = self.clips[clip_index]
        latents = self._slice_latents(clip.video_path, 0, frames)
        pose = self._slice_latents(clip.pose_path, 0, frames)
        with safe_open(clip.mimi_path, framework='pt') as handle:
            codes = handle.get_slice('codes')[:, :AUDIO_CODE_QUANTUM].long()
        return latents, pose, codes


class PretrainChunkSource:
    """
    A PretrainClipSource behind the AloeChunkSource window interface. Every clip has 2 * length <= AUDIO_CODE_QUANTUM,
    so its last frame is <= AUDIO_CODE_QUANTUM // 2 - 1 and the code shift a is 0. No clip has bad audio.

    latents, pose_latents: [C, length, h, w]
    codes: [num_codebooks, >= 2 * length], zero-padded to [num_codebooks, AUDIO_CODE_QUANTUM]
    """
    bank = None

    def __init__(self, source, root_dir):
        self.source, self.clips, self.lengths = source, source.clips, source.lengths
        self.bad_audio = np.zeros(len(self.clips), dtype=bool)
        rels = [os.path.relpath(os.path.dirname(clip.latent_path), root_dir) for clip in self.clips]
        self.chunk_ids = np.fromiter((_stable_id(rel) for rel in rels), dtype=np.int32, count=len(rels))
        self.person_ids = np.fromiter((_stable_id(os.path.basename(rel).split('_')[0]) for rel in rels), dtype=np.int32, count=len(rels))

    def load_window(self, clip_index, ofi):
        clip = self.source.load(clip_index)
        real = ofi[ofi >= 1]
        body_start, body_end = int(real.min()), int(real.max()) + 1
        length = min(clip.codes.shape[1], AUDIO_CODE_QUANTUM)
        codes = torch.zeros(clip.codes.shape[0], AUDIO_CODE_QUANTUM, dtype=torch.long)
        codes[:, :length] = clip.codes[:, :length]
        latents, pose = clip.latents, clip.pose_latents
        return latents[:, :1], latents[:, body_start:body_end], pose[:, :1], pose[:, body_start:body_end], body_start, codes, ofi.clone(), 0


class ShuffledDenseWindows:
    """ref_drop = (counts, p): a window drops counts[i] of its NUM_REFS ref slots with probability p[i]; missing
    refs count towards that number, the rest are drawn uniformly from the present refs. Seeded per
    (SHUFFLE_SEED, epoch, window).
    min_over: a process group over which windows/epoch is reduced to its minimum, so per-node sources of
    different sizes (rank, world_size local to the node) run the same number of steps.
    windows_per_epoch: caps windows/epoch, so each epoch's order is a random subset of that size."""

    def __init__(self, source, num_history, num_present, history_dropout, rank, world_size, batch_size,
                 accumulation_steps, ref_drop=None, min_over=None, windows_per_epoch=None):
        self.source = source
        self.num_history = num_history
        self.num_present = num_present
        assert len(history_dropout) == num_history and sum(history_dropout) <= 1, history_dropout
        self._drop_cdf = np.cumsum(history_dropout)
        self._ref_drop = None
        if ref_drop is not None:
            counts, p = ref_drop
            assert len(counts) == len(p) and abs(sum(p) - 1) < 1e-6 and max(counts) <= NUM_REFS, ref_drop
            assert source.bank is not None
            self._ref_drop = np.array(counts), np.cumsum(p)
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        window_counts = (source.lengths - 1) // num_present
        self._prefix = np.zeros(len(window_counts) + 1, dtype=np.int64)
        np.cumsum(window_counts, out=self._prefix[1:])
        quantum = batch_size * world_size * accumulation_steps
        total = int(self._prefix[-1]) if windows_per_epoch is None else min(int(self._prefix[-1]), windows_per_epoch)
        self._windows_per_epoch = total // quantum * quantum
        if min_over is not None:
            count = torch.tensor(self._windows_per_epoch)
            dist.all_reduce(count, op=dist.ReduceOp.MIN, group=min_over)
            self._windows_per_epoch = int(count)
        if not self._windows_per_epoch:
            raise RuntimeError('Shuffled dense window stream is smaller than one global accumulation step')
        self._perm_epoch = None
        self._perm = None
        if is_main_process():
            print(
                f'ShuffledDenseWindows: {len(source.clips)} chunks, {self._windows_per_epoch} windows/epoch, '
                f'{len(self)} microbatches/epoch'
            )

    def __len__(self):
        return self._windows_per_epoch // (self.batch_size * self.world_size)

    def _order(self, epoch):
        if self._perm_epoch == epoch:
            return self._perm
        n = int(self._prefix[-1])
        rng = np.random.default_rng(np.random.SeedSequence([SHUFFLE_SEED, epoch]))
        perm = rng.permutation(n).astype(np.int32)
        self._perm = perm[:self._windows_per_epoch]
        self._perm_epoch = epoch
        return self._perm

    def _window(self, epoch, window_index):
        return DenseWindows._window(self, epoch, window_index)

    @staticmethod
    def _gather(frame0, body, body_start, ofi):
        src = torch.cat([frame0, body], dim=1)
        idx = ofi.clone()
        idx[ofi >= 1] = ofi[ofi >= 1] - body_start + 1
        return DenseWindows._gather(src, idx)

    def _ref_keep(self, epoch, window_index, present):
        counts, cdf = self._ref_drop
        rng = np.random.default_rng(np.random.SeedSequence([SHUFFLE_SEED, epoch, window_index, REF_DROP_STREAM]))
        count = int(counts[min(int(np.searchsorted(cdf, rng.random(), side='right')), len(counts) - 1)])
        available = np.flatnonzero(present.numpy())
        n = min(len(available), max(0, count - (NUM_REFS - len(available))))
        keep = present.clone()
        keep[torch.from_numpy(rng.choice(available, n, replace=False))] = False
        return keep

    def __getitem__(self, index):
        epoch, microbatch = divmod(index, len(self))
        first = (microbatch * self.world_size + self.rank) * self.batch_size
        order = self._order(epoch)
        window_ids = [int(order[first + offset]) for offset in range(self.batch_size)]
        windows = [self._window(epoch, window_id) for window_id in window_ids]
        loaded = [self.source.load_window(clip, ofi) for clip, ofi in windows]
        ofi = torch.stack([item[6] for item in loaded])
        unshifted = [w for _, w in windows]
        codes = torch.stack([item[5] for item in loaded])
        batch = {
            'latents_low': torch.stack([self._gather(item[0], item[1], item[4], u) for item, u in zip(loaded, unshifted)]),
            'pose_latents_low': torch.stack([self._gather(item[2], item[3], item[4], u) for item, u in zip(loaded, unshifted)]),
            'original_frame_indices': ofi,
            'present_valid': ofi[:, 1 + self.num_history:] >= 0,
            'window_source': torch.arange(self.batch_size),
            'codes': codes,
            'chunk_index': torch.tensor([self.source.chunk_ids[clip] for clip, _ in windows], dtype=torch.int32),
            'person_index': torch.tensor([self.source.person_ids[clip] for clip, _ in windows], dtype=torch.int32),
        }
        if self.source.bad_audio is not None:
            batch['bad_audio'] = torch.from_numpy(self.source.bad_audio[[clip for clip, _ in windows]])
        if self._ref_drop is not None:
            refs = [
                self.source.load_refs(clip, item[0], np.random.default_rng(
                    np.random.SeedSequence([SHUFFLE_SEED, epoch, w, REF_PICK_STREAM])))
                for w, (clip, _), item in zip(window_ids, windows, loaded)
            ]
            present = torch.stack([ref[1] for ref in refs])
            batch['ref_latents'] = torch.stack([ref[0] for ref in refs])
            batch['ref_keep'] = torch.stack([self._ref_keep(epoch, w, p) for w, p in zip(window_ids, present)])
        return batch


class AloeEvalClips:
    """Rows past the last clip wrap around with valid=False, so every clip is scored exactly once at any world size.
    With refs, each clip also carries donor_ref_* = the refs of the next clip (cyclically) of a different person.
    Eval refs are a fixed pick per clip, so they match across steps."""

    def __init__(self, source, num_history, num_present, rank, world_size, batch_size):
        self.source = source
        self.prefix = 1 + num_history + num_present
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        if source.bank is not None:
            people, n = source.person_ids, len(source.clips)
            self.donors = [next((i + k) % n for k in range(1, n) if people[(i + k) % n] != people[i]) for i in range(n)]
        if is_main_process():
            print(f'AloeEvalClips: {len(source.clips)} chunks, {len(self)} microbatches')

    def __len__(self):
        return -(-len(self.source.clips) // (self.batch_size * self.world_size))

    def __getitem__(self, index):
        first = (index * self.world_size + self.rank) * self.batch_size
        n = len(self.source.clips)
        rows = range(first, first + self.batch_size)
        clips = [row % n for row in rows]
        items = [self.source.load_prefix(clip, self.prefix) for clip in clips]
        codes = torch.zeros(self.batch_size, items[0][2].shape[0], AUDIO_CODE_QUANTUM, dtype=torch.long)
        for row, (_, _, item_codes) in enumerate(items):
            length = min(item_codes.shape[1], AUDIO_CODE_QUANTUM)
            codes[row, :, :length] = item_codes[:, :length]
        batch = {
            'latents_low': torch.stack([latents for latents, _, _ in items]),
            'pose_latents_low': torch.stack([pose for _, pose, _ in items]),
            'source_id': torch.tensor(clips),
            'codes': codes,
            'valid': torch.tensor([row < n for row in rows]),
        }
        if self.source.bank is not None:
            refs = [self.source.load_eval_refs(clip, latents[:, :1]) for clip, (latents, _, _) in zip(clips, items)]
            batch['ref_latents'] = torch.stack([ref[0] for ref in refs])
            batch['ref_keep'] = torch.stack([ref[1] for ref in refs])
            donors = [self.source.load_eval_refs(self.donors[clip], latents[:, :1]) for clip, (latents, _, _) in zip(clips, items)]
            batch['donor_ref_latents'] = torch.stack([ref[0] for ref in donors])
            batch['donor_ref_keep'] = torch.stack([ref[1] for ref in donors])
        return batch
