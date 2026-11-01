"""
cached_universal.py's sampler on studio chunks read straight from the dataset csv. The sink's K/V changes per clip,
so the pads, which attend the sink, are recomputed from it in start_clip; everything else is cached_universal.py's.

With [refs] (a model trained with num_refs): the window is [R refs | sink | H history | P present], the sink and
the window frames at RoPE positions WINDOW_OFFSET.., the refs at position 0. A ref is context-free like the sink (it
attends only itself, t = 0, no pose, ref_patch_embedding, the pad token when missing), so each ref's K/V is one
solo pass per clip into its own pool entry, and every other frame reads all refs before the sink. The refs are
eval's pick (AloeChunkSource.load_eval_refs) from refs.bank, one bank split dir of <uuid>/property_<N>/.

Clips: `rows` (indices) of `csv` under `root_dir`, each `<paths>/` with video_latents.safetensors and the
`pose_file` (key latents, bf16 [L, 48, h, w]) and mimi.safetensors (key codes, int16 [codebooks, 2L]). Clip n is
csv row n, seed = seed + n. Relative paths resolve against this directory.
With [hidden] (a model trained with Orpheus hiddens): llm_hiddens_root mirrors root_dir, and each clip's precomputed
<paths>/llm_hiddens.safetensors (teacher-forced on the ground-truth codes) conditions its present frames.
Output: <output_dir with {step}>/samples_<n>.mp4 (ground truth), sample_<n>.mp4, sample_<n>_sxs.mp4 (both with the
Mimi-decoded audio) and sample_<n>_latents.safetensors.
"""

import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cached_universal import SCRATCH, SINK, CachedNet, generate
from dataloading.ds_aloe import AloeChunkSource
from dataloading.sources import AUDIO_CODE_QUANTUM
from inference.common import FPS, decode_wan22_vae, resolve_path, run_parallel, side_by_side
from inference.loader import build_models

TAG = 'sink'
MIMI_FILE = 'mimi.safetensors'


def load_chunk(clip_dir, pose_file, max_frames, mimi_file=MIMI_FILE):
    """common.load_sample(..., slide_audio=True) on a studio chunk."""
    latents = load_file(f'{clip_dir}/video_latents.safetensors')['latents'].permute(1, 0, 2, 3)
    pose = load_file(f'{clip_dir}/{pose_file}')['latents'].permute(1, 0, 2, 3)
    codes = load_file(f'{clip_dir}/{mimi_file}')['codes'].long()
    L = min(latents.shape[1], pose.shape[1], codes.shape[1] // 2)
    table = torch.zeros(1, codes.shape[0], AUDIO_CODE_QUANTUM, dtype=torch.long)
    length = min(codes.shape[1], AUDIO_CODE_QUANTUM)
    table[0, :, :length] = codes[:, :length]
    frames = min(L, max_frames or L)
    return latents[:, :frames].float(), pose[:, :frames].float(), table, codes[:, :2 * L]


class SinkCachedNet(CachedNet):
    refs = None  # (ref_latents [B, R, C, h, w], ref_keep [B, R]), set per batch

    @torch.no_grad()
    def start_clip(self, sink, cuda_graph):
        H, P, device = self.H, self.P, self.device
        idx = lambda values: torch.tensor(list(values), dtype=torch.long, device=device) + self.offsets
        self.audio_cache.clear()
        if self.R:
            latents, keep = self.refs
            _, ks, vs = self._frame(latents.flatten(0, 1), None, ~keep.flatten(), self.zero_t, self.ref_freqs, None,
                                    None, None, head=False, embedding=self.initial.ref_patch_embedding)
            entries = idx(self.fixed[:-1]).flatten()
            self.pool_k.index_copy_(1, entries, ks.flatten(1, 2))
            self.pool_v.index_copy_(1, entries, vs)
        _, ks, vs = self._frame(sink, None, False, self.zero_t, self.slot_freqs[:1], None, None, None, head=False)
        self._store(idx([SINK]), ks, vs)
        pad_history = range(self.pad_history0, self.pad_history0 + H)
        _, ks, vs = self._frame(sink, None, True, self.zero_t, self.anchor_freqs, self.sink_ctx, self.history_freqs,
                                None, head=False)
        self._store(idx(pad_history), ks, vs)
        for j in range(P - 1):
            ctx = [*self.fixed, *pad_history[j:], *range(self.pad_present0 + max(0, j - H), self.pad_present0 + j)]
            _, ks, vs = self._frame(sink, None, True, self.zero_t, self.present_freqs[j:j + 1], idx(ctx), None, None,
                                    head=False)
            self._store(idx([self.pad_present0 + j]), ks, vs)
        if cuda_graph and self.graphs is None:
            self.buf_ints.copy_(torch.tensor([SINK] * self.num_ctx + [SCRATCH, 0, 0], device=device))
            self.buf_store.fill_(SCRATCH)
            start = time.time()
            self.graphs = [self._capture(self._step), self._capture(self._commit_step)]
            if self.guidance_scale != 1.0:
                self.graphs.append(self._capture(self._guided_step))
            print(f'[{TAG}] {len(self.graphs)} graphs captured in {time.time() - start:.0f}s', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='sink_universal.yaml')
    parser.add_argument('--parallel', action='store_true', help='one worker per visible GPU')
    parser.add_argument('--shard', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--num-shards', type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    config_path = base_dir / args.config
    config = yaml.safe_load(open(config_path))
    output_dir = Path(resolve_path(config['output_dir'].format(step=config['step']), base_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.shard in (None, 0):
        shutil.copy2(config_path, output_dir / config_path.name)

    csv_path = resolve_path(config['csv'], base_dir)
    rows = list(csv.DictReader(open(csv_path)))
    nums = config['rows']
    if args.parallel:
        print(f'[{TAG}] rows {nums[0]}..{nums[-1]} of {config["csv"]}')
        return run_parallel(__file__, args.config, len(nums))
    tag = TAG if args.shard is None else f'{TAG}:w{args.shard}'
    if args.shard is not None:
        nums = nums[args.shard::args.num_shards]

    if config['type'] != 'universal_wide':
        raise ValueError(f"{config['type']}: caching assumes universal_wide's context-free history")
    device = 'cuda'
    models = build_models(config, base_dir, device)
    substeps, num_steps = config['substeps'], config['num_steps']
    if num_steps != substeps * models.num_present:
        raise ValueError(f'num_steps must be substeps x num_present: {num_steps} != {substeps} x {models.num_present}')
    guidance_scale, guidance_steps = config['audio_guidance_scale'], config['audio_guidance_steps']
    root_dir = resolve_path(config['root_dir'], base_dir)
    source = None
    if 'refs' in config:  # load_eval_refs seeds on the hash of the csv paths, so the pick is eval's
        source = AloeChunkSource(root_dir, os.path.dirname(csv_path), os.path.basename(csv_path),
                                 bank=resolve_path(config['refs']['bank'], base_dir), pose_file=config['pose_file'],
                                 mimi_file=MIMI_FILE)
    batch = config.get('batch_size', 1)
    cached = None
    print(f'[{tag}] {len(nums)} clips, batch {batch}, {num_steps} steps per frame ({substeps} per band), '
          f'seed {config["seed"]}, {models.net.layers[0].num_refs} refs, '
          f'audio guidance {guidance_scale} on the first '
          f'{guidance_steps} steps, compile {config["compile"]}, cuda_graph {config["cuda_graph"]}')

    for i in range(0, len(nums), batch):
        group, clips, audios, refs, hiddens = nums[i:i + batch], [], [], [], []
        for n in group:
            latents, pose, code_table, codes = load_chunk(
                os.path.join(root_dir, rows[n]['paths']), config['pose_file'], config['max_latent_frames'])
            if 'llm_hiddens_root' in config:
                hiddens.append(load_file(os.path.join(resolve_path(config['llm_hiddens_root'], base_dir),
                                                      rows[n]['paths'], 'llm_hiddens.safetensors'))['hiddens'])
            if source is not None:
                ref_latents, ref_keep = source.load_eval_refs(n, latents[:, :1])
                refs.append((ref_latents.transpose(0, 1), ref_keep))
                print(f'[{tag}] row {n} {rows[n]["uuid"]}: refs {ref_keep.int().tolist()}')
            if cached is None:
                cached = SinkCachedNet(models, latents.shape[2:], substeps, device, config['compile'], guidance_scale,
                                       batch)
            audio = models.mimi.decode(codes[None].to(device)).audio_values[0, 0].float().cpu().numpy()
            gt_path = output_dir / f'samples_{n}.mp4'
            if not gt_path.exists():
                decode_wan22_vae(models.vae, latents[None], str(gt_path), fps=FPS, audio_np=audio, tag=tag)
            clips.append((latents, pose, code_table, codes))
            audios.append(audio)
        if refs:  # generate fills a short batch with copies of its last clip
            refs += refs[-1:] * (batch - len(group))
            cached.refs = tuple(torch.stack(part).to(device) for part in zip(*refs))
        if hiddens:
            cached.hiddens = [h.to(device) for h in hiddens + hiddens[-1:] * (batch - len(group))]
        outputs = generate(cached, clips, [config['seed'] + n for n in group], device, config['cuda_graph'],
                           guidance_steps)
        for n, (latents, _, _, _), audio, generated in zip(group, clips, audios, outputs):
            print(f'[{tag}] row {n} {rows[n]["uuid"]}: {latents.shape[1]} conditioned frames -> '
                  f'{generated.shape[1]} latent frames')
            gt_path, out_path = output_dir / f'samples_{n}.mp4', output_dir / f'sample_{n}.mp4'
            save_file({'latents': generated.cpu()}, str(output_dir / f'sample_{n}_latents.safetensors'))
            decode_wan22_vae(models.vae, generated[None], str(out_path), fps=FPS, audio_np=audio, tag=tag)
            side_by_side(gt_path, out_path, output_dir / f'sample_{n}_sxs.mp4', tag=tag)


if __name__ == '__main__':
    main()
