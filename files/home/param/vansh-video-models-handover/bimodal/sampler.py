"""
One-stage, interleaved, open-loop rollouts of a bimodal model; the sink is GT frame 0. One loop over frames advances
the stabilising wavefront: present band b of P covers (t_h + (1 - t_h) b/P, t_h + (1 - t_h)(b+1)/P],
t_h = history_t, in `substeps` Euler steps. A frame leaving it at t_h is committed as is, as history at t_h, so the
stabilising mode only reads its own frames, and is resolved at once: renoised to t_start,
x = a x_f + sigma eps_f (a = (1 - t_start)/(1 - t_h), sigma^2 = t_start^2 - (a t_h)^2), then num_steps Euler
steps t_start -> 0 against [sink | resolved f-H..f-1]. One checkpoint_path in both blocks loads its weights once.
wavefront: false runs the stabilising mode frame -> band -> substep (cached_universal.generate, one frame per pass);
true runs it as rounds (generate_wavefront, the P present frames in one pass per substep): the same t, slots, context
and audio table for every (frame, band, substep), so the same trajectories up to GEMM shapes.
Each clip's codes come from <paths>/<mimi_file>, the encoding the checkpoint was trained on (mimi_local.safetensors:
Mimi's encoder attention in its 250-step window, as the pretrain-ds codes; mimi.safetensors: over the whole chunk).
Output per clip n (csv row n): sample_<n>_latents.safetensors (GT frame 0, then the resolved frames),
sample_<n>_stabilised_latents.safetensors (GT frame 0, then the committed frames), samples_<n>.mp4 (GT),
sample_<n>.mp4, sample_<n>_stabilised.mp4 and sample_<n>_sxs.mp4 (GT | resolved), with the Mimi audio.
Batches are the rows in groups of batch_size; --shard k --num-shards n takes the k-th of n contiguous chunks of them.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from flash_attn_interface import flash_attn_func
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / 'production')]

from bimodal.cached_resolver import ResolverNet
from bimodal.audio_tables import resolver_tokens
from cached_universal import SCRATCH, SINK, audio_start, generate
from inference.common import FPS, decode_wan22_vae, sliding_code_table
from inference.loader import build_models
from inference.lora import project_qkv
from models.pose import rope
from models.pose.audio import audio_latents_from_codes
from models.wan.model import sinusoidal_embedding_1d
from sink_universal import SinkCachedNet, load_chunk
from utils.common import AUTOCAST_DTYPE

TAG = 'bimodal'
MODE_KEYS = ('checkpoint_path', 'num_history', 'num_present')
RESOLVE_SEED_STRIDE = 1_000_003


def side_by_side(left_path, right_path, output_path):
    """
    common.side_by_side without +faststart: ffmpeg 6.1.1 writes a second moov with this filter graph.
    """
    cmd = [
        'ffmpeg', '-y', '-i', str(left_path), '-i', str(right_path),
        '-filter_complex', '[0:v][1:v]hstack=inputs=2:shortest=1[v]', '-map', '[v]', '-map', '1:a?',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-shortest',
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


class StabiliserCachedNet(SinkCachedNet):
    def __init__(self, models, latent_hw, substeps, device, compile, batch, history_t):
        super().__init__(models, latent_hw, substeps, device, compile, 1.0, batch)
        P, m = self.P, self.m
        u = torch.tensor([(b + 1) / P - s / (P * m) for b in range(P) for s in range(m)], dtype=torch.float32)
        self.t_band = history_t + (1 - history_t) * u  # [P * m] fp32, band-major
        self.t_table = (self.t_band * 1000).to(device)
        self.dt = -(1 - history_t) / (P * m)
        self.history_t = (torch.full((1, 1), history_t) * 1000).to(device)

    def _commit(self):
        return self._frame(self.buf_latent, self.buf_pose, False, self.history_t, self.anchor_freqs, self.sink_ctx,
                           self.history_freqs, None, head=False)


class WavefrontNet(StabiliserCachedNet):
    """
    Round r holds frame r + j at band j in present slot j; a frame below 1 (the pad) or past the clip is invalid: the
    pad token at t = 0, no pose, no audio, as training's present pads. Each substep is one pass of the P present
    frames against the pool's [sink | history slots 1..H], slot j reading the H slots before it (history, then this
    pass's slots < j) and itself, TransformerLayer's present rule. A frame's left neighbours at (band b - i, substep
    s) are in the same pass, so the pool's present ring is unused; the sink, pads, history ring and commit are
    StabiliserCachedNet's. References use SinkCachedNet.refs: ([B, R, C, h, w], [B, R]); their per-layer K/V is
    refreshed once per clip, not once per step.

    With guidance_scale != 1, conditional and whole-window audio-dropout passes share one batched forward and
    read the same fixed/history cache. The learned no_audio_tokens replace every present audio token in the
    dropout branch, leaving pose and references unchanged. Only slot P-1, the first/highest-noise denoising band,
    uses v_null + guidance_scale * (v_cond - v_null); later bands use the conditional velocity unchanged.
    Guidance neither normalizes to unit length nor rescales to the conditioned velocity norm.
    Optional telemetry records per-clip FP32 spatial/channel norms without changing the Euler update.
    """

    guidance_metric_names = ('conditioned_norm', 'null_norm', 'guided_norm', 'delta_norm',
                             'conditioned_guided_cosine')

    def __init__(self, models, latent_hw, substeps, device, compile, batch, history_t, guidance_scale=1.0,
                 guidance_telemetry=False):
        super().__init__(models, latent_hw, substeps, device, compile, batch, history_t)
        H, P, C = self.H, self.P, self.initial.patch_embedding.in_channels
        assert self.buf_hidden is None, 'wavefront sampling does not support hidden conditioning'
        self.guidance_scale = float(guidance_scale)
        if not torch.isfinite(torch.tensor(self.guidance_scale)) or self.guidance_scale < 0:
            raise ValueError('audio guidance scale must be finite and nonnegative')
        if guidance_telemetry and self.guidance_scale == 1.0:
            raise ValueError('guidance telemetry requires a guidance scale other than 1')
        self.guidance_metrics = (torch.zeros(batch, len(self.guidance_metric_names), dtype=torch.float32, device=device)
                                 if guidance_telemetry else None)
        self.guidance_trace = None
        self.guidance_trace_timesteps = None
        self.guidance_trace_count = 0
        self.wave_latent = torch.zeros(batch, C, P, *latent_hw, device=device)  # present slots, Euler-updated in place
        self.wave_pose = torch.zeros(batch, C, P, *latent_hw, device=device)
        self.wave_valid = torch.zeros(P, dtype=torch.bool, device=device)
        self.wave_ints = torch.zeros(self.num_ctx + 1, dtype=torch.long, device=device)
        self.wave_audio = torch.zeros(batch, P, self.K + 1, self.initial.dim, dtype=AUTOCAST_DTYPE, device=device)
        self.slot_steps = torch.arange(P, device=device) * self.m  # slot j's t: t_table[j m + s]
        self.wave_freqs = self.present_freqs.flatten(0, 1)  # [P N, 1, c, 2]
        fixed_count = self.R + 1
        self.key_slots = torch.tensor([
            [*range(fixed_count), *range(fixed_count + slot, fixed_count + H + slot + 1)]
            for slot in range(P)
        ], device=device)
        self.waving = torch.compile(self._wave, dynamic=False) if compile else self._wave

    def _wave(self):
        """
        One substep of the P present frames, a pure function of the static buffers (compiled).

        -> velocity [B, C, P, h, w]
        """
        initial, N, n, d, B, H, P = self.initial, self.N, self.n, self.d, self.B, self.H, self.P
        valid = self.wave_valid[None, :, None, None]
        ctx = (self.wave_ints[None, :self.num_ctx] + self.offsets).flatten()
        latent, pose, audio = self.wave_latent, self.wave_pose, self.wave_audio
        if self.guidance_scale != 1.0:
            latent = torch.cat((latent, latent))
            pose = torch.cat((pose, pose))
            audio = torch.cat((audio, initial.audio_encoder.no_audio_tokens.to(audio.dtype).expand_as(audio)))
            ctx = ctx.repeat(2)
            B *= 2
        t = (self.t_table.index_select(0, self.slot_steps + self.wave_ints[self.num_ctx]) * self.wave_valid).view(1, P)
        with torch.autocast('cuda', dtype=AUTOCAST_DTYPE, cache_enabled=False):
            x = initial.patch_embedding(latent).flatten(2).transpose(1, 2).unflatten(1, (P, N))
            x = torch.where(valid, x, initial.pad_token.to(x.dtype))
            pose = initial.pose_patch_embedding(pose.to(x.dtype)).flatten(2).transpose(1, 2)
            x = x + pose.unflatten(1, (P, N)) * valid
            e = initial.time_embedding(sinusoidal_embedding_1d(initial.freq_dim, t.flatten()).unflatten(0, t.shape))
            e0 = initial.time_projection(e).unflatten(2, (6, initial.dim))

            for i, layer in enumerate(self.blocks):
                block, attention = layer.block, layer.block.self_attn
                modulation = (block.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
                normalized = (block.norm1(x) * (1 + modulation[1]) + modulation[0]).flatten(1, 2)
                q, k, v = project_qkv(attention, normalized)
                q = rope.apply_rope(attention.norm_q(q).view(B, P * N, n, d), self.wave_freqs)
                k = rope.apply_rope(attention.norm_k(k).view(B, P * N, n, d), self.wave_freqs)
                v = v.view(B, P, N, n, d).bfloat16()
                keys = torch.cat((self.pool_k[i].index_select(0, ctx).view(B, self.num_ctx, N, n, d),
                                  k.bfloat16().view(B, P, N, n, d)), dim=1)
                values = torch.cat((self.pool_v[i].index_select(0, ctx).view(B, self.num_ctx, N, n, d), v), dim=1)
                keys, values = (z[:, self.key_slots].flatten(2, 3).flatten(0, 1) for z in (keys, values))
                output = flash_attn_func(q.bfloat16().view(B * P, N, n, d), keys, values).to(q.dtype)
                x = x + attention.o(output.view(B, P, N, n * d)) * modulation[2]

                if layer.audio_block is not None:
                    x = x + layer.audio_block.forward_present(x, audio) * valid

                residual = block.ffn(block.norm2(x) * (1 + modulation[4]) + modulation[3])
                x = x + residual * modulation[5]

            final = self.final.head
            with torch.autocast('cuda', dtype=torch.float32, cache_enabled=False):
                modulation = (final.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
                x = final.head(final.norm(x) * (1 + modulation[1]) + modulation[0])
        velocity = self.final.unpatchify(x.flatten(1, 2), (P, *self.grid[1:])).float()
        if self.guidance_scale != 1.0:
            conditioned, unconditioned = velocity.chunk(2)
            first = unconditioned[:, :, -1:] + self.guidance_scale * (
                conditioned[:, :, -1:] - unconditioned[:, :, -1:])
            if self.guidance_metrics is not None:
                conditioned_first, null_first = conditioned[:, :, -1:], unconditioned[:, :, -1:]
                dimensions = (1, 2, 3, 4)
                conditioned_norm = torch.linalg.vector_norm(conditioned_first, dim=dimensions)
                null_norm = torch.linalg.vector_norm(null_first, dim=dimensions)
                guided_norm = torch.linalg.vector_norm(first, dim=dimensions)
                delta_norm = torch.linalg.vector_norm(first - conditioned_first, dim=dimensions)
                cosine = (conditioned_first * first).sum(dim=dimensions) / (
                    conditioned_norm * guided_norm).clamp_min(1e-12)
                self.guidance_metrics.copy_(torch.stack(
                    (conditioned_norm, null_norm, guided_norm, delta_norm, cosine), dim=1))
            velocity = torch.cat((conditioned[:, :, :-1], first), dim=2)
        return velocity

    def _wave_step(self):
        self.wave_latent.copy_(self.wave_latent + self.waving() * self.dt)

    def step(self, guided=False):
        if self.graphs is None:
            return self._wave_step()
        self.graphs[0].replay()

    @torch.no_grad()
    def start_clip(self, sink, cuda_graph):
        super().start_clip(sink, False)
        if cuda_graph and self.graphs is None:
            self.wave_ints.copy_(torch.tensor([SINK] * self.num_ctx + [0], device=self.device))
            self.buf_store.fill_(SCRATCH)
            start = time.time()
            self.graphs = [self._capture(self._wave_step), self._capture(self._commit_step)]
            print(f'[{TAG}] wavefront and commit graphs captured in {time.time() - start:.0f}s', flush=True)

    @torch.no_grad()
    def set_wave_audio(self, code_tables, codes, f_last, frames):
        """
        The encoded table of the round's window (last real frame f_last; cached per table start as set_audio) and
        each slot's K tokens + the null token -> wave_audio. frames [P]: the slots' frames, clamped into the clip.
        """
        a = (audio_start(f_last),) * self.B
        if a not in self.audio_cache:
            for old in [key for key in self.audio_cache if key[0] < a[0] - self.P]:
                del self.audio_cache[old]
            tables = [sliding_code_table(table, c, f_last)[0] for table, c in zip(code_tables, codes)]
            latents = audio_latents_from_codes(self.mimi, torch.cat(tables), self.num_codebooks)
            with torch.autocast('cuda', dtype=AUTOCAST_DTYPE):
                self.audio_cache[a] = self.initial.audio_encoder(latents)
        positions = self.K * (frames[:, None] - a[0] - 1).clamp(min=0) + self.token_offsets  # [P, K]
        null = self.initial.audio_encoder.null_token.to(AUTOCAST_DTYPE).expand(self.B, self.P, 1, -1)
        self.wave_audio.copy_(torch.cat((self.audio_cache[a][:, positions], null), dim=2))


@torch.no_grad()
def generate_wavefront(cached, clips, seeds, device, cuda_graph, on_commit=None):
    """
    generate's contract for a WavefrontNet, with one length per batch (a round advances every clip). Round r of
    2 - P .. L - 1: frame r + P - 1 enters slot P - 1 from its noise, the slots shift left, m passes, and frame r, now
    at history_t, is committed. Returns each clip's latents [C, L, h, w].
    When enabled, cached.guidance_trace is [L-1, m, actual clips, metrics], indexed by entering frame minus one
    and substep. It excludes graph warmup, padded batch clips and invalid highest-noise slots during drain.
    guidance_trace_timesteps contains the normalized diffusion times for its substeps; guidance_trace_count
    counts written frame/substep observations per clip. Readback is left to the caller, outside GPU replay.
    """
    H, P, m, B = cached.H, cached.P, cached.m, cached.B
    count, fill = len(clips), B - len(clips)
    clips, seeds = clips + clips[-1:] * fill, seeds + seeds[-1:] * fill
    lengths = {latents.shape[1] for latents, _, _, _ in clips}
    assert len(lengths) == 1, f'a round advances every clip: lengths {lengths}'
    L = lengths.pop()
    if L < 2:
        raise RuntimeError(f'clip has {L} conditioned frames, need >= 2')
    C, _, h, w = clips[0][0].shape
    noise = torch.stack([torch.randn(C, L, h, w, generator=torch.Generator(device=device).manual_seed(seed), device=device)
                         for seed in seeds])  # generate's draw
    pose = torch.stack([clip_pose for _, clip_pose, _, _ in clips]).to(device)
    sink = torch.stack([latents[:, 0] for latents, _, _, _ in clips]).to(device)
    code_tables = [code_table.to(device) for _, _, code_table, _ in clips]
    codes = [c.to(device) for _, _, _, c in clips]
    cached.start_clip(sink, cuda_graph)
    if cached.guidance_metrics is not None:
        cached.guidance_trace = torch.full((L - 1, m, count, len(cached.guidance_metric_names)), float('nan'),
                                           dtype=torch.float32, device=device)
        cached.guidance_trace_timesteps = cached.t_table[
            cached.slot_steps[-1] + torch.arange(m, device=device)] / 1000
        cached.guidance_trace_count = 0

    rounds = range(2 - P, L)
    frames = torch.tensor([[r + j for j in range(P)] for r in rounds], device=device)
    valid, slots = (frames >= 1) & (frames < L), frames.clamp(0, L - 1)
    history_slot = lambda r, i: (cached.history_entries(r - 1 - H + i)[i - 1] if r - 1 - H + i >= 1
                                 else cached.pad_history0 + i - 1)
    ctx = torch.tensor([[*cached.fixed] + [history_slot(r, i) for i in range(1, H + 1)] for r in rounds], device=device)
    commits = torch.tensor([list(cached.history_entries(f)) for f in range(1, L)], device=device)
    empty = torch.zeros_like(noise[:, :, :1])

    history = [sink]
    torch.cuda.synchronize()
    start = time.time()
    for index, r in enumerate(rounds):
        enter = noise[:, :, r + P - 1:r + P] if r + P - 1 < L else empty
        cached.wave_latent.copy_(torch.cat((cached.wave_latent[:, :, 1:], enter), dim=2))
        cached.wave_pose.copy_(pose.index_select(2, slots[index]))
        cached.wave_valid.copy_(valid[index])
        cached.wave_ints[:cached.num_ctx].copy_(ctx[index])
        cached.set_wave_audio(code_tables, codes, min(r + P - 1, L - 1), slots[index])
        for s in range(m):
            cached.wave_ints[cached.num_ctx].fill_(s)
            cached.step()
            if cached.guidance_metrics is not None and 1 <= r + P - 1 < L:
                cached.guidance_trace[r + P - 2, s].copy_(cached.guidance_metrics[:count])
                cached.guidance_trace_count += 1
        if r < 1:
            continue
        cached.buf_latent.copy_(cached.wave_latent[:, :, 0])
        cached.buf_pose.copy_(pose[:, :, r])
        cached.buf_store.copy_(commits[r - 1])
        cached.commit()
        history.append(cached.buf_latent.clone())
        if on_commit is not None:
            on_commit(r, history[-1])
        if r % 100 == 0:
            torch.cuda.synchronize()
            elapsed = time.time() - start
            print(f'[{TAG}] {r}/{L - 1} frames, {elapsed:.0f}s, {1000 * elapsed / r:.0f} ms/frame', flush=True)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    frames = L - 1
    print(f'[{TAG}] wavefront {B} x {frames} frames in {elapsed:.1f}s: {1000 * elapsed / frames:.0f} ms/frame '
          f'({len(rounds) * m} passes of {P} frames + {frames} commits, and on_commit), '
          f'RTF {elapsed / (frames * 4 / FPS):.2f}', flush=True)
    generated = torch.stack(history, dim=2)
    return [generated[b] for b in range(count)]


def build_modes(config, base_dir, device):
    """
    Each mode's Models from {**config, **config[mode]}; one load when both name one checkpoint_path.

    -> (stabilising, resolving)
    """
    stabilising = build_models({**config, **config['stabilising']}, base_dir, device)
    resolving = config['resolving']
    if resolving['checkpoint_path'] == config['stabilising']['checkpoint_path']:
        return stabilising, stabilising.view(resolving['num_history'], resolving['num_present'])
    return stabilising, build_models({**config, **resolving}, base_dir, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    args = parser.parse_args()
    base_dir = Path(__file__).parent
    config = yaml.safe_load(open(args.config))
    device = 'cuda'

    nums, B = config['rows'], config['batch_size']
    batches = [nums[i:i + B] for i in range(0, len(nums), B)]
    batches = batches[args.shard * len(batches) // args.num_shards:(args.shard + 1) * len(batches) // args.num_shards]
    tag = f'{TAG}:{args.shard}/{args.num_shards}'
    print(f'[{tag}] rows {batches}', flush=True)
    if not batches:
        return

    if config['type'] != 'universal_wide_bimodal':
        raise ValueError(f"{config['type']}: this sampler is the bimodal model's")
    stab, res = config['stabilising'], config['resolving']
    for mode in (stab, res):
        assert set(MODE_KEYS) <= mode.keys(), f'mode blocks need {MODE_KEYS}'
    for path in (config['csv'], config['root_dir'], config['output_dir'], stab['checkpoint_path'], res['checkpoint_path']):
        assert os.path.isabs(path), f'{path}: paths must be absolute'
    stabilising, resolving = build_modes(config, base_dir, device)
    shared = stabilising.net.layers[0].audio_encoder is resolving.net.layers[0].audio_encoder
    resolver = ResolverNet(resolving)
    a = (1 - res['t_start']) / (1 - stab['history_t'])
    sigma = (res['t_start'] ** 2 - (a * stab['history_t']) ** 2) ** 0.5
    ts = torch.linspace(res['t_start'], 0, res['num_steps'] + 1, device=device)
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / Path(args.config).name)
    rows = list(csv.DictReader(open(config['csv'])))
    cached = None
    for group in batches:
        clips, audios = [], []
        for n in group:
            latents, pose, code_table, codes = load_chunk(
                os.path.join(config['root_dir'], rows[n]['paths']), config['pose_file'], config['max_latent_frames'],
                config['mimi_file'])
            if cached is None:
                net = WavefrontNet if config['wavefront'] else StabiliserCachedNet
                cached = net(stabilising, latents.shape[2:], stab['substeps'], device, config['compile'], B, stab['history_t'])
            audio = stabilising.mimi.decode(codes[None, :, :2 * latents.shape[1]].to(device)).audio_values[0, 0].float().cpu().numpy()
            decode_wan22_vae(stabilising.vae, latents[None], str(output_dir / f'samples_{n}.mp4'), fps=FPS, audio_np=audio, tag=tag)
            clips.append((latents, pose, code_table, codes))
            audios.append(audio)
        lengths = {c[0].shape[1] for c in clips}
        assert len(lengths) == 1, f'a batch resolves frame f of every clip at once: lengths {lengths}'
        filled = clips + clips[-1:] * (B - len(group))  # as generate fills a short batch
        seeds = [config['seed'] + n for n in group + group[-1:] * (B - len(group))]
        poses = torch.stack([c[1] for c in filled]).to(device)
        code_rows = torch.stack([c[3] for c in filled]).to(device)
        resolver_tables = [c[2].to(device) for c in filled]
        resolver_audio_cache = {}
        resolved = torch.zeros_like(poses)
        resolved[:, :, 0] = torch.stack([c[0][:, 0] for c in filled]).to(device)
        C, _, h, w = clips[0][0].shape

        def resolve(f, x):
            eps = torch.stack([torch.randn(C, h, w, device=device, generator=torch.Generator(device=device).manual_seed(
                seed * RESOLVE_SEED_STRIDE + f)) for seed in seeds])
            tokens = (cached.audio_cache[(audio_start(f),) * B] if shared else
                      resolver_tokens(resolver, resolver_tables, code_rows, f, resolver_audio_cache))
            with torch.autocast('cuda', dtype=AUTOCAST_DTYPE):
                resolver.frame(resolved, poses, code_rows, f, a * x + sigma * eps, ts, tokens)

        print(f'[{tag}] rows {group}: {lengths.pop() - 1} frames; stabilising {stab["substeps"]} substeps x '
              f'{stabilising.num_present} bands, t 1 -> {stab["history_t"]}; resolving '
              f'{a:.4f} x + {sigma:.4f} eps, {res["num_steps"]} steps t {res["t_start"]} -> 0; shared weights {shared}; '
              f'wavefront {config["wavefront"]}', flush=True)
        sample = generate_wavefront if config['wavefront'] else generate
        stabilised = sample(cached, clips, seeds[:len(group)], device, config['cuda_graph'], on_commit=resolve)
        for b, (n, audio) in enumerate(zip(group, audios)):
            for name, latents in ((f'sample_{n}', resolved[b]), (f'sample_{n}_stabilised', stabilised[b])):
                save_file({'latents': latents.cpu()}, str(output_dir / f'{name}_latents.safetensors'))
                decode_wan22_vae(stabilising.vae, latents[None], str(output_dir / f'{name}.mp4'), fps=FPS, audio_np=audio, tag=tag)
            side_by_side(output_dir / f'samples_{n}.mp4', output_dir / f'sample_{n}.mp4', output_dir / f'sample_{n}_sxs.mp4')
            print(f'[{tag}] row {n} {rows[n]["uuid"]} -> {output_dir}', flush=True)


if __name__ == '__main__':
    main()
