"""
Fused distillation rollouts: GT frames 1..H seed the stabilising wavefront in its present slots (noised along the
band schedule with fixed per-frame noise), and the new frames H+1..H+F are stabilised and resolved with every pass
carrying all the work available at that moment, so no pass is a pure prefill:

  S  the P stabilising present frames (slot j at band j, Euler-stepped),
  R  the resolving present frame (renoised t_h -> t_start on entry, one Euler step per pass),
  A  the frame that just left band 0 at t_h, as stabilising history (sink + self at the anchor; K stored at every
     history slot, and read at slot H by this pass's S),
  Q  the newest resolved frame (GT for seeds), as resolving history (sink + self at slot Hr; read by this pass's R).

A and Q ride on the first pass of a round only. Round r (1 .. H+F+1) holds frames r .. r+P-1; frames past H+F are
pads, and attention is causal (slot j reads the sink, the H frames to its left and itself), so the trimmed passes run
only the first P' = min(P, L - r) stabilising slots: dropping right-side slots changes no left slot, only GEMM shapes.
A layout is (P', R, A, Q), one compiled graph each: S4 (r = 1, s > 0 of r <= H + 1), S4+A+Q (s = 0 of r = 2 .. H + 1,
no dummy R), S{P'}+R(+A+Q) once the resolver runs, and R / R+Q (round H+F+1: no stabilising frame, so no A). The
slots past P' of the wave buffers are left as they are; they only ever hold pads. (H+F+P-1) m + m passes for F new
frames, against (H+F) m four-frame passes + commits + (m + 1) eager resolver passes per frame for the unfused
components (reference()).

  python bimodal/fused_rollout_trim.py CONFIG ROW MODE BATCH OUT   (MODE: fused | reference)
"""

import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch._dynamo
import yaml
from flash_attn_interface import flash_attn_func
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / 'production')]

from bimodal.cached_resolver import ResolverNet
from bimodal.sampler import WavefrontNet, build_modes
from cached_universal import SINK, audio_start
from models.pose import rope
from models.wan.model import sinusoidal_embedding_1d
from sink_universal import load_chunk
from utils.common import AUTOCAST_DTYPE

TAG = 'trim'
# Every layout shares one lambda code object, so dynamo's per-code recompile limit (8) would silently drop the
# ninth layout to eager.
torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)


def layout_name(key):
    Ps, res, a, q = key
    return (f'S{Ps}' if Ps else '') + 'R' * res + 'A' * a + 'Q' * q


def layout_frames(key):
    Ps, res, a, q = key
    return Ps + res + a + q


def pass_layout(r, s, L, H, P):
    """
    Round r, substep s -> (P', R, A, Q): the round's real stabilising frames min(P, L - r) (the first slots), the
    resolver once frame r - 1 >= H + 1 resolves, and on s = 0 of r >= 2 the commits: A while stabilising frames
    remain to read it, Q always (it only matters from r = H + 1 on, as fused_rollout.py).
    """
    Ps = max(0, min(P, L - r))
    res = r - 1 >= H + 1
    commit = s == 0 and r >= 2
    return Ps, res, commit and Ps > 0, commit


def plan(L, H, P, m):
    return [pass_layout(r, s, L, H, P) for r in range(1, L + 1) for s in range(m)]


class FusedNet(WavefrontNet):
    def __init__(self, models, resolving, latent_hw, substeps, device, compile, batch, history_t, t_start, num_steps):
        super().__init__(models, latent_hw, substeps, device, compile, batch, history_t)
        assert num_steps == substeps, 'one resolver step per stabilising substep'
        B, P, N, n, d, C, D = self.B, self.P, self.N, self.n, self.d, self.initial.patch_embedding.in_channels, self.initial.dim
        self.Hr, layers = resolving.num_history, len(self.blocks)
        assert self.Hr == 2
        self.forced = torch.zeros(P, dtype=torch.bool, device=device)
        self.forced_x0 = torch.zeros(B, C, P, *latent_hw, device=device)
        self.forced_eps = torch.zeros(B, C, P, *latent_hw, device=device)
        self.res_latent = torch.zeros(B, C, 1, *latent_hw, device=device)
        self.res_pose = torch.zeros(B, C, 1, *latent_hw, device=device)
        self.res_audio = torch.zeros(B, 1, self.K + 1, D, dtype=AUTOCAST_DTYPE, device=device)
        self.res_t = torch.zeros(1, device=device)  # * 1000
        self.res_dt = -t_start / num_steps
        self.commit_latent = torch.zeros(B, C, 2, *latent_hw, device=device)  # [A, Q]
        self.commit_pose = torch.zeros(B, C, 2, *latent_hw, device=device)
        self.commit_t = torch.tensor([history_t * 1000, 0.0], device=device)
        # Resolving history per layer: [older K at slot 1, newer K at slot 2], their V, and the newer's K at slot 1.
        self.res_k = torch.zeros(layers, B, 2, N, n, d, dtype=torch.bfloat16, device=device)
        self.res_v = torch.zeros(layers, B, 2, N, n, d, dtype=torch.bfloat16, device=device)
        self.res_k1 = torch.zeros(layers, B, N, n, d, dtype=torch.bfloat16, device=device)
        self.compile_layouts = compile
        self.fused = {}  # layout (P', R, A, Q) -> its (compiled) pass, built on first use
        self.fused_graphs = None
        self.capture_seconds = 0.0

    def layout_fn(self, key):
        if key not in self.fused:
            Ps, res, a, q = key
            assert Ps or res, f'{layout_name(key)}: a layout needs a stabilising or resolving frame'
            sf = self.slot_freqs
            freqs = torch.cat([self.wave_freqs[:Ps * self.N]] * bool(Ps) + [sf[3]] * res + [sf[self.H]] * a + [sf[self.Hr]] * q)
            fn = lambda: self._fused(Ps, res, a, q, freqs)
            self.fused[key] = torch.compile(fn, dynamic=False) if self.compile_layouts else fn
        return self.fused[key]

    def _fused(self, Ps, res, a, q_commit, freqs):
        """
        One pass of the layout's frames [S_0 .. S_{P'-1} | R | A | Q], a pure function of the static buffers
        -> (v_S and v_R, A's K/V or None, Q's K/V or None).
        """
        initial, N, n, d, B, H = self.initial, self.N, self.n, self.d, self.B, self.H
        latents, poses, ts, valids = [], [], [], []
        if Ps:
            t = self.t_table.index_select(0, self.slot_steps[:Ps] + self.wave_ints[1 + H]) * self.wave_valid[:Ps]
            u = (t / 1000)[None, None, :, None, None]
            forced = self.forced[None, None, :Ps, None, None]
            latents.append(torch.where(forced, (1 - u) * self.forced_x0[:, :, :Ps] + u * self.forced_eps[:, :, :Ps],
                                       self.wave_latent[:, :, :Ps]))
            poses.append(self.wave_pose[:, :, :Ps])
            ts.append(t)
            valids.append(self.wave_valid[:Ps])
        one = torch.ones(1, dtype=torch.bool, device=self.device)
        if res:
            latents.append(self.res_latent)
            poses.append(self.res_pose)
            ts.append(self.res_t)
            valids.append(one)
        nc = int(a) + int(q_commit)  # commit frames, [A, Q] or [Q]
        if nc:
            c0 = 2 - nc
            latents.append(self.commit_latent[:, :, c0:])
            poses.append(self.commit_pose[:, :, c0:])
            ts.append(self.commit_t[c0:])
            valids.append(one.expand(nc))
        latent, pose, t, valid = (torch.cat(z, dim=2 if z[0].dim() == 5 else 0) for z in (latents, poses, ts, valids))
        F = t.shape[0]
        heads = Ps + int(res)
        ia, iq = heads, heads + int(a)  # A's and Q's frames
        v_mask = valid[None, :, None, None]
        ctx = (self.wave_ints[None, :1 + H] + self.offsets).flatten()
        sink_ctx = (torch.full((1, 1), SINK, device=self.device) + self.offsets).flatten()
        key_slots = self.key_slots[:Ps]
        a_ks, a_vs, q_k1s, q_k2s, q_vs = [], [], [], [], []
        with torch.autocast('cuda', dtype=AUTOCAST_DTYPE, cache_enabled=False):
            x = initial.patch_embedding(latent).flatten(2).transpose(1, 2).unflatten(1, (F, N))
            x = torch.where(v_mask, x, initial.pad_token.to(x.dtype))
            x = x + initial.pose_patch_embedding(pose.to(x.dtype)).flatten(2).transpose(1, 2).unflatten(1, (F, N)) * v_mask
            e = initial.time_embedding(sinusoidal_embedding_1d(initial.freq_dim, t).unflatten(0, (1, F)))
            e0 = initial.time_projection(e).unflatten(2, (6, initial.dim))
            audio = torch.cat([self.wave_audio[:, :Ps]] * bool(Ps) + [self.res_audio] * res, dim=1)

            for i, layer in enumerate(self.blocks):
                block, attention = layer.block, layer.block.self_attn
                modulation = (block.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
                normalized = (block.norm1(x) * (1 + modulation[1]) + modulation[0]).flatten(1, 2)
                k_raw = attention.norm_k(attention.k(normalized)).view(B, F * N, n, d)
                q = rope.apply_rope(attention.norm_q(attention.q(normalized)).view(B, F * N, n, d), freqs)
                q = q.bfloat16().view(B, F, N, n, d)
                k = rope.apply_rope(k_raw, freqs).bfloat16().view(B, F, N, n, d)
                v = attention.v(normalized).view(B, F, N, n, d).bfloat16()
                sink_k = self.pool_k[i].index_select(0, sink_ctx).view(B, 1, N, n, d)
                sink_v = self.pool_v[i].index_select(0, sink_ctx).view(B, 1, N, n, d)
                outputs = []
                if Ps:
                    hist_k = self.pool_k[i].index_select(0, ctx).view(B, 1 + H, N, n, d)
                    hist_v = self.pool_v[i].index_select(0, ctx).view(B, 1 + H, N, n, d)
                    if a:  # A is history slot H
                        hist_k = torch.cat((hist_k[:, :H], k[:, ia:ia + 1]), dim=1)
                        hist_v = torch.cat((hist_v[:, :H], v[:, ia:ia + 1]), dim=1)
                    keys = torch.cat((hist_k, k[:, :Ps]), dim=1)[:, key_slots].flatten(2, 3).flatten(0, 1)
                    values = torch.cat((hist_v, v[:, :Ps]), dim=1)[:, key_slots].flatten(2, 3).flatten(0, 1)
                    outputs.append(flash_attn_func(q[:, :Ps].flatten(0, 1), keys, values).view(B, Ps, N, n, d))
                if res:
                    if q_commit:  # older: last newer re-rotated at slot 1; newer: Q at slot 2
                        rk = torch.stack((self.res_k1[i], k[:, iq]), dim=1)
                        rv = torch.stack((self.res_v[i][:, 1], v[:, iq]), dim=1)
                    else:
                        rk, rv = self.res_k[i], self.res_v[i]
                    keys = torch.cat((sink_k, rk, k[:, Ps:Ps + 1]), dim=1).flatten(1, 2)
                    values = torch.cat((sink_v, rv, v[:, Ps:Ps + 1]), dim=1).flatten(1, 2)
                    outputs.append(flash_attn_func(q[:, Ps], keys, values)[:, None])
                if nc:  # A and Q: sink + self
                    keys = torch.cat((sink_k.expand(B, nc, N, n, d)[:, :, None], k[:, heads:, None]), dim=2)
                    values = torch.cat((sink_v.expand(B, nc, N, n, d)[:, :, None], v[:, heads:, None]), dim=2)
                    outputs.append(flash_attn_func(q[:, heads:].flatten(0, 1), keys.flatten(0, 1).flatten(1, 2),
                                                   values.flatten(0, 1).flatten(1, 2)).view(B, nc, N, n, d))
                    raw = k_raw.view(B, F, N, n, d)
                    if a:
                        a_ks.append(rope.apply_rope(raw[:, ia].repeat(1, H, 1, 1), self.history_freqs.flatten(0, 1))
                                    .bfloat16().view(B, H, N, n, d))
                        a_vs.append(v[:, ia])
                    if q_commit:
                        q_k1s.append(rope.apply_rope(raw[:, iq], self.slot_freqs[1]).bfloat16())
                        q_k2s.append(k[:, iq])
                        q_vs.append(v[:, iq])
                output = torch.cat(outputs, dim=1).to(x.dtype)
                x = x + attention.o(output.flatten(3)) * modulation[2]
                if layer.audio_block is not None and heads:
                    present = x[:, :heads]
                    update = layer.audio_block.forward_present(present, audio) * v_mask[:, :heads]
                    x = torch.cat((present + update, x[:, heads:]), dim=1)
                residual = block.ffn(block.norm2(x) * (1 + modulation[4]) + modulation[3])
                x = x + residual * modulation[5]

            final = self.final.head
            with torch.autocast('cuda', dtype=torch.float32, cache_enabled=False):
                modulation = (final.modulation.unsqueeze(0) + e[:, :heads].unsqueeze(2)).chunk(2, dim=2)
                y = final.head(final.norm(x[:, :heads]) * (1 + modulation[1]) + modulation[0])
        velocity = self.final.unpatchify(y.flatten(1, 2), (heads, *self.grid[1:])).float()
        a_kv = (torch.stack(a_ks), torch.stack(a_vs)) if a else None
        q_kv = (torch.stack(q_k1s), torch.stack(q_k2s), torch.stack(q_vs)) if q_commit else None
        return velocity, a_kv, q_kv

    def _fused_step(self, key):
        Ps, res, a, q = key
        velocity, a_kv, q_kv = self.layout_fn(key)()
        if Ps:  # only the real slots; the rest hold pads nothing reads
            wave = self.wave_latent[:, :, :Ps]
            wave.copy_(wave + velocity[:, :, :Ps] * self.dt)
        if res:
            self.res_latent.copy_(self.res_latent + velocity[:, :, Ps:Ps + 1] * self.res_dt)
        if a:
            self._store(self.buf_store + self.offsets, *a_kv)
        if q:
            q_k1, q_k2, q_v = q_kv
            self.res_k[:, :, 0].copy_(self.res_k1)
            self.res_v[:, :, 0].copy_(self.res_v[:, :, 1])
            self.res_k[:, :, 1].copy_(q_k2)
            self.res_v[:, :, 1].copy_(q_v)
            self.res_k1.copy_(q_k1)

    def run(self, key):
        if self.fused_graphs is None:
            return self._fused_step(key)
        self.fused_graphs[key].replay()

    @torch.no_grad()
    def capture(self, keys):
        start = time.time()
        self.fused_graphs = {}
        saved = [z.clone() for z in (self.wave_latent, self.res_latent, self.res_k, self.res_v, self.res_k1)]
        self.buf_store.fill_(0)  # SCRATCH
        for key in keys:
            begin = time.time()
            self.fused_graphs[key] = self._capture(lambda key=key: self._fused_step(key))
            print(f'[{TAG}] layout {layout_name(key)} ({layout_frames(key)} frames) compiled + captured in '
                  f'{time.time() - begin:.0f}s', flush=True)
        for z, s in zip((self.wave_latent, self.res_latent, self.res_k, self.res_v, self.res_k1), saved):
            z.copy_(s)
        self.capture_seconds = time.time() - start
        print(f'[{TAG}] {len(keys)} fused graphs compiled + captured in {self.capture_seconds:.0f}s', flush=True)


def seed_noise(shape, seed, device):
    return torch.randn(shape, generator=torch.Generator(device=device).manual_seed(seed), device=device)


def task_inputs(clips, seeds, H, Fnew, device):
    """Per clip: sink, GT frames, pose, noise for every frame (GT forcing and new frames), resolver renoise."""
    C, _, h, w = clips[0][0].shape
    L = 1 + H + Fnew
    gt = torch.stack([c[0][:, :L] for c in clips]).to(device)
    pose = torch.stack([c[1][:, :L] for c in clips]).to(device)
    noise = torch.stack([seed_noise((C, L, h, w), s, device) for s in seeds])
    renoise = torch.stack([seed_noise((C, L, h, w), s * 7919 + 1, device) for s in seeds])
    return gt, pose, noise, renoise


@torch.no_grad()
def fused(net, resolver, clips, seeds, device, H, Fnew, t_start, history_t, cuda_graph=True, timed=True):
    """-> resolved [B, C, 1 + H + Fnew, h, w] (GT 0..H, then the new frames) and the stabilised new frames."""
    P, m, B = net.P, net.m, net.B
    gt, pose, noise, renoise = task_inputs(clips, seeds, H, Fnew, device)
    L = 1 + H + Fnew
    code_tables = [c[2].to(device) for c in clips]
    codes = [c[3].to(device) for c in clips]
    code_rows = torch.stack(codes)
    a = (1 - t_start) / (1 - history_t)
    sigma = (t_start ** 2 - (a * history_t) ** 2) ** 0.5
    net.start_clip(gt[:, :, 0], False)
    net.set_wave_audio(code_tables, codes, L - 1, torch.arange(P, device=device).clamp(max=L - 1))
    tokens = net.audio_cache[(audio_start(L - 1),) * B]
    schedule = plan(L, H, P, m)
    if cuda_graph and net.fused_graphs is None:
        net.capture(list(dict.fromkeys(schedule)))
    resolved = gt.clone()
    resolved[:, :, H + 1:] = 0
    stabilised = torch.zeros_like(gt)
    hist = lambda f, i: net.history_entries(f)[i - 1] if f >= 1 else net.pad_history0 + i - 1
    ts = torch.linspace(t_start, 0, m + 1).tolist()
    torch.cuda.synchronize()
    start = time.time()
    passes, counts = 0, Counter()
    for r in range(1, L + 1):  # frame r in slot 0; frame r - 1 leaves into history, frame r - 1 >= H + 1 resolves
        frames = [r + j for j in range(P)]
        valid = [1 <= f < L for f in frames]
        clamp = [min(max(f, 0), L - 1) for f in frames]
        Ps = pass_layout(r, 0, L, H, P)[0]
        assert valid == [True] * Ps + [False] * (P - Ps), f'round {r}: the real frames must be the first {Ps} slots'
        if r > 1:
            stabilised[:, :, r - 1] = net.wave_latent[:, :, 0]
        enter = noise[:, :, r + P - 1:r + P] if r + P - 1 < L else torch.zeros_like(noise[:, :, :1])
        net.wave_latent.copy_(torch.cat((net.wave_latent[:, :, 1:], enter), dim=2))
        net.wave_pose.copy_(pose[:, :, clamp])
        net.wave_valid.copy_(torch.tensor(valid, device=device))
        net.forced.copy_(torch.tensor([1 <= f <= H for f in frames], device=device))
        net.forced_x0.copy_(gt[:, :, clamp])
        net.forced_eps.copy_(noise[:, :, clamp])
        net.wave_ints[:1 + net.H].copy_(torch.tensor([SINK] + [hist(r - 1 - net.H + i, i) for i in range(1, net.H + 1)], device=device))
        net.set_wave_audio(code_tables, codes, min(r + P - 1, L - 1), torch.tensor(clamp, device=device))
        g = r - 1  # resolving this round
        res = g >= H + 1
        if r >= 2:
            x_h = (1 - history_t) * gt[:, :, g] + history_t * noise[:, :, g] if g <= H else stabilised[:, :, g]
            stabilised[:, :, g] = x_h
            net.commit_latent[:, :, 0] = x_h
            net.commit_latent[:, :, 1] = resolved[:, :, max(g - 1, 0)]
            net.commit_pose[:, :, 0] = pose[:, :, g]
            net.commit_pose[:, :, 1] = pose[:, :, max(g - 1, 0)]
            net.buf_store.copy_(torch.tensor(list(net.history_entries(g)), device=device))
            net.res_latent[:, :, 0] = a * x_h + sigma * renoise[:, :, g] if res else 0
            net.res_pose[:, :, 0] = pose[:, :, g]
            if res:
                net.res_audio.copy_(resolver.audio(code_rows, g, tokens))
        for s in range(m):
            net.wave_ints[1 + net.H].fill_(s)
            net.res_t.fill_(ts[s] * 1000)
            key = schedule[(r - 1) * m + s]
            assert key[1] == res
            net.run(key)
            passes += 1
            counts[key] += 1
        if res:
            resolved[:, :, g] = net.res_latent[:, :, 0]
    torch.cuda.synchronize()
    elapsed = time.time() - start
    if timed:
        frames_run = sum(layout_frames(key) * c for key, c in counts.items())
        print(f'[{TAG}] fused B={B}: {Fnew} frames in {elapsed * 1000:.0f} ms, {passes} passes, {frames_run} frames, '
              f'{elapsed * 1000 / passes:.2f} ms/pass', flush=True)
        if not getattr(net, 'printed_counts', False):
            print(f'[{TAG}] passes per layout: ' + ', '.join(
                f'{layout_name(key)} x{c} ({layout_frames(key) * c} frames)' for key, c in counts.items()), flush=True)
            net.printed_counts = True
    return resolved, stabilised, elapsed


@torch.no_grad()
def reference(net, resolver, clips, seeds, device, H, Fnew, t_start, history_t, cuda_graph=True, timed=True):
    """The same rollout from the unfused components: WavefrontNet passes, commit passes, ResolverNet.frame."""
    P, m, B = net.P, net.m, net.B
    gt, pose, noise, renoise = task_inputs(clips, seeds, H, Fnew, device)
    L = 1 + H + Fnew
    code_tables = [c[2].to(device) for c in clips]
    codes = [c[3].to(device) for c in clips]
    code_rows = torch.stack(codes)
    a = (1 - t_start) / (1 - history_t)
    sigma = (t_start ** 2 - (a * history_t) ** 2) ** 0.5
    net.start_clip(gt[:, :, 0], cuda_graph)
    net.set_wave_audio(code_tables, codes, L - 1, torch.arange(P, device=device).clamp(max=L - 1))
    tokens = net.audio_cache[(audio_start(L - 1),) * B]
    resolved = gt.clone()
    resolved[:, :, H + 1:] = 0
    stabilised = torch.zeros_like(gt)
    hist = lambda f, i: net.history_entries(f)[i - 1] if f >= 1 else net.pad_history0 + i - 1
    ts = torch.linspace(t_start, 0, m + 1, device=device)
    t_table = net.t_table / 1000
    torch.cuda.synchronize()
    start = time.time()
    for r in range(1, L):
        frames = [r + j for j in range(P)]
        clamp = [min(max(f, 0), L - 1) for f in frames]
        enter = noise[:, :, r + P - 1:r + P] if r + P - 1 < L else torch.zeros_like(noise[:, :, :1])
        net.wave_latent.copy_(torch.cat((net.wave_latent[:, :, 1:], enter), dim=2))
        net.wave_pose.copy_(pose[:, :, clamp])
        net.wave_valid.copy_(torch.tensor([1 <= f < L for f in frames], device=device))
        net.wave_ints[:1 + net.H].copy_(torch.tensor([SINK] + [hist(r - 1 - net.H + i, i) for i in range(1, net.H + 1)], device=device))
        net.set_wave_audio(code_tables, codes, min(r + P - 1, L - 1), torch.tensor(clamp, device=device))
        for s in range(m):
            for j, f in enumerate(frames):
                if 1 <= f <= H:
                    u = t_table[j * m + s]
                    net.wave_latent[:, :, j] = (1 - u) * gt[:, :, f] + u * noise[:, :, f]
            net.wave_ints[1 + net.H].fill_(s)
            net.step()
        x_h = (1 - history_t) * gt[:, :, r] + history_t * noise[:, :, r] if r <= H else net.wave_latent[:, :, 0].clone()
        stabilised[:, :, r] = x_h
        net.buf_latent.copy_(x_h)
        net.buf_pose.copy_(pose[:, :, r])
        net.buf_store.copy_(torch.tensor(list(net.history_entries(r)), device=device))
        net.commit()
        if r >= H + 1:
            with torch.autocast('cuda', dtype=AUTOCAST_DTYPE):
                resolver.frame(resolved, pose, code_rows, r, a * x_h + sigma * renoise[:, :, r], ts, tokens)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    if timed:
        passes = (L - 1) * (m + 1) + Fnew * (m + 1)
        print(f'[{TAG}] reference B={B}: {Fnew} frames in {elapsed * 1000:.0f} ms, {passes} passes', flush=True)
    return resolved, stabilised, elapsed


def main():
    config_path, row, mode, B, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
    config = yaml.safe_load(open(config_path))
    device = 'cuda'
    H, Fnew = 3, 4
    stab, res = config['stabilising'], config['resolving']
    stabilising, resolving = build_modes(config, Path(__file__).parent, device)
    import csv, os
    rows = list(csv.DictReader(open(config['csv'])))
    clips = [load_chunk(os.path.join(config['root_dir'], rows[row + b]['paths']), config['pose_file'], 64, config['mimi_file'])
             for b in range(B)]
    seeds = [config['seed'] + row + b for b in range(B)]
    args = (stab['history_t'],)
    if mode == 'fused':
        net = FusedNet(stabilising, resolving, clips[0][0].shape[2:], stab['substeps'], device, config['compile'], B,
                       stab['history_t'], res['t_start'], res['num_steps'])
        run = fused
    else:
        net = WavefrontNet(stabilising, clips[0][0].shape[2:], stab['substeps'], device, config['compile'], B, stab['history_t'])
        run = reference
    resolver = ResolverNet(resolving)
    times = []
    for it in range(4):
        resolved, stabilised, elapsed = run(net, resolver, clips, seeds, device, H, Fnew, res['t_start'], stab['history_t'],
                                            cuda_graph=config['cuda_graph'])
        times.append(elapsed)
    compiled = f', compile + capture {net.capture_seconds:.0f}s' if mode == 'fused' else ''
    print(f'[{TAG}] {mode} B={B} steady {1000 * min(times[1:]):.0f} ms per rollout of {Fnew} frames '
          f'({1000 * min(times[1:]) / Fnew / B:.1f} ms per frame per clip; rollouts '
          f'{", ".join(f"{1000 * t:.0f}" for t in times)} ms{compiled})', flush=True)
    os.makedirs(out, exist_ok=True)
    path = f'{out}/{mode}_B{B}_row{row}.safetensors'
    save_file({'resolved': resolved.cpu().contiguous(), 'stabilised': stabilised.cpu().contiguous()}, path)
    print(f'[{TAG}] saved {path}', flush=True)


if __name__ == '__main__':
    main()
