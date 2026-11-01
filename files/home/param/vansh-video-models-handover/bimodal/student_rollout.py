"""
Differentiable fused rollouts of the bimodal model for distillation: the sink is GT frame 0, GT frames 1..H seed the
stabilising present (noised along the schedule with fixed per-frame noise), and F new frames are stabilised and
resolved. Nothing is written in place, so gradients reach every pass through the K/V each frame leaves behind.

Schedule. band_steps[j] is the number of steps of present slot j (band j, 0 = least noise); a round is
M = max(band_steps) passes and moves every frame down one band. A slot with M steps is evaluated and stepped every
pass; a slot with 1 step is evaluated at the round's first pass, at the top of its band, its K/V held for the rest of
the round (its latent and its left context do not change) and its step applied at the round's end. The frame leaving
slot 0 at history_t is renoised to t_start and resolved in the next round, one of resolve_steps = M steps per pass.
The first pass of a round also carries the commits: A, the frame that just left at history_t, as stabilising
history (sink + self at the anchor slot H), read at slot H by this pass's present frames; Q, the newest resolved
frame (GT for seeds), as resolving history (sink + self at slot Hr), read by this pass's resolving frame.
Right-pad present frames (past the rollout) are never evaluated: attention is causal.

Each pass is one forward of all its frames: the shared Linear / FFN / AdaLN / audio work runs on the concatenated
tokens, attention in three groups by key count (present: sink + H left + self; resolving: sink + Hr + self;
commits: sink + self). checkpoint=True recomputes each block in backward from its input and the K/V it reads.

  python bimodal/student_rollout.py CONFIG ROW TEST OUT    (TEST: teacher | student)
"""

import math
import os
import sys
import time
from pathlib import Path

import torch
from flash_attn_interface import flash_attn_func
from torch.utils.checkpoint import checkpoint

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / 'production')]

from cached_universal import audio_start
from inference.common import sliding_code_table
from models.pose import rope
from models.pose.audio import audio_latents_from_codes
from models.wan.model import sinusoidal_embedding_1d
from utils.common import AUTOCAST_DTYPE

TAG = 'student'


class StudentRollout:
    def __init__(self, models, latent_hw, band_steps, resolve_steps, history_t=0.2, t_start=0.4, num_seeds=3,
                 num_new=4, resolving_history=2, checkpoint_blocks=True, compile=False):
        layers = models.net.layers
        self.initial, self.blocks, self.final = layers[0], list(layers[1:-1]), layers[-1]
        self.mimi, self.num_codebooks = models.mimi, models.num_codebooks
        self.H, self.P, self.Hr = models.num_history, models.num_present, resolving_history
        H, P = self.H, self.P
        self.M = max(band_steps)
        assert len(band_steps) == P and all(k in (1, self.M) for k in band_steps), band_steps
        assert resolve_steps == self.M, 'one resolving step per pass'
        assert num_seeds == H, 'the seeds fill the history once they leave the present'
        self.band_steps, self.h, self.t_start = list(band_steps), history_t, t_start
        self.num_seeds, self.num_new = num_seeds, num_new
        attention = self.blocks[0].block.self_attn
        self.n, self.d = attention.num_heads, attention.head_dim
        patch = self.initial.patch_embedding.kernel_size
        self.grid = (1, latent_hw[0] // patch[1], latent_hw[1] // patch[2])
        self.N = self.grid[1] * self.grid[2]
        self.K = self.initial.audio_cross_attention_codes
        freqs = rope.cached_freqs_i({}, (1 + H + P, *self.grid[1:]), self.blocks[0].freqs, 0)
        self.slot_freqs = freqs.unflatten(0, (1 + H + P, self.N))  # [slot, N, 1, c, 2]
        self.checkpoint_blocks = checkpoint_blocks
        self.width = (1 - history_t) / P
        self.a = (1 - t_start) / (1 - history_t)
        self.sigma = (t_start ** 2 - (self.a * history_t) ** 2) ** 0.5
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
        if os.environ.get('EMULATE_CASTS') == '1':  # inductor rounds intermediates as eager autocast does
            torch._inductor.config.emulate_precision_casts = True
        self.block_fn = torch.compile(self._block, dynamic=False) if compile else self._block

    # ---- schedule ----
    def top(self, j):
        return self.h + self.width * (j + 1)

    def slot_t(self, j, p):
        """Slot j's t at pass p of a round, for a slot evaluated at p."""
        return self.top(j) - p * self.width / self.band_steps[j]

    # ---- the network ----
    def embed(self, latents, poses, ts, valid, posed):
        """latents / poses [B, C, F, h, w], ts list of F floats; valid False: the pad token; posed False: no pose."""
        initial, N = self.initial, self.N
        F = len(ts)
        x = initial.patch_embedding(latents).flatten(2).transpose(1, 2).unflatten(1, (F, N))
        x = torch.where(torch.tensor(valid, device=x.device)[None, :, None, None], x, initial.pad_token.to(x.dtype))
        if any(posed):
            pose = initial.pose_patch_embedding(poses.to(x.dtype)).flatten(2).transpose(1, 2).unflatten(1, (F, N))
            x = x + pose * torch.tensor(posed, device=x.device)[None, :, None, None]
        t = torch.tensor(ts, device=x.device, dtype=torch.float32) * 1000
        e = initial.time_embedding(sinusoidal_embedding_1d(initial.freq_dim, t).unflatten(0, (1, F)))
        return x, e, initial.time_projection(e).unflatten(2, (6, initial.dim))

    def _block(self, layer, plan, heads, x, e0, freqs, audio, *ext):
        """
        One block of a pass. plan: per attention group (frames, specs), specs[f] the frame's keys in slot order,
        each ('e', k, slot) -> ext[2k] (K before RoPE, rotated here at slot), ext[2k + 1] (V), or ('f', g) -> this
        pass's frame g.
        -> x, K before RoPE, V [B, F, N, n, d]
        """
        block, attention = layer.block, layer.block.self_attn
        B, F, N = x.shape[0], x.shape[1], self.N
        n, d = self.n, self.d
        modulation = (block.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
        normalized = (block.norm1(x) * (1 + modulation[1]) + modulation[0]).flatten(1, 2)
        q = rope.apply_rope(attention.norm_q(attention.q(normalized)).view(B, F * N, n, d), freqs)
        k_raw = attention.norm_k(attention.k(normalized)).view(B, F * N, n, d)
        k = rope.apply_rope(k_raw, freqs).bfloat16().view(B, F, N, n, d)
        q = q.bfloat16().view(B, F, N, n, d)
        v = attention.v(normalized).view(B, F, N, n, d).bfloat16()
        key = lambda s: rope.apply_rope(ext[2 * s[1]], self.slot_freqs[s[2]]).bfloat16() if s[0] == 'e' else k[:, s[1]]
        value = lambda s: ext[2 * s[1] + 1] if s[0] == 'e' else v[:, s[1]]
        outputs = []
        for frames, specs in plan:
            keys = torch.stack([torch.cat([key(s) for s in spec], dim=1) for spec in specs], dim=1)
            values = torch.stack([torch.cat([value(s) for s in spec], dim=1) for spec in specs], dim=1)
            G = len(frames)
            out = flash_attn_func(q[:, frames[0]:frames[0] + G].flatten(0, 1), keys.flatten(0, 1), values.flatten(0, 1))
            outputs.append(out.view(B, G, N, n, d))
        x = x + attention.o(torch.cat(outputs, dim=1).to(x.dtype).flatten(3)) * modulation[2]
        if layer.audio_block is not None and heads:
            x = torch.cat((x[:, :heads] + layer.audio_block.forward_present(x[:, :heads], audio), x[:, heads:]), dim=1)
        x = x + block.ffn(block.norm2(x) * (1 + modulation[4]) + modulation[3]) * modulation[5]
        return x, k_raw.view(B, F, N, n, d), v

    def forward_pass(self, frames, plan, ext, audio, heads):
        """
        frames: list of (latent [B, C, h, w], pose [B, C, h, w] or None, t, slot, valid), groups contiguous in plan
        order; ext[i]: layer i's external K/V list; audio [B, heads, K + 1, D].
        -> velocity [B, C, heads, h, w] or None, per-layer K before RoPE and V [B, F, N, n, d]
        """
        latents = torch.stack([f[0] for f in frames], dim=2)
        poses = torch.stack([f[1] if f[1] is not None else torch.zeros_like(f[0]) for f in frames], dim=2)
        with torch.autocast('cuda', dtype=AUTOCAST_DTYPE):
            x, e, e0 = self.embed(latents, poses, [f[2] for f in frames], [f[4] for f in frames],
                                  [f[1] is not None for f in frames])
            freqs = torch.cat([self.slot_freqs[f[3]] for f in frames])
            ks, vs = [], []
            for i, layer in enumerate(self.blocks):
                args = (layer, plan, heads, x, e0, freqs, audio, *ext[i])
                if self.checkpoint_blocks and torch.is_grad_enabled():
                    x, k, v = checkpoint(self.block_fn, *args, use_reentrant=False)
                else:
                    x, k, v = self.block_fn(*args)
                ks.append(k)
                vs.append(v)
            velocity = None
            if heads:
                final = self.final.head
                with torch.autocast('cuda', dtype=torch.float32):
                    modulation = (final.modulation.unsqueeze(0) + e[:, :heads].unsqueeze(2)).chunk(2, dim=2)
                    y = final.head(final.norm(x[:, :heads]) * (1 + modulation[1]) + modulation[0])
                velocity = self.final.unpatchify(y.flatten(1, 2), (heads, *self.grid[1:])).float()
        return velocity, ks, vs

    def context_frame(self, latent, pose, t, slot, sink_kv, valid=True):
        """A frame attending sink + self at `slot` (sink_kv None: the sink itself, at slot 0) -> per-layer K raw, V."""
        if sink_kv is None:
            plan = [([0], [[('f', 0)]])]
            ext = [[] for _ in self.blocks]
        else:
            plan = [([0], [[('e', 0, 0), ('f', 0)]])]
            ext = sink_kv
        _, ks, vs = self.forward_pass([(latent, pose, t, slot, valid)], plan, ext, None, 0)
        return [k[:, 0] for k in ks], [v[:, 0] for v in vs]

    # ---- audio ----
    def audio_tokens(self, code_tables, codes, f_last):
        tables = [sliding_code_table(table, c, f_last)[0] for table, c in zip(code_tables, codes)]
        latents = audio_latents_from_codes(self.mimi, torch.cat(tables), self.num_codebooks)
        with torch.autocast('cuda', dtype=AUTOCAST_DTYPE):
            return self.initial.audio_encoder(latents), audio_start(f_last)

    def frame_audio(self, tokens, a, f):
        positions = self.K * (max(f - a, 1) - 1) + torch.arange(self.K, device=tokens.device)
        null = self.initial.audio_encoder.null_token.to(tokens.dtype).reshape(1, 1, -1).expand(tokens.shape[0], 1, -1)
        return torch.cat((tokens[:, positions], null), dim=1)

    # ---- rollout ----
    def rollout(self, gt, pose, noise, renoise, code_tables, codes, log=None):
        """
        gt, pose, noise, renoise [B, C, L, h, w], L = 1 + num_seeds + num_new (frame 0 the sink).
        -> resolved [B, C, num_new, h, w], stabilised [B, C, num_new, h, w]
        """
        H, P, M, Hr = self.H, self.P, self.M, self.Hr
        L = 1 + self.num_seeds + self.num_new
        first_new = 1 + self.num_seeds
        tokens, a = self.audio_tokens(code_tables, codes, L - 1)

        sink_k, sink_v = self.context_frame(gt[:, :, 0], None, 0.0, 0, None)
        sink = [[k, v] for k, v in zip(sink_k, sink_v)]
        pad_k, pad_v = self.context_frame(torch.zeros_like(gt[:, :, 0]), None, 0.0, H, sink, valid=False)
        stab_history = {}  # frame -> (K raw, V) per layer; frames < 1 are the pad
        res_history = {}
        latent = {}  # stabilising latents of frames in the present
        x_h = {}  # frames at history_t
        resolved, stabilised = {}, {}
        res_x = None
        passes = 0
        for r in range(1, L + 1):
            slots = [j for j in range(P) if r + j < L]  # valid present slots (frames r + j)
            if r + P - 1 < L:
                latent[r + P - 1] = noise[:, :, r + P - 1]
            g = r - 1  # left the present at history_t last round; resolving this round
            resolving = g >= first_new
            commit_a = 2 <= r <= L - 1
            commit_q = r >= first_new  # Q = frame r - 2, resolving history of frames r - 1 and r
            if g >= 1:
                x_h[g] = (1 - self.h) * gt[:, :, g] + self.h * noise[:, :, g] if g < first_new else latent.pop(g)
                if g >= first_new:
                    stabilised[g] = x_h[g]
            if resolving:
                res_x = self.a * x_h[g] + self.sigma * renoise[:, :, g]
            held_kv, held_v = {}, {}
            for p in range(M):
                evaluated = [j for j in slots if self.band_steps[j] == M or p == 0]
                frames, plan, specs = [], [], []
                ext = [[] for _ in self.blocks]

                def external(kv_per_layer, slot):
                    index = len(ext[0]) // 2
                    for i, (k, v) in enumerate(kv_per_layer):
                        ext[i] += [k, v]
                    return ('e', index, slot)

                sink_ref = external(sink, 0)
                a_index = len(evaluated) + resolving
                q_index = a_index + 1
                with_commits = p == 0 and (commit_a or commit_q)
                # present frames
                window = {}  # window position -> spec, history positions 1..H, present H+1+j
                for w in range(1, H + 1 if slots else 1):
                    fh = r - 1 - H + w
                    if w == H and with_commits and commit_a:
                        window[w] = ('f', a_index)
                    elif fh >= 1:
                        k, v = stab_history[fh]
                        window[w] = external(list(zip(k, v)), w)
                    else:
                        window[w] = external(list(zip(pad_k, pad_v)), w)
                for idx, j in enumerate(evaluated):
                    window[H + 1 + j] = ('f', idx)
                for j in slots:
                    if j not in evaluated:
                        window[H + 1 + j] = external(held_kv[j], H + 1 + j)
                stab_specs = []
                for idx, j in enumerate(evaluated):
                    fr = r + j
                    t = self.slot_t(j, p)
                    x = (1 - t) * gt[:, :, fr] + t * noise[:, :, fr] if fr < first_new else latent[fr]
                    frames.append((x, pose[:, :, fr], t, H + 1 + j, True))
                    stab_specs.append([sink_ref] + [window[w] for w in range(1 + j, 1 + H + j)] + [('f', idx)])
                if evaluated:
                    plan.append((list(range(len(evaluated))), stab_specs))
                if resolving:
                    t = self.t_start * (1 - p / M)
                    frames.append((res_x, pose[:, :, g], t, Hr + 1, True))
                    if with_commits and commit_q:
                        older = res_history[g - 2]
                        older_ref = external(list(zip(*older)), 1)
                        newer_ref = ('f', q_index)
                    else:
                        older, newer = res_history[g - 2], res_history[g - 1]
                        older_ref = external(list(zip(*older)), 1)
                        newer_ref = external(list(zip(*newer)), 2)
                    plan.append(([len(evaluated)], [[sink_ref, older_ref, newer_ref, ('f', len(evaluated))]]))
                heads = len(frames)
                if with_commits:
                    qf = max(g - 1, 0)
                    q_latent = gt[:, :, qf] if qf < first_new else resolved[qf]
                    frames.append((x_h[g] if g >= 1 else gt[:, :, 0], pose[:, :, max(g, 0)], self.h, H, True))
                    frames.append((q_latent, pose[:, :, qf], 0.0, Hr, True))
                    plan.append(([a_index, q_index], [[sink_ref, ('f', a_index)], [sink_ref, ('f', q_index)]]))
                audio = torch.stack([self.frame_audio(tokens, a, r + j) for j in evaluated]
                                    + ([self.frame_audio(tokens, a, g)] if resolving else []), dim=1) if heads else None
                velocity, ks, vs = self.forward_pass(frames, plan, ext, audio, heads)
                passes += 1
                for idx, j in enumerate(evaluated):
                    fr = r + j
                    if self.band_steps[j] == M:
                        if fr >= first_new:
                            latent[fr] = frames[idx][0] + velocity[:, :, idx] * (-self.width / M)
                    else:
                        held_kv[j] = [(ks[i][:, idx], vs[i][:, idx]) for i in range(len(self.blocks))]
                        held_v[j] = (frames[idx][0], velocity[:, :, idx])
                if resolving:
                    res_x = res_x + velocity[:, :, len(evaluated)] * (-self.t_start / M)
                if with_commits:
                    if commit_a:
                        stab_history[g] = ([k[:, a_index] for k in ks], [v[:, a_index] for v in vs])
                    if commit_q:
                        res_history[max(g - 1, 0)] = ([k[:, q_index] for k in ks], [v[:, q_index] for v in vs])
            for j, (x, v) in held_v.items():
                if r + j >= first_new:
                    latent[r + j] = x + v * (-self.width)
            if resolving:
                resolved[g] = res_x
            for fr in [k for k in res_history if k < g - 2]:
                del res_history[fr]
        order = range(first_new, L)
        return torch.stack([resolved[f] for f in order], dim=2), torch.stack([stabilised[f] for f in order], dim=2), passes


def main():
    import csv
    import yaml
    from safetensors.torch import save_file
    from bimodal.fused_rollout import task_inputs
    from bimodal.sampler import build_modes
    from sink_universal import load_chunk

    config_path, row, test, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    checkpoint_blocks = os.environ.get('CHECKPOINT', '1') == '1'
    config = yaml.safe_load(open(config_path))
    device = 'cuda'
    stab, res = config['stabilising'], config['resolving']
    stabilising, resolving = build_modes(config, Path(__file__).parent, device)
    rows = list(csv.DictReader(open(config['csv'])))
    clips = [load_chunk(os.path.join(config['root_dir'], rows[row]['paths']), config['pose_file'], 64, config['mimi_file'])]
    seeds = [config['seed'] + row]
    gt, pose, noise, renoise = task_inputs(clips, seeds, 3, 4, device)
    code_tables = [c[2].to(device) for c in clips]
    codes = [c[3].to(device) for c in clips]
    compile = os.environ.get('COMPILE', '0') == '1'
    if test == 'teacher':
        steps, resolve_steps = [stab['substeps']] * 4, res['num_steps']
    else:
        steps, resolve_steps = [2, 2, 2, 2], 2
    make = lambda c: StudentRollout(stabilising, clips[0][0].shape[2:], steps, resolve_steps, stab['history_t'],
                                    res['t_start'], checkpoint_blocks=checkpoint_blocks, compile=c)
    net = make(compile)
    os.makedirs(out, exist_ok=True)
    if test == 'teacher':
        with torch.no_grad():
            resolved, stabilised, passes = net.rollout(gt, pose, noise, renoise, code_tables, codes)
        full_r, full_s = gt.clone(), gt.clone()
        full_r[:, :, 4:], full_s[:, :, 4:] = resolved, stabilised
        save_file({'resolved': full_r.cpu().contiguous(), 'stabilised': full_s.cpu().contiguous()}, f'{out}/student_teacher_row{row}.safetensors')
        print(f'[{TAG}] teacher schedule: {passes} passes', flush=True)
        return
    rel = lambda a, b: ' '.join(f'{100 * ((a[:, :, i] - b[:, :, i]).norm() / b[:, :, i].norm()).item():.3f}' for i in range(a.shape[2]))
    with torch.no_grad():
        eager_resolved, _, _ = make(False).rollout(gt, pose, noise, renoise, code_tables, codes)
        for it in range(3):
            torch.cuda.synchronize()
            start = time.time()
            resolved, _, passes = net.rollout(gt, pose, noise, renoise, code_tables, codes)
            torch.cuda.synchronize()
            print(f'[{TAG}] no-grad rollout {it}: {passes} passes, {1000 * (time.time() - start):.0f} ms', flush=True)
    print(f'[{TAG}] compiled={compile} vs eager, resolved rel-L2 %: {rel(resolved, eager_resolved)}', flush=True)
    params = [p for p in stabilising.net.parameters() if p.is_floating_point()]
    for p in params:
        p.requires_grad_(True)
    print(f'[{TAG}] student schedule {steps} + {resolve_steps}, checkpoint={checkpoint_blocks}, compile={compile}, '
          f'{sum(p.numel() for p in params):,} trainable', flush=True)
    target = torch.randn(1, gt.shape[1], 4, *gt.shape[3:], device=device)
    for it in range(5):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        resolved, stabilised, passes = net.rollout(gt, pose, noise, renoise, code_tables, codes)
        torch.cuda.synchronize()
        forward = time.time() - start
        loss = (resolved * target).mean()
        loss.backward()
        torch.cuda.synchronize()
        total = time.time() - start
        grad = math.sqrt(sum(float(p.grad.float().square().sum()) for p in params if p.grad is not None))
        for p in params:
            p.grad = None
        print(f'[{TAG}] it {it}: {passes} passes, forward {forward * 1000:.0f} ms, forward+backward {total * 1000:.0f} ms, '
              f'peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, grad norm {grad:.3e}', flush=True)
    from torch._dynamo.utils import counters
    print(f'[{TAG}] dynamo unique graphs {counters["stats"]["unique_graphs"]}, calls {counters["stats"]["calls_captured"]}', flush=True)


if __name__ == '__main__':
    main()
