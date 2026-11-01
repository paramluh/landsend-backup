"""
Bimodal DMD pieces: the student rollout, the two scoring layouts, and the losses.

Rows are [sink | GT 1..3 | targets 4..7]: the student stabilises frames 4..7 from GT seeds 1..3 with the fused
schedule (bimodal/student_rollout.py: every pass carries all the work available, no pure prefills) and, unless
stab_only, resolves them. The generator output y lives at level t0: t0 = history_t (0.2) for the stabilised frames,
t0 = 0 for the resolved ones. Scoring noises y to t > t0 with a fresh draw, along

    x_t = a y + s eps,   a = (1 - t) / (1 - t0),   s = sqrt(t^2 - (t0 a)^2),

which has the rectified-flow marginal at t whenever y has it at t0 (y = (1 - t0) x0 + t0 n gives
x_t = (1 - t) x0 + (t0 a n + s eps), noise variance t^2). Its path velocity v* = a' y + s' eps,
a' = -1 / (1 - t0), s' = (t + t0^2 a / (1 - t0)) / s, regresses onto the same field as FM's eps - x0
(E[v* | x_t] = E[eps_total - x0 | x_t]); t0 = 0 gives v* = eps - y exactly. So the critic's FM target is v*, and the
universal velocity DMD carries over unchanged: g = (vT - vF) / mean|vT - v*|, L = 0.5 |y - sg(y - g)|^2, which for
t0 = 0.2 equals DMD's normalized x_t0-space gradient (E_F[y | x_t] - E_T[y | x_t]) / mean|y - E_T[y | x_t]|.
"""

import torch

from distillation.bimodal_net import BimodalNet

SINK_SLOT, H, P, HR = 0, 3, 4, 2


# ---- noising path ----
def path(t0, t):
    a = (1 - t) / (1 - t0)
    s = (t.square() - (t0 * a).square()).clamp_min(1e-12).sqrt()
    da = torch.full_like(t, -1 / (1 - t0))
    ds = (t + t0 * t0 * a / (1 - t0)) / s
    return a, s, da, ds


def noised(y, t0, t, eps):
    """y [B, C, F, h, w], t [B, F] -> x_t, v*"""
    a, s, da, ds = (c[:, None, :, None, None] for c in path(t0, t))
    y = y.detach().float()
    return a * y + s * eps, da * y + ds * eps


def stab_times(B, device, t_min, t_max, history_t=0.2):
    """Present slot j uniform in band j of (history_t, 1], intersected with [t_min, t_max]."""
    width = (1 - history_t) / P
    lower = (history_t + width * torch.arange(P, device=device)).clamp_min(t_min)
    upper = (history_t + width * torch.arange(1, P + 1, device=device)).clamp_max(t_max)
    q = 1 - torch.rand(B, P, device=device)
    return lower + q * (upper - lower)


def resolve_times(B, device, t_min, t_max):
    return t_min + (1 - torch.rand(B, P, device=device)) * (t_max - t_min)


def dmd_loss(y, v_teacher, v_critic, v_star, epsilon):
    """-> loss (mean over frames and rows), per-frame loss [B, F], normalizer [B, F], |vT - vF| [B, F]"""
    y = y.float()
    with torch.no_grad():
        vt, vf = v_teacher.float(), v_critic.float()
        normalizer = (vt - v_star).abs().mean(dim=(1, 3, 4))
        gradient = (vt - vf) / normalizer.clamp_min(epsilon)[:, None, :, None, None]
        target = y - gradient
        difference = (vt - vf).abs().mean(dim=(1, 3, 4))
    frames = 0.5 * (y - target).square().mean(dim=(1, 3, 4))
    return frames.mean(), frames.detach(), normalizer, difference


def fm_loss(v_critic, v_star):
    frames = (v_critic.float() - v_star).square().mean(dim=(1, 3, 4))
    return frames.mean(), frames.detach()


# ---- scoring layouts ----
def stab_pass(net: BimodalNet, rows, x_present, t_present, history, audio, audio_mask):
    """[sink | 3 history at history_t | 4 present], the stabilising training window. history [B, C, 3, h, w] (noised)."""
    gt, pose = rows['latents'], rows['pose']
    frames, present_specs, history_specs = [], [], []
    for j in range(P):
        frames.append((x_present[:, :, j], pose[:, :, 4 + j], t_present[:, j], H + 1 + j, True))
    for w in range(1, H + 1):
        frames.append((history[:, :, w - 1], pose[:, :, w], rows['history_t'], H, True))
    frames.append((gt[:, :, 0], None, 0.0, SINK_SLOT, True))
    sink = 2 * H + 1
    for j in range(P):
        left = [('r', P + w - 1, w) if w <= H else ('f', w - H - 1) for w in range(1 + j, 1 + H + j)]
        present_specs.append([('f', sink)] + left + [('f', j)])
    for w in range(1, H + 1):
        history_specs.append([('f', sink), ('f', P + w - 1)])
    plan = [(0, present_specs), (P, history_specs), (sink, [[('f', sink)]])]
    velocity, _, _ = net.forward_pass(frames, plan, None, audio, audio_mask, P)
    return velocity


def resolve_pass(net: BimodalNet, rows, x_present, t_present, clean, audio, audio_mask):
    """The four resolving windows [sink | f-2 f-1 clean | f] packed into one pass: sink and the five distinct
    history frames once each. clean [B, C, 5, h, w]: frames 2..6 at t = 0 (GT, then the generated ones)."""
    gt, pose = rows['latents'], rows['pose']
    frames = [(x_present[:, :, j], pose[:, :, 4 + j], t_present[:, j], HR + 1, True) for j in range(P)]
    frames += [(clean[:, :, i], pose[:, :, 2 + i], 0.0, HR, True) for i in range(5)]
    sink = P + 5
    frames.append((gt[:, :, 0], None, 0.0, SINK_SLOT, True))
    present_specs = [[('f', sink), ('r', P + j, 1), ('r', P + j + 1, 2), ('f', j)] for j in range(P)]
    history_specs = [[('f', sink), ('f', P + i)] for i in range(5)]
    plan = [(0, present_specs), (P, history_specs), (sink, [[('f', sink)]])]
    velocity, _, _ = net.forward_pass(frames, plan, None, audio, audio_mask, P)
    return velocity


def both_pass(net: BimodalNet, rows, x_stab, t_stab, history, x_res, t_res, clean, audio, audio_mask):
    """stab_pass and resolve_pass in one pass of 17 frames (one shared sink): [stab present 4 | resolving present 4 |
    stab history 3 | resolving history 5 | sink]. audio [B, 8, K+1, D]. -> (stab velocity, resolving velocity)"""
    gt, pose = rows['latents'], rows['pose']
    frames = [(x_stab[:, :, j], pose[:, :, 4 + j], t_stab[:, j], H + 1 + j, True) for j in range(P)]
    frames += [(x_res[:, :, j], pose[:, :, 4 + j], t_res[:, j], HR + 1, True) for j in range(P)]
    frames += [(history[:, :, w - 1], pose[:, :, w], rows['history_t'], H, True) for w in range(1, H + 1)]
    frames += [(clean[:, :, i], pose[:, :, 2 + i], 0.0, HR, True) for i in range(5)]
    sink = 2 * P + H + 5
    frames.append((gt[:, :, 0], None, 0.0, SINK_SLOT, True))
    sh, rh = 2 * P, 2 * P + H  # first stab / resolving history frame
    stab_specs = []
    for j in range(P):
        left = [('r', sh + w - 1, w) if w <= H else ('f', w - H - 1) for w in range(1 + j, 1 + H + j)]
        stab_specs.append([('f', sink)] + left + [('f', j)])
    res_specs = [[('f', sink), ('r', rh + j, 1), ('r', rh + j + 1, 2), ('f', P + j)] for j in range(P)]
    history_specs = [[('f', sink), ('f', i)] for i in range(sh, sink)]
    plan = [(0, stab_specs), (P, res_specs), (sh, history_specs), (sink, [[('f', sink)]])]
    velocity, _, _ = net.forward_pass(frames, plan, None, audio, audio_mask, 2 * P)
    return velocity[:, :, :P], velocity[:, :, P:]


# ---- the student rollout ----
class Rollout:
    """
    bimodal/student_rollout.py's fused schedule over BimodalNet: GT seeds 1..H noised into the stabilising present
    along the band schedule (fixed per-frame noise), band_steps[j] Euler steps in band j (1 or M), resolve_steps = M
    resolving steps per frame, commits riding on the first pass of each round, right pads never evaluated.
    """

    def __init__(self, band_steps, resolve_steps, history_t, t_start):
        self.M = max(band_steps)
        assert len(band_steps) == P and all(k in (1, self.M) for k in band_steps) and resolve_steps == self.M
        self.band_steps, self.h, self.t_start = list(band_steps), history_t, t_start
        self.width = (1 - history_t) / P
        self.a = (1 - t_start) / (1 - history_t)
        self.sigma = (t_start ** 2 - (self.a * history_t) ** 2) ** 0.5

    def slot_t(self, j, p):
        return self.h + self.width * (j + 1) - p * self.width / self.band_steps[j]

    def __call__(self, net: BimodalNet, rows, tokens, noise, renoise, stab_only):
        """
        rows: latents / pose [B, C, 8, h, w], ofi [B, 8], drop [B]; noise / renoise [B, C, 8, h, w] (generation).
        -> stabilised [B, C, 4, h, w] (frames 4..7 at history_t), resolved [B, C, 4, h, w] or None
        """
        M, layers = self.M, len(net.transformer.blocks)
        gt, pose, ofi, drop = rows['latents'], rows['pose'], rows['ofi'], rows['drop']
        L, first_new = 1 + H + P, 1 + H
        B = gt.shape[0]

        _, ks, vs = net.forward_pass([(gt[:, :, 0], None, 0.0, SINK_SLOT, True)], [(0, [[('f', 0)]])], None, None, None, 0,
                                     keep=(0,))
        sink = [(ks[i][:, 0], vs[i][:, 0]) for i in range(layers)]
        # History slots before frame 1 (the seeds start in the present): the pad token at the anchor, t = 0.
        _, ks, vs = net.forward_pass([(torch.zeros_like(gt[:, :, 0]), None, 0.0, H, False)],
                                     [(0, [[('e', 0, SINK_SLOT), ('f', 0)]])],
                                     [[k, v] for k, v in sink], None, None, 0, keep=(0,))
        pad = [(ks[i][:, 0], vs[i][:, 0]) for i in range(layers)]
        stab_history, res_history, latent, x_h = {}, {}, {}, {}
        resolved, stabilised = {}, {}
        res_x = None
        last_round = L - 1 if stab_only else L
        for r in range(1, last_round + 1):
            slots = [j for j in range(P) if r + j < L]
            if r + P - 1 < L:
                latent[r + P - 1] = noise[:, :, r + P - 1]
            g = r - 1
            resolving = g >= first_new and not stab_only
            commit_a = 2 <= r <= L - 1
            commit_q = r >= first_new and not stab_only
            if g >= 1:
                x_h[g] = (1 - self.h) * gt[:, :, g] + self.h * noise[:, :, g] if g < first_new else latent.pop(g)
                if g >= first_new:
                    stabilised[g] = x_h[g]
            if resolving:
                res_x = self.a * x_h[g] + self.sigma * renoise[:, :, g]
            held_kv, held_v = {}, {}
            for p in range(M):
                evaluated = [j for j in slots if self.band_steps[j] == M or p == 0]
                frames = []
                ext = [[] for _ in range(layers)]

                def external(kv, slot):
                    index = len(ext[0]) // 2
                    for i, (k, v) in enumerate(kv):
                        ext[i] += [k, v]
                    return ('e', index, slot)

                sink_ref = external(sink, 0)
                with_commits = p == 0 and (commit_a or commit_q)
                a_index = len(evaluated) + resolving
                q_index = a_index + commit_a
                window = {}
                if slots:
                    for w in range(1, H + 1):
                        fh = r - 1 - H + w
                        if w == H and with_commits and commit_a:
                            window[w] = ('f', a_index)
                        else:
                            window[w] = external(list(zip(*stab_history[fh])) if fh >= 1 else pad, w)
                for idx, j in enumerate(evaluated):
                    window[H + 1 + j] = ('f', idx)
                for j in slots:
                    if j not in evaluated:
                        window[H + 1 + j] = external(held_kv[j], H + 1 + j)
                plan = []
                stab_specs = []
                for idx, j in enumerate(evaluated):
                    fr = r + j
                    t = self.slot_t(j, p)
                    x = (1 - t) * gt[:, :, fr] + t * noise[:, :, fr] if fr < first_new else latent[fr]
                    frames.append((x, pose[:, :, fr], t, H + 1 + j, True))
                    stab_specs.append([sink_ref] + [window[w] for w in range(1 + j, 1 + H + j)] + [('f', idx)])
                if evaluated:
                    plan.append((0, stab_specs))
                if resolving:
                    t = self.t_start * (1 - p / M)
                    frames.append((res_x, pose[:, :, g], t, HR + 1, True))
                    older = external(list(zip(*res_history[g - 2])), 1)
                    newer = ('f', q_index) if with_commits and commit_q else external(list(zip(*res_history[g - 1])), 2)
                    plan.append((len(evaluated), [[sink_ref, older, newer, ('f', len(evaluated))]]))
                heads = len(frames)
                commit_specs = []
                if with_commits and commit_a:
                    frames.append((x_h[g], pose[:, :, g], self.h, H, True))
                    commit_specs.append([sink_ref, ('f', a_index)])
                if with_commits and commit_q:
                    qf = g - 1
                    frames.append((gt[:, :, qf] if qf < first_new else resolved[qf], pose[:, :, qf], 0.0, HR, True))
                    commit_specs.append([sink_ref, ('f', q_index)])
                if commit_specs:
                    plan.append((heads, commit_specs))
                head_frames = [r + j for j in evaluated] + ([g] if resolving else [])
                audio = net.frame_audio(tokens, ofi[:, head_frames], drop) if heads else None
                audio_mask = torch.ones(B, heads, dtype=torch.bool, device=gt.device) if heads else None
                keep = [idx for idx, j in enumerate(evaluated) if self.band_steps[j] != M]
                keep += [a_index] if with_commits and commit_a else []
                keep += [q_index] if with_commits and commit_q else []
                slot_of = {frame: position for position, frame in enumerate(keep)}
                velocity, ks, vs = net.forward_pass(frames, plan, ext, audio, audio_mask, heads, keep=keep)
                for idx, j in enumerate(evaluated):
                    fr = r + j
                    if self.band_steps[j] == M:
                        if fr >= first_new:
                            latent[fr] = frames[idx][0] + velocity[:, :, idx] * (-self.width / M)
                    else:
                        held_kv[j] = [(ks[i][:, slot_of[idx]], vs[i][:, slot_of[idx]]) for i in range(layers)]
                        held_v[j] = (frames[idx][0], velocity[:, :, idx])
                if resolving:
                    res_x = res_x + velocity[:, :, len(evaluated)] * (-self.t_start / M)
                if with_commits and commit_a:
                    stab_history[g] = ([k[:, slot_of[a_index]] for k in ks], [v[:, slot_of[a_index]] for v in vs])
                if with_commits and commit_q:
                    res_history[g - 1] = ([k[:, slot_of[q_index]] for k in ks], [v[:, slot_of[q_index]] for v in vs])
            for j, (x, v) in held_v.items():
                if r + j >= first_new:
                    latent[r + j] = x + v * (-self.width)
            if resolving:
                resolved[g] = res_x
            for fr in [k for k in res_history if k < g - 2]:
                del res_history[fr]
            for fr in [k for k in stab_history if k < r - H]:
                del stab_history[fr]
        order = range(first_new, L)
        if stab_only:  # the last frame leaves band 0 at the end of the last round
            stabilised[L - 1] = latent.pop(L - 1)
        y02 = torch.stack([stabilised[f] for f in order], dim=2)
        y0 = None if stab_only else torch.stack([resolved[f] for f in order], dim=2)
        return y02, y0

    def passes(self, stab_only):
        L = 1 + H + P
        return self.M * ((L - 1) if stab_only else L)


class UniformRollout(Rollout):
    """
    The same schedule with one static pass layout: [4 stabilising slots | resolving frame | A | Q] every pass, empty
    slots as pads (t = 0, pad token, no audio; outputs discarded). A (the newest stabilising history frame) and Q (the
    newest resolved frame) are recomputed every pass rather than read from the cache, so every pass reads the same
    externals: sink, stabilising history slots 1 and 2, the older resolving history frame. band_steps must be flat.
    """

    def __call__(self, net: BimodalNet, rows, tokens, noise, renoise, stab_only):
        M, layers = self.M, len(net.transformer.blocks)
        assert all(k == M for k in self.band_steps), 'the uniform layout needs flat band steps'
        gt, pose, ofi, drop = rows['latents'], rows['pose'], rows['ofi'], rows['drop']
        L, first_new = 1 + H + P, 1 + H
        B, device = gt.shape[0], gt.device
        zero = torch.zeros_like(gt[:, :, 0])
        _, ks, vs = net.forward_pass([(gt[:, :, 0], None, 0.0, SINK_SLOT, True)], [(0, [[('f', 0)]])], None, None, None, 0,
                                     keep=(0,))
        sink = [(ks[i][:, 0], vs[i][:, 0]) for i in range(layers)]
        _, ks, vs = net.forward_pass([(zero, None, 0.0, H, False)], [(0, [[('e', 0, SINK_SLOT), ('f', 0)]])],
                                     [[k, v] for k, v in sink], None, None, 0, keep=(0,))
        pad = [(ks[i][:, 0], vs[i][:, 0]) for i in range(layers)]
        stab_history, res_history, latent, x_h, resolved, stabilised = {}, {}, {}, {}, {}, {}
        res_x = zero
        R, A, Q = P, P + 1, P + 2
        specs = [[('e', 0, SINK_SLOT)] + [('e', w, w) if w < H else ('f', A) if w == H else ('f', w - H - 1)
                                          for w in range(1 + j, 1 + H + j)] + [('f', j)] for j in range(P)]
        plan = [(0, specs), (R, [[('e', 0, SINK_SLOT), ('e', 3, 1), ('f', Q), ('f', R)]]),
                (A, [[('e', 0, SINK_SLOT), ('f', A)], [('e', 0, SINK_SLOT), ('f', Q)]])]
        last_round = L - 1 if stab_only else L
        for r in range(1, last_round + 1):
            if r + P - 1 < L:
                latent[r + P - 1] = noise[:, :, r + P - 1]
            g = r - 1
            resolving = g >= first_new and not stab_only
            if g >= 1:
                x_h[g] = (1 - self.h) * gt[:, :, g] + self.h * noise[:, :, g] if g < first_new else latent.pop(g)
                if g >= first_new:
                    stabilised[g] = x_h[g]
            if resolving:
                res_x = self.a * x_h[g] + self.sigma * renoise[:, :, g]
            older = [stab_history.get(r - 1 - H + w, pad) for w in (1, 2)]
            res_older = res_history.get(g - 2, pad)
            ext = [[sink[i][0], sink[i][1], older[0][i][0], older[0][i][1], older[1][i][0], older[1][i][1],
                    res_older[i][0], res_older[i][1]] for i in range(layers)]
            qf = g - 1
            q_valid = qf >= 1 and (qf < first_new or qf in resolved)  # stab_only: never resolved, a pad
            q_latent = (gt[:, :, qf] if qf < first_new else resolved[qf]) if q_valid else zero
            a_latent = x_h[g] if g >= 1 else zero
            for p in range(M):
                frames = []
                for j in range(P):
                    fr, t = r + j, self.slot_t(j, p)
                    if fr >= L:
                        frames.append((zero, None, 0.0, H + 1 + j, False))
                    else:
                        x = (1 - t) * gt[:, :, fr] + t * noise[:, :, fr] if fr < first_new else latent[fr]
                        frames.append((x, pose[:, :, fr], t, H + 1 + j, True))
                frames.append((res_x, pose[:, :, max(g, 0)] if resolving else None,
                               self.t_start * (1 - p / M) if resolving else 0.0, HR + 1, resolving))
                frames.append((a_latent, pose[:, :, g] if g >= 1 else None, self.h if g >= 1 else 0.0, H, g >= 1))
                frames.append((q_latent, pose[:, :, qf] if q_valid else None, 0.0, HR, q_valid))
                head_frames = [min(r + j, L - 1) for j in range(P)] + [max(g, 0)]
                audio = net.frame_audio(tokens, ofi[:, head_frames], drop)
                audio_mask = torch.tensor([r + j < L for j in range(P)] + [resolving], device=device)[None].expand(B, -1)
                velocity, ks, vs = net.forward_pass(frames, plan, ext, audio, audio_mask, P + 1, keep=(A, Q))
                for j in range(P):
                    fr = r + j
                    if first_new <= fr < L:
                        latent[fr] = frames[j][0] + velocity[:, :, j] * (-self.width / M)
                if resolving:
                    res_x = res_x + velocity[:, :, R] * (-self.t_start / M)
                if p == 0:
                    if g >= 1:
                        stab_history[g] = [(k[:, 0], v[:, 0]) for k, v in zip(ks, vs)]
                    if q_valid:
                        res_history[qf] = [(k[:, 1], v[:, 1]) for k, v in zip(ks, vs)]
            if resolving:
                resolved[g] = res_x
            for fr in [k for k in res_history if k < g - 2]:
                del res_history[fr]
            for fr in [k for k in stab_history if k < r - H]:
                del stab_history[fr]
        order = range(first_new, L)
        if stab_only:
            stabilised[L - 1] = latent.pop(L - 1)
        y02 = torch.stack([stabilised[f] for f in order], dim=2)
        y0 = None if stab_only else torch.stack([resolved[f] for f in order], dim=2)
        return y02, y0


class UniformStabilizerRollout(Rollout):
    """UniformRollout's stabiliser with static [4 present | A] passes; resolving-only R/Q work is omitted."""

    def __call__(self, net: BimodalNet, rows, tokens, noise, renoise, stab_only):
        assert stab_only, 'uniform_stab only supports stabilizer training'
        M, layers = self.M, len(net.transformer.blocks)
        assert all(k == M for k in self.band_steps), 'the uniform layout needs flat band steps'
        gt, pose, ofi, drop = rows['latents'], rows['pose'], rows['ofi'], rows['drop']
        L, first_new = 1 + H + P, 1 + H
        B, device = gt.shape[0], gt.device
        zero = torch.zeros_like(gt[:, :, 0])
        _, ks, vs = net.forward_pass([(gt[:, :, 0], None, 0.0, SINK_SLOT, True)], [(0, [[('f', 0)]])], None, None, None, 0,
                                     keep=(0,))
        sink = [(ks[i][:, 0], vs[i][:, 0]) for i in range(layers)]
        _, ks, vs = net.forward_pass([(zero, None, 0.0, H, False)], [(0, [[('e', 0, SINK_SLOT), ('f', 0)]])],
                                     [[k, v] for k, v in sink], None, None, 0, keep=(0,))
        pad = [(ks[i][:, 0], vs[i][:, 0]) for i in range(layers)]
        history, latent, stabilised = {}, {}, {}
        A = P
        specs = [[('e', 0, SINK_SLOT)] + [('e', w, w) if w < H else ('f', A) if w == H else ('f', w - H - 1)
                                          for w in range(1 + j, 1 + H + j)] + [('f', j)] for j in range(P)]
        plan = [(0, specs), (A, [[('e', 0, SINK_SLOT), ('f', A)]])]
        for r in range(1, L):
            if r + P - 1 < L:
                latent[r + P - 1] = noise[:, :, r + P - 1]
            g = r - 1
            a_latent = zero
            if g >= 1:
                a_latent = (1 - self.h) * gt[:, :, g] + self.h * noise[:, :, g] if g < first_new else latent.pop(g)
                if g >= first_new:
                    stabilised[g] = a_latent
            older = [history.get(r - 1 - H + w, pad) for w in (1, 2)]
            ext = [[sink[i][0], sink[i][1], older[0][i][0], older[0][i][1], older[1][i][0], older[1][i][1]]
                   for i in range(layers)]
            head_frames = [min(r + j, L - 1) for j in range(P)]
            audio = net.frame_audio(tokens, ofi[:, head_frames], drop)
            audio_mask = torch.tensor([r + j < L for j in range(P)], device=device)[None].expand(B, -1)
            for p in range(M):
                frames = []
                for j in range(P):
                    fr, t = r + j, self.slot_t(j, p)
                    if fr >= L:
                        frames.append((zero, None, 0.0, H + 1 + j, False))
                    else:
                        x = (1 - t) * gt[:, :, fr] + t * noise[:, :, fr] if fr < first_new else latent[fr]
                        frames.append((x, pose[:, :, fr], t, H + 1 + j, True))
                frames.append((a_latent, pose[:, :, g] if g >= 1 else None, self.h if g >= 1 else 0.0, H, g >= 1))
                velocity, ks, vs = net.forward_pass(frames, plan, ext, audio, audio_mask, P, keep=(A,))
                for j in range(P):
                    fr = r + j
                    if first_new <= fr < L:
                        latent[fr] = frames[j][0] + velocity[:, :, j] * (-self.width / M)
                if p == 0 and g >= 1:
                    history[g] = [(k[:, 0], v[:, 0]) for k, v in zip(ks, vs)]
            for fr in [k for k in history if k < r - H]:
                del history[fr]
        stabilised[L - 1] = latent.pop(L - 1)
        return torch.stack([stabilised[f] for f in range(first_new, L)], dim=2), None
