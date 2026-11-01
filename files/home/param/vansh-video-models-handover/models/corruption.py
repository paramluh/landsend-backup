import math

import torch
import torch.nn.functional as F

from models.mosaic import mosaic_pattern


# Unit per-channel directions (48 channels, in per-channel stds); positive = darker.
# DRIFT: background drift of the hdrop step 402 rollouts. DARK: VAE latent change of a pixel gain < 1.
DRIFT_DIRECTION = [
    0.1350, 0.0582, -0.2033, 0.2570, -0.1669, 0.0320, 0.0428, -0.1531,
    0.0885, 0.1645, -0.0972, -0.3216, 0.1931, -0.1232, 0.0909, 0.0109,
    -0.0531, -0.0740, 0.0718, 0.1994, 0.2095, -0.2331, -0.0526, 0.1129,
    0.1826, -0.1488, -0.0129, 0.1200, -0.1995, -0.1425, -0.2610, -0.0691,
    0.0275, -0.0949, -0.0626, 0.1443, 0.0030, -0.1548, -0.0661, 0.1938,
    -0.1344, -0.2326, 0.1005, 0.1659, -0.0452, -0.0363, 0.1539, -0.0324,
]
DARK_DIRECTION = [
    -0.0086, 0.2173, -0.0198, 0.1703, -0.1324, 0.0299, 0.1285, 0.0531,
    0.1699, 0.2323, -0.0196, -0.2310, -0.0321, 0.0961, 0.2272, -0.1451,
    -0.1511, -0.0588, -0.1401, 0.0850, 0.1132, -0.2911, -0.1831, -0.0606,
    0.1206, 0.0529, -0.0278, 0.0659, -0.0939, 0.1376, -0.2365, 0.0996,
    -0.0499, -0.1635, 0.1908, 0.1551, 0.1274, -0.2737, 0.0607, 0.2335,
    -0.0014, -0.2400, 0.1135, -0.0127, -0.0938, -0.0034, -0.1154, 0.1740,
]


def _log_uniform(bounds, n, device):
    low, high = bounds
    return torch.empty(n, device=device).uniform_(math.log(low), math.log(high)).exp()


def skeleton(pose, tau_pose):
    """[B, C, h, w] pose latent -> bool [B, h, w]: cells above the empty render (the frame's median) by tau_pose."""
    empty = pose.flatten(2).median(2).values
    dp = (pose - empty[..., None, None]).pow(2).mean(1).sqrt()
    return dp > dp.flatten(1).median(1).values[:, None, None] + tau_pose


def background_mask(frame, pose, tau, tau_pose):
    """[B, C, h, w] latent and pose latent of one frame -> [B, h, w] soft background weight. Wall = cells close
    to the frame's own top-corner mean (in per-channel spatial stds); person = the rest plus the pose skeleton,
    grown by a cell each and feathered by one cell."""
    scale = frame.std(dim=(2, 3)).clamp_min(1e-3)
    wall = torch.cat([frame[:, :, :4, :2], frame[:, :, :4, -2:]], dim=3).mean(dim=(2, 3))
    far = ((frame - wall[..., None, None]) / scale[..., None, None]).pow(2).mean(1).sqrt() > tau
    grown = F.max_pool2d(skeleton(pose, tau_pose)[:, None].float(), 3, 1, 1) > 0
    person = F.max_pool2d((far[:, None] | grown).float(), 3, 1, 1)
    return F.avg_pool2d(1 - person, 3, 1, 1, count_include_pad=False)[:, 0]


def pose_hull(pose, tau_pose, cap=0.75):
    """[B, C, h, w] pose latent -> bool [B, h, w] person, from the skeleton's per-row spans: head span = first
    skeleton row >= 4 cells wide (else the first), with a hair cap of max(3, cap x head width) rows above it; face
    rows (skeleton rows from the top, gaps <= 1 row) = union(head span, row span), empty rows among them the head
    span; every row after the face = the span of the last skeleton row at or above it."""
    sk = skeleton(pose, tau_pose)
    B, h, w = sk.shape
    i = torch.arange(h, device=sk.device)[None]
    j = torch.arange(w, device=sk.device)
    row = sk.any(2)
    has = row.any(1, keepdim=True)
    lo = torch.where(sk, j, w).min(2).values
    hi = torch.where(sk, j, -1).max(2).values
    first = row.int().argmax(1, keepdim=True)
    wide = row & (hi - lo >= 3)
    head = torch.where(wide.any(1, keepdim=True), wide.int().argmax(1, keepdim=True), first)
    hl, hr = lo.gather(1, head), hi.gather(1, head)
    last = torch.where(row, i, -1).cummax(1).values
    prev = torch.cat([last.new_full((B, 1), -1), last[:, :-1]], 1)
    gap = row & (prev >= 0) & (i - prev > 2)
    after = i >= torch.where(gap.any(1, keepdim=True), gap.int().argmax(1, keepdim=True), h)
    last = last.clamp_min(0)
    l = torch.where(after, lo.gather(1, last), torch.where(row, torch.minimum(lo, hl), hl))
    r = torch.where(after, hi.gather(1, last), torch.where(row, torch.maximum(hi, hr), hr))
    span = lambda a, b: (j >= a[..., None]) & (j <= b[..., None])
    body = ((i >= first) & has)[..., None] & span(l, r)
    hair = ((i >= head - (cap * (hr - hl + 1)).round().clamp_min(3)) & (i < head) & has)[..., None] & span(hl, hr)
    return body | hair


def pose_background_mask(pose, tau_pose):
    """[B, C, h, w] pose latent of one frame -> [B, h, w] soft background weight: 1 - pose_hull, grown by a cell and
    feathered by one cell; no colour test."""
    person = F.max_pool2d(pose_hull(pose, tau_pose)[:, None].float(), 3, 1, 1)
    return F.avg_pool2d(1 - person, 3, 1, 1, count_include_pad=False)[:, 0]


def _per_slot(fn, *frames):
    """fn on every slot of [B, C, S, h, w] tensors -> [S, B, h, w]."""
    B, _, S, h, w = frames[0].shape
    return fn(*(f.permute(2, 0, 1, 3, 4).flatten(0, 1) for f in frames)).view(S, B, h, w)


class HistoryCorruption:
    """Teaches the model not to copy a drifted history: the real history frames of a window get the same pattern d
    (sink, pads, present frames and pose are never touched), in per-channel stds of that window's real history:
      d = color * (a_person + mask * a_background) + mosaic, with exactly one of mosaic (probability mosaic_p) or
      color (probability color_p) per window, else clean;
      color:  a per-channel offset (the DC FFT bin), unit channel-RMS, along DARK / DRIFT / a random direction
              (probabilities `directions`), random sign; a_person ~ log-uniform(person_color) over the whole
              frame, a_background ~ log-uniform(background_color) added on the background;
      mask:   each slot's background_mask of its own clean frame and pose;
      mosaic: mosaic_pattern on each slot's pose_background_mask: a broadband field in the real mosaics' channel
              directions with non-DC channel-RMS a ~ log-uniform(mosaic), plus a DC of mosaic_dc * a along their
              mean colour direction, on the top fraction ~ U(mosaic_coverage) of that mask; the draw stays shared.
    slots = [full, leading, trailing]: which real history slots get d, all of them, the newest n or the oldest n,
    n ~ U{1, 2, 3}."""

    def __init__(self, config):
        self.color_p = config['color_p']
        self.mosaic_p = config['mosaic_p']
        self.mosaic = config['mosaic']
        self.mosaic_dc = config['mosaic_dc']
        self.mosaic_coverage = config['mosaic_coverage']
        self.person_color = config['person_color']
        self.background_color = config['background_color']
        self.directions = torch.tensor(config['directions'], dtype=torch.float)
        self.mask_tau = config['mask_tau']
        self.mask_tau_pose = config['mask_tau_pose']
        self.slots = torch.tensor(config['slots'], dtype=torch.float)
        self.measured = torch.tensor([DARK_DIRECTION, DRIFT_DIRECTION])

    @torch.no_grad()
    def __call__(self, latents, pose_latents, ofi, num_history):
        B, C, _, h, w = latents.shape
        device = latents.device
        history = latents[:, :, 1:1 + num_history]
        real = ofi[:, 1:1 + num_history] >= 1
        u = torch.rand(B, device=device)
        mosaic_on = u < self.mosaic_p
        color_on = ~mosaic_on & (u < self.mosaic_p + self.color_p)
        hit = (color_on | mosaic_on) & real.any(1)
        if not hit.any():
            return latents

        r = real.to(latents.dtype)[:, None, :, None, None]
        count = r.sum(dim=(2, 3, 4)).clamp_min(1) * h * w
        mean = (history * r).sum(dim=(2, 3, 4)) / count
        scale = (((history - mean[..., None, None, None]) ** 2 * r).sum(dim=(2, 3, 4)) / count).sqrt()
        pose = pose_latents[:, :, 1:1 + num_history]
        mask = _per_slot(lambda f, p: background_mask(f, p, self.mask_tau, self.mask_tau_pose), history, pose)

        choice = torch.multinomial(self.directions, B, replacement=True).to(device)
        direction = torch.randn(B, C, device=device)
        measured = choice < 2
        direction[measured] = self.measured.to(device)[choice[measured]]
        direction = direction / direction.norm(dim=1, keepdim=True)
        sign = torch.randint(0, 2, (B, 1), device=device) * 2 - 1
        color = color_on[:, None] * sign * math.sqrt(C) * direction
        a_person = _log_uniform(self.person_color, B, device)[:, None, None, None]
        a_background = _log_uniform(self.background_color, B, device)[:, None, None, None]

        m = mask.unsqueeze(-3)
        a = mosaic_on * _log_uniform(self.mosaic, B, device)
        coverage = torch.empty(B, device=device).uniform_(*self.mosaic_coverage)
        envelope_mask = _per_slot(lambda p: pose_background_mask(p, self.mask_tau_pose), pose)
        pattern = mosaic_pattern(envelope_mask, a, self.mosaic_dc * a, coverage)

        d = scale[..., None, None] * (color[..., None, None] * (a_person + m * a_background) + pattern)
        d = d.permute(1, 2, 0, 3, 4)
        selected = real & hit[:, None]
        profile = torch.multinomial(self.slots, B, replacement=True).to(device)[:, None]
        n = torch.randint(1, num_history, (B, 1), device=device)
        from_newest = real.flip(1).cumsum(1).flip(1)
        from_oldest = real.cumsum(1)
        selected &= (profile == 0) | ((profile == 1) & (from_newest <= n)) | ((profile == 2) & (from_oldest <= n))
        weight = selected.to(latents.dtype)
        latents[:, :, 1:1 + num_history] += weight[:, None, :, None, None] * d
        return latents
