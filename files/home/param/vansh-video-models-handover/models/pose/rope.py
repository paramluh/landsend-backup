"""
Precomputed RoPE for the pose DiT blocks: the per-position multiplier for a single
uniform grid at frame_offset (0 unless refs precede the window), hoisted out of the
per-call loop in models.wan.model.rope_apply. Same math as rope_apply for a uniform
grid -- but a different grid with the same token count would be silently WRONG, not
just slow. See docs/benchmarking/round-2.md.

apply_rope is written in real arithmetic (cos/sin pairs) rather than complex multiply:
view_as_complex is a torch.compile graph break, which split every block into three
graphs and left the rotation and its casts eager. Results agree with the complex form
to fp32 rounding, below the bf16 cast that follows.
"""
import torch


def build_freqs_i(freqs, f, h, w):
    c = freqs.size(1)
    fs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    return torch.cat([
        fs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        fs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        fs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, c)


def build_freqs_block(freqs, t_positions, h, w):
    """
    Like build_freqs_i but with the temporal index chosen per frame from
    `t_positions` (a LongTensor), so heterogeneous-resolution blocks can be
    concatenated along the token axis. Bit-identical to build_freqs_i when
    t_positions == arange(f).
    """
    c = freqs.size(1)
    fs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f = t_positions.numel()
    temporal = fs[0][t_positions].view(f, 1, 1, -1).expand(f, h, w, -1)
    height = fs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1)
    width = fs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    return torch.cat([temporal, height, width], dim=-1).reshape(f * h * w, 1, c)


def cached_freqs_i(cache, grid, freqs, frame_offset=0):
    """
    Cached on the (f, h, w) grid, plus frame_offset when the grid's first frame is at
    temporal position frame_offset != 0 (host-side dict, no device sync). Interleaved
    resolutions have distinct grids, so each builds its multiplier once and all stay
    resident. Stored as real [s, 1, c, 2] (cos, sin) for apply_rope.
    """
    key = (*grid, frame_offset) if frame_offset else grid
    freqs_i = cache.get(key)
    if freqs_i is None:
        if frame_offset:
            f, h, w = grid
            freqs_i = build_freqs_block(freqs, torch.arange(frame_offset, frame_offset + f, device=freqs.device), h, w)
        else:
            freqs_i = build_freqs_i(freqs, *grid)
        freqs_i = torch.view_as_real(freqs_i).contiguous()
        cache[key] = freqs_i
    return freqs_i


def apply_rope(x, freqs_i):
    """
    x [B, s, n, d] -> RoPE-rotated fp32 against the precomputed [s, 1, c, 2] (cos, sin)
    freqs_i, pairing adjacent channels as (re, im).
    """
    x = x.float().unflatten(-1, (-1, 2))
    cos, sin = freqs_i[..., 0], freqs_i[..., 1]
    re = x[..., 0] * cos - x[..., 1] * sin
    im = x[..., 0] * sin + x[..., 1] * cos
    return torch.stack((re, im), dim=-1).flatten(3)
