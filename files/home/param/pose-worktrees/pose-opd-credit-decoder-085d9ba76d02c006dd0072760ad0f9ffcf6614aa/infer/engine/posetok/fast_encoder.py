# wan-vae-collection@588eaeb optimisations/encoder.py (trimmed: _FastEncoder/_FastStudent/FastElectronEncoder only, Muon/Proton/Kaon/TAE dropped)
"""
Fast Wan encoders: mathematically equivalent to the official streaming
1,4,4,... implementations, but parallel over time: the video runs in
macro-chunks (first 1+K frames, then K at a time) carrying only the causal
tail states between chunks. bf16 end to end (no autocast), channels_last_3d,
torch.compile with static per-chunk shapes. encode(videos) -> fp32 scaled
latents.
"""

import torch
import torch.nn.functional as F

from .vae import CACHE_T, ResidualBlock, patchify

_RESBLOCKS = (ResidualBlock,)


class _FastEncoder:

    def _finish(self, enc, chunk, use_compile, channels_last, compile_mode):
        assert chunk % 4 == 0 and chunk > 0
        self.chunk = chunk
        self.channels_last = channels_last

        torch.backends.cudnn.benchmark = True

        enc = enc.eval().requires_grad_(False)
        if channels_last:
            for p in list(enc.parameters()) + list(enc.buffers()):
                if p.dim() == 5:
                    p.data = p.data.contiguous(
                        memory_format=torch.channels_last_3d)
                elif p.dim() == 4:
                    p.data = p.data.contiguous(
                        memory_format=torch.channels_last)
        self.enc = enc

        self._run_fn = (torch.compile(self._run, dynamic=False,
                                      mode=compile_mode)
                        if use_compile else self._run)

    # -- one macro-chunk, parallel over time ---------------------------------

    def _run(self, x, caches):
        """x [B,C,t,h,w] prepped input; caches None (first chunk, causal
        zero-pad) or list of per-layer tail states. Returns (z, new_caches)."""
        enc = self.enc
        i = 0
        new_caches = []

        def cc(m, x):  # kernel-3 causal conv, streaming-cache equivalent
            nonlocal i
            new_caches.append(x[:, :, -CACHE_T:])
            c = caches[i] if caches is not None else None
            i += 1
            return m(x, c)

        def spatial(m, x):  # 2d op applied per frame, layout-preserving
            b, c, t, h, w = x.shape
            y = m(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
            return y.view(b, t, *y.shape[1:]).permute(0, 2, 1, 3, 4)

        def down3d(spatial_m, time_conv, x):
            nonlocal i
            y = spatial(spatial_m, x)
            new_caches.append(y[:, :, -1:])
            if caches is None:
                # windows (y0,y1,y2),(y2,y3,y4),... ; y0 passes through
                y = torch.cat([y[:, :, :1], time_conv(y)], 2)
            else:
                y = time_conv(torch.cat([caches[i], y], 2))
            i += 1
            return y

        def resample(m, x):
            if m.mode == "downsample3d":
                return down3d(m.resample, m.time_conv, x)
            return spatial(m.resample, x)

        def resblock(m, x):
            h = m.shortcut(x)  # identity or 1x1 conv (no temporal state)
            r = m.residual  # norm, silu, conv, norm, silu, dropout, conv
            y = cc(r[2], r[1](r[0](x)))
            y = cc(r[6], r[4](r[3](y)))
            return y + h

        x = cc(enc.conv1, x)
        x = self._stages(x, resblock, resample, down3d)
        for mod in enc.middle:
            x = resblock(mod, x) if isinstance(mod, _RESBLOCKS) else mod(x)
        x = cc(enc.head[2], enc.head[1](enc.head[0](x)))
        z = F.conv3d(x, self.qw, self.qb)
        return z, new_caches

    # -- public API -----------------------------------------------------------

    @torch.no_grad()
    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        """videos [B,3,T,H,W] in [-1,1], T = 1+4k -> latents [B,z,1+k,h,w]."""
        T = videos.shape[2]
        assert (T - 1) % 4 == 0, "T must be 1 + 4k"
        K = self.chunk

        x = self._prep(videos)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last_3d)

        # Chunks take their natural size (first 1+K, then K, tail = remainder;
        # all multiples of 4, so the causal-state chaining stays exact). Each
        # distinct chunk length compiles once per process.
        # Clones detach z and the tail states from torch.compile's cudagraph
        # memory pool (replays overwrite prior outputs); tiny vs chunk compute.
        with torch.autocast("cuda", enabled=False):
            torch.compiler.cudagraph_mark_step_begin()
            z, caches = self._run_fn(x[:, :, :min(1 + K, T)], None)
            zs = [z.clone()]
            caches = [c.clone() for c in caches]
            for lo in range(min(1 + K, T), T, K):
                torch.compiler.cudagraph_mark_step_begin()
                z, caches = self._run_fn(x[:, :, lo:min(lo + K, T)], caches)
                zs.append(z.clone())
                caches = [c.clone() for c in caches]
        z = torch.cat(zs, 2) if len(zs) > 1 else zs[0]
        return z.float()


class _FastStudent(_FastEncoder):
    """
    Student module (ElectronEncoder, weights loaded) -> chunked inference.
    The module's own quant conv is the scaled-latent head.
    """

    def __init__(self, net, dtype=torch.bfloat16, device: str = "cuda",
                 chunk: int = 32, use_compile: bool = True,
                 channels_last: bool = True, compile_mode: str = "default"):
        self.dtype = dtype
        self.device = device
        net = net.to(self.dtype).to(device)
        self.qw, self.qb = net.quant.weight, net.quant.bias
        self._finish(net, chunk, use_compile, channels_last, compile_mode)

    def _prep(self, videos):
        return patchify(videos.to(self.dtype), self.patch)

    def _stages(self, x, resblock, resample, down3d):
        for blk in self.blocks:
            sc = blk.shortcut(x) if hasattr(blk, "shortcut") else None
            for mod in blk.blocks:
                x = resblock(mod, x)
            if blk.downsample:
                x = down3d(blk.spatial, blk.time_conv, x)
            if sc is not None:
                x = x + sc
        return x

    @property
    def blocks(self):
        enc = self.enc
        return [enc.down0, enc.down1, enc.down2] + (
            [enc.down3] if hasattr(enc, "down3") else [])


class FastElectronEncoder(_FastStudent):
    """Wan 2.2 student: [B,3,1+4k,H,W] -> [B,48,1+k,H/16,W/16]."""
    patch = 4
