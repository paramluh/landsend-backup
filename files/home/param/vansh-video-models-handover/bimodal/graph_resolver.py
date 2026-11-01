"""Static-buffer CUDA graph for the unchanged separate teacher resolver.

Captures clean-context construction and all Euler steps. Audio window encoding
and selection remain outside capture, so sliding tables retain their semantics.
"""
import torch

from bimodal.cached_resolver import ResolverNet, context_frames
from utils.common import AUTOCAST_DTYPE


class GraphResolver(ResolverNet):
    def __init__(self, models, num_steps=10):
        super().__init__(models)
        if num_steps < 1:
            raise ValueError('At least one resolver step is required')
        self.num_steps = num_steps
        self.graph = None
        self.static_shape = None

    def _run(self):
        with torch.autocast('cuda', dtype=AUTOCAST_DTYPE, cache_enabled=False):
            context = self.context(self.cx, self.cp, self.ofi)
            x = self.x
            for k in range(self.num_steps):
                # Absolute f is used only for the present/non-pad condition;
                # RoPE uses the fixed present slot H+1, and audio is preselected.
                x = x + (self.ts[k+1]-self.ts[k]) * self.velocity(
                    x, self.p, 1, self.ts[k]*1000, context, self.at)
            self.result.copy_(x)

    @torch.no_grad()
    def frame(self, out, pose, codes, f, x, ts, tokens=None):
        if f < 1 or ts.numel() != self.num_steps + 1:
            raise ValueError('Graph resolver requires a generated frame and its complete schedule')
        if x.device.type != 'cuda':
            raise ValueError('Graph resolver requires CUDA')
        ofi = context_frames(f, self.H)
        index = [max(j, 0) for j in ofi]
        with torch.autocast('cuda', dtype=AUTOCAST_DTYPE):
            audio = self.audio(codes, f, tokens)
        if self.static_shape is None:
            self.static_shape = (tuple(x.shape), x.dtype, tuple(audio.shape), audio.dtype)
            self.cx = torch.empty_like(out[:, :, index])
            self.cp = torch.empty_like(pose[:, :, index])
            self.ofi = torch.empty((x.shape[0], self.H+1), device=x.device, dtype=torch.long)
            self.x, self.p = torch.empty_like(x), torch.empty_like(pose[:, :, f])
            self.at, self.ts = torch.empty_like(audio), torch.empty_like(ts)
            self.result = torch.empty_like(x)
        if self.static_shape != (tuple(x.shape), x.dtype, tuple(audio.shape), audio.dtype):
            raise ValueError('Captured resolver input shape or dtype changed')
        self.cx.copy_(out[:, :, index]); self.cp.copy_(pose[:, :, index])
        self.ofi.copy_(torch.tensor([ofi]*x.shape[0], device=x.device))
        self.x.copy_(x); self.p.copy_(pose[:, :, f])
        self.at.copy_(audio); self.ts.copy_(ts)
        if self.graph is None:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._run()
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._run()
        self.graph.replay()
        out[:, :, f].copy_(self.result)
