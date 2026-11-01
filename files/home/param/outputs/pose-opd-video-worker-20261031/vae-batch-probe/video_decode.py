"""Batch-preserving adapter for Amu's fixed-shape FP32 streaming Wan VAE.

The native implementation specializes its buffers and output to batch one.
This adapter changes those shapes only; weights, precision, causal cache
updates, first-frame path and CUDA-graph schedule remain the same.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

import fast_decode as native


NATIVE_SHA256 = '588c9c908b6bf65568b1cf6251a4505ce9fa7e2ec8764af8e3c4fe7788dc2d7b'


def postprocess(y):
    """Preserve batch: [B,12,T,h,w] -> uint8 [B,T,2h,2w,3]."""
    b, c4, t, h, w = y.shape
    c = c4 // 4
    x = y.float().reshape(b, c, 2, 2, t, h, w).permute(0, 1, 4, 5, 3, 6, 2)
    x = x.reshape(b, c, t, 2 * h, 2 * w)
    return x.clamp(-1, 1).add(1).div(2).mul(255).to(torch.uint8).permute(0, 2, 3, 4, 1).contiguous()


class _BatchedSteady(native._Steady):
    def _cc(self, k, name, a):
        # Discover exactly the native buffer shapes, retaining the batch axis.
        # Subsequent eager/compiled calls use the original native causal conv.
        if not self.alloc:
            return super()._cc(k, name, a)
        for buffers in self.bufs:
            assert len(buffers) == k[0]
            buffers.append(torch.zeros(
                a.shape[0], a.shape[1], a.shape[2] + 2, a.shape[3], a.shape[4],
                dtype=self.W[name][0].dtype, device=a.device).contiguous(memory_format=self.mf))
        self.alloc = False
        try:
            return super()._cc(k, name, a)
        finally:
            self.alloc = True

    def step(self, z, p=0):
        # This is native._Steady.step with only its final batch-preserving
        # postprocess changed. All arithmetic helpers are inherited unchanged.
        k = [0, p]
        x = z / self.s1 + self.s0
        w, b, _ = self.W['conv2']
        x = F.conv3d(x, w, b)
        x = self._cc(k, 'decoder.conv1', x.to(self.dt))
        x = self._res('decoder.middle.0', x, k, False)
        x = self._attn('decoder.middle.1', x)
        x = self._res('decoder.middle.2', x, k, False)
        for subs, du in self.ups:
            xm = x
            for kind, n, has_sc in subs:
                xm = self._res(n, xm, k, has_sc) if kind == 'res' else self._resample(kind, n, xm, k)
            x = xm if du is None else xm.to(self.res_dt) + self._dupup(du, x).to(self.res_dt)
        a = self._act(x, 'decoder.head.0', self.head_dt)
        return postprocess(self._cc(k, 'decoder.head.2', a))


class BatchedStreamDecoder(native.StreamDecoder):
    """Fixed B streaming decoder; push([B,48,h,w]) returns [B,n,H,W,3].

    n=1 at clip start and n=4 thereafter. A single pinned host buffer holds
    one four-frame chunk per row; complete movies never accumulate on GPU.
    Native FP32 precision is retained, including the original first chunk.
    """

    def __init__(self, vae, batch_size=4, *, compile=True, graph=True,
                 latent_hw=(30, 16), compile_mode='default', log=None):
        if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() != NATIVE_SHA256:
            raise ValueError('Batched VAE adapter requires the audited native decoder version')
        if batch_size < 1:
            raise ValueError('Decoder batch size must be positive')
        self.batch_size = batch_size
        self.vae, self.model = vae, vae.model
        self.mode, self.host = 'fp32', 'pinned'
        self.device, self.scale = self.model.conv2.weight.device, vae.scale
        self.benchmark = True
        self.graph, self.last_gpu_ms, self._stage = None, None, None
        self._e0 = torch.cuda.Event(enable_timing=True)
        self._e1 = torch.cuda.Event(enable_timing=True)
        self._e2 = torch.cuda.Event()
        self.log = log or (lambda *a: None)
        self.reset()
        self.st = _BatchedSteady(self.model, self.scale, torch.float32, channels_last=True,
                                 head_dtype=torch.float32, res_fp32=True, subpixel=True,
                                 pingpong=True, gemm_max_m=0)
        self.np = 2
        self.z = torch.zeros(batch_size, self.model.z_dim, 1, *latent_hw, device=self.device)
        with torch.no_grad(), native._cudnn_benchmark(True), torch.cuda.device(self.device):
            self.st.alloc = True
            self.st.step(self.z)
            self.st.alloc = False
            assert len(self.st.bufs[0]) <= self.st.n_cache
            fn = self.st.step
            if compile:
                fn = torch.compile(fn, fullgraph=True, dynamic=False, mode=compile_mode)
            self.fn = fn
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(3):
                    for parity in range(2):
                        fn(self.z, parity)
            torch.cuda.current_stream(self.device).wait_stream(side)
            if graph:
                self.graph, self.out, pool = [], [], None
                for parity in range(2):
                    captured = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(captured, pool=pool):
                        self.out.append(fn(self.z, parity))
                    pool = captured.pool()
                    self.graph.append(captured)
            torch.cuda.synchronize(self.device)

    def _ref_chunk(self, lat, first):
        model, scale = self.model, self.scale
        z = lat / scale[1].view(1, model.z_dim, 1, 1, 1) + scale[0].view(1, model.z_dim, 1, 1, 1)
        x = model.conv2(z)
        out = model.decoder(x, feat_cache=self.fm, feat_idx=[0], first_chunk=first)
        return postprocess(out)

    @torch.no_grad()
    def push_gpu(self, latent):
        if latent.shape != (self.batch_size, self.model.z_dim, *self.z.shape[-2:]):
            raise ValueError(f'Unexpected batched latent shape {tuple(latent.shape)}')
        index = self.n
        self.n += 1
        lat = latent.to(self.device, torch.float32).unsqueeze(2)
        if index == 0:
            out = self._ref_chunk(lat, True)
            self.st.load_first_chunk_caches(self.fm)
            self.fm = None
            return out
        parity = index % 2
        if self.graph is not None:
            self.z.copy_(lat)
            self.graph[parity].replay()
            return self.out[parity]
        with native._cudnn_benchmark(True):
            return self.fn(lat, parity)
