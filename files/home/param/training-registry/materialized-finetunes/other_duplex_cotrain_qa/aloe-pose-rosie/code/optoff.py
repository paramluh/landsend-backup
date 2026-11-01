"""aloe-ft-32-samsep (10-24): main-AdamW state offload (ALOE_OPT_OFFLOAD=1), the ONLY way the reference recipe's 39B FSDP2 state fits world
14 on 80 GB H100s (peak ~79 GiB of 79.65 without it: 38.0 + 574 / W GiB, fitted on jobs 26232 / 25963 / 23079).

OffloadAdamW = torch.optim.AdamW (same constructor args: fused=True, betas, eps, weight_decay) whose exp_avg / exp_avg_sq live in PINNED
HOST memory between steps. step(): the params with a gradient are taken in fixed-size chunks (in param-group order); for each chunk the
states are copied host -> GPU (rebuilt as DTensors with the parameter's own mesh / placements / shape / stride), the UNCHANGED
torch.optim.AdamW.step runs on that chunk only (every other param has grad None for the call, so AdamW skips it), and the states go back to
their host buffers. AdamW (weight_decay 0, no amsgrad) is elementwise per tensor, and the per-tensor 'step' counters stay on the GPU, so the
update of every element is bit-identical to one un-chunked fused step: same kernel, same inputs, same dtypes. The trainer checks that in
process at its first two optimizer steps (spec / proj / vgain tensors vs a fresh fused AdamW on clones). Between steps opt.state holds plain
host tensors: safety_ppm.save_local / load_local read and write them like any other state tensor."""
import torch


def _is_dt(t):
    return hasattr(t, "to_local")


def _loc(t):
    return t.to_local() if _is_dt(t) else t


class OffloadAdamW(torch.optim.AdamW):
    KEYS = ("exp_avg", "exp_avg_sq")

    def __init__(self, params, chunk_gb=2.0, **kw):
        super().__init__(params, **kw)
        assert kw.get("fused") is True and not kw.get("amsgrad", False), "offload path is written for the reference's fused AdamW, no amsgrad"
        self.chunk_bytes = int(float(chunk_gb) * 2 ** 30)
        self.host = {}                                              # (id(p), key) -> pinned host buffer (reused every step)

    def _to_dev(self, p):
        st = self.state.get(p)
        if not st:
            return
        for k in self.KEYS:
            h = st.get(k)
            if h is None or h.is_cuda or _is_dt(h):
                continue
            g = h.to(_loc(p).device, non_blocking=True)
            if _is_dt(p):
                from torch.distributed.tensor import DTensor
                g = DTensor.from_local(g, p.device_mesh, p.placements, run_check=False, shape=p.shape, stride=p.stride())
            st[k] = g

    def _to_host(self, p):
        st = self.state.get(p)
        if not st:
            return
        for k in self.KEYS:
            t = st.get(k)
            if t is None:
                continue
            loc = _loc(t)
            if not loc.is_cuda:
                continue
            key = (id(p), k)
            buf = self.host.get(key)
            if buf is None or tuple(buf.shape) != tuple(loc.shape) or buf.dtype != loc.dtype:
                buf = torch.empty(loc.shape, dtype=loc.dtype, device="cpu", pin_memory=True)
                self.host[key] = buf
            buf.copy_(loc, non_blocking=True)
            st[k] = buf

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None
        ps = [p for g in self.param_groups for p in g["params"] if p.grad is not None]
        grads = {id(p): p.grad for p in ps}
        for p in ps:
            p.grad = None
        i = 0
        while i < len(ps):
            ch, nb = [], 0
            while i < len(ps) and (not ch or nb + 8 * _loc(ps[i]).numel() <= self.chunk_bytes):
                ch.append(ps[i]); nb += 8 * _loc(ps[i]).numel(); i += 1
            for p in ch:
                p.grad = grads[id(p)]
                self._to_dev(p)
            super().step()
            for p in ch:
                self._to_host(p)
                p.grad = None
        torch.cuda.synchronize()                                    # host buffers valid (save_local / digests read them on the CPU)
        for p in ps:
            p.grad = grads[id(p)]
        return None

    def host_bytes(self):
        return sum(b.numel() * b.element_size() for b in self.host.values())
