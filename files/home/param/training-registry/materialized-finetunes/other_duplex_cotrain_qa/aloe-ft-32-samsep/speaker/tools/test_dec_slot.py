"""CPU unit test (10-24) of the speaker trainer's decoder update path (train_speaker_vox.nanosteps_tw / _dec_update) with a toy decoder, gloo
world 1 (WORLD patched to 1): A) per-slot mode, 2 slots, decoder LR 0: K updates per slot (2K total), LR indices K x slot base + j, decoder
update count +1 per main step, and the tower (hv) gradient == autograd of the step's token mean (sum of every frame's 31-token loss / (31 N),
N = the main step's frames over both slots); B) per-slot mode with ONE slot == the standard K-nanostep path (29345 / 21980 rule) bit for bit
(decoder params after K AdamW updates at a real LR, and the hv gradient); C) a dummy-only slot (skip_update) takes no update.
usage (compute node, env = run_speaker.env + the LW64 overrides): python -I test_dec_slot.py <code dir>"""
import copy
import os
import sys

sys.path.insert(0, sys.argv[1])
os.environ.update(RANK="0", WORLD_SIZE="16", LOCAL_RANK="0", LOCAL_WORLD_SIZE="8", COTRAIN_CFG_DRY="1", MASTER_ADDR="127.0.0.1", MASTER_PORT="29613")
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import train_speaker_vox as T  # noqa: E402

torch.Tensor.pin_memory = lambda self, *a, **k: self                # CPU test: no CUDA pinned memory
dist.init_process_group("gloo", rank=0, world_size=1)
T.WORLD = 1
K = 4
T.CFG["dec_k"] = K
torch.manual_seed(0)
H, Tn = 8, 50


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(H, 31) * 0.1)

    def forward(self, hd, depth, fl, wf, n_valid=None, zb=None):
        e = ((hd[fl] @ self.W) - depth[fl, 1:].float() / 2048.0) ** 2
        nv = int(n_valid)
        return (wf[:, None] * e).sum(), e[:nv].sum().detach(), 31 * nv, e[:nv].sum(0).detach()


def micro(n_light):
    hv = torch.randn(Tn, H)
    depth = torch.randint(0, 2048, (Tn, 32))
    light = torch.randperm(Tn)[:n_light].sort().values
    return hv, depth, light


def mjs(light):
    return [float(len(range(j, len(light), K))) for j in range(K)]


rec = []
ok = {}
# ---- A: per-slot, 2 slots, LR 0
dec = Toy(); do = T.DecOpt(dec, 0.0, 1.0); do.lam = lambda t: (rec.append(t), 1.0)[1]
W0 = dec.W.detach().clone()
ms = [micro(37), micro(23)]
Mjs = [mjs(m[2]) for m in ms]
N = sum(sum(x) for x in Mjs)
accs = []
for g, (hv, depth, light) in enumerate(ms):
    nano = {"Mj": Mjs[g], "N": N, "first": True, "last": True, "inc": g == 1, "lr_base": g * K, "skip_update": False, "step": 0, "lrank": g}
    acc, st, lrs, (Mj, N_), _ = T.nanosteps_tw(dec, do, hv, depth, light, nano=nano)
    accs.append(acc)
exp = []
for hv, depth, light in ms:
    h = hv.clone().requires_grad_()
    e = ((h[light] @ W0) - depth[light, 1:].float() / 2048.0) ** 2
    (e.sum() / (31.0 * N)).backward()
    exp.append(h.grad)
ok["A_updates_2K"] = rec == list(range(2 * K))
ok["A_micros_+1"] = do.micros == 1
ok["A_W_unchanged_lr0"] = torch.equal(dec.W.detach(), W0)
ok["A_hv_grad_step_mean"] = all(torch.allclose(a, b, rtol=1e-5, atol=1e-8) for a, b in zip(accs, exp))
# ---- B: per-slot with one slot == the standard K-nanostep path
hv, depth, light = micro(41)
outs = []
for mode in ("std", "slot"):
    torch.manual_seed(1)
    dec = Toy(); do = T.DecOpt(dec, 1e-2, 1.0); do.lam = lambda t: 1.0
    nano = {"Mj": mjs(light), "first": True, "last": True, "step": 0, "lrank": 0}
    if mode == "slot":
        nano.update({"N": sum(mjs(light)), "inc": True, "lr_base": 0, "skip_update": False})
    acc, st, lrs, _, _ = T.nanosteps_tw(dec, do, hv, depth, light, nano=nano)
    outs.append((dec.W.detach().clone(), acc.clone(), lrs, do.micros))
ok["B_params_bitwise"] = torch.equal(outs[0][0], outs[1][0])
ok["B_hv_grad_bitwise"] = torch.equal(outs[0][1], outs[1][1])
ok["B_lrs_counts"] = outs[0][2] == outs[1][2] and len(outs[0][2]) == K and outs[0][3] == outs[1][3] == 1
ok["B_W_moved"] = not torch.equal(outs[0][0], Toy().W.detach())
# ---- C: dummy-only slot -> no update
dec = Toy(); do = T.DecOpt(dec, 1e-2, 1.0); rec2 = []; do.lam = lambda t: (rec2.append(t), 1.0)[1]
W0 = dec.W.detach().clone()
nano = {"Mj": [0.0] * K, "N": 1.0, "first": True, "last": True, "inc": True, "lr_base": 0, "skip_update": True, "step": 0, "lrank": 0}
_, _, lrs, _, _ = T.nanosteps_tw(dec, do, hv, depth, light, dummy=True, nano=nano)
ok["C_no_update"] = torch.equal(dec.W.detach(), W0) and not rec2 and not lrs and do.micros == 1
print({k: bool(v) for k, v in ok.items()})
print("TEST", "PASS" if all(ok.values()) else "FAIL")
