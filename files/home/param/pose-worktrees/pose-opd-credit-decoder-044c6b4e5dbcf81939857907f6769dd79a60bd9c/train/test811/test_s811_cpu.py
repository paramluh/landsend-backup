"""CPU unit tests of the 8,1,1 helpers (s811.py) + the autograd bridge (model_ppm.GradScale semantics, re-implemented on a toy graph).
usage: python test811/test_s811_cpu.py   (no GPU, no data)"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s811 as S8  # noqa: E402

ok = []


def check(name, cond):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, flush=True)


# ---- expand_seq
check("expand pose*8,speech,text", S8.expand_seq("pose*8,speech,text") == ["pose"] * 8 + ["speech", "text"])
check("expand pose×8 (unicode)", S8.expand_seq("pose×8,speech,text") == ["pose"] * 8 + ["speech", "text"])
check("expand plain list unchanged", S8.expand_seq("pose,speech,speech,text") == ["pose", "speech", "speech", "text"])
check("expand modes", S8.expand_seq("full,frozen*7,full,full") == ["full"] + ["frozen"] * 7 + ["full", "full"])

# ---- coefs: defaults = PPM's per-task coefficients
SEQ = S8.expand_seq("pose*8,speech,text")
MOD = S8.expand_seq("full,frozen*7,full,full")
BB, TW = {"pose": 1.0, "speech": 2.0, "text": 1.0}, {"pose": 1.0, "speech": 2.0}
C = S8.coefs(SEQ, MOD, BB, TW)
check("backbone: only full micros", [c["c_bb"] > 0 for c in C] == [m == "full" for m in MOD])
check("backbone task shares = PPM (pose .25 speech .5 text .25)",
      abs(sum(c["c_bb"] for c in C if c["kind"] == "pose") - 0.25) < 1e-12 and abs(C[8]["c_bb"] - 0.5) < 1e-12 and abs(C[9]["c_bb"] - 0.25) < 1e-12)
check("pose side: every pose micro 1/32 (sum .25 = PPM tower_p)", all(abs(c["c_p"] - 1 / 32) < 1e-12 for c in C[:8]) and C[8]["c_p"] == 0 == C[9]["c_p"])
check("speech side: pose micros 1/32 each (sum .25), speech .5 (= PPM tower_s 1/4 pose + 2/4 speech)",
      all(abs(c["c_s"] - 1 / 32) < 1e-12 for c in C[:8]) and abs(C[8]["c_s"] - 0.5) < 1e-12 and C[9]["c_s"] == 0)
check("speech parts on every pose micro (rule all)", all(c["sp"] for c in C[:9]))
# PPM GA in 8,1,1 mode with all-full micros and default weights = PPM's single coefficient 1/4 everywhere (bridge 1)
Cp = S8.coefs(["pose", "speech", "speech", "text"], ["full"] * 4, BB, TW)
check("PPM GA, all full, defaults -> every coefficient 1/4 (= PPM wt/NM)",
      all(abs(c["c_bb"] - 0.25) < 1e-12 for c in Cp) and all(abs(c["c_s"] - 0.25) < 1e-12 for c in Cp[:3]) and abs(Cp[0]["c_p"] - 0.25) < 1e-12)
# rule full: frozen pose micros skip the speech parts; speech side re-normalised over the micros that run it
Cf = S8.coefs(SEQ, MOD, BB, TW, sp_rule="full")
check("sp_rule full: speech parts only on the full pose micro", [c["sp"] for c in Cf] == [True] + [False] * 7 + [True, False])
check("sp_rule full: speech side pose share .25 on the one full pose micro", abs(Cf[0]["c_s"] - 0.25) < 1e-12 and all(c["c_s"] == 0 for c in Cf[1:8]))
# micro norm
Cm = S8.coefs(SEQ, MOD, {"pose": 1, "speech": 1, "text": 1}, {"pose": 1, "speech": 1}, norm="micro")
check("micro norm: backbone 1/3 per full micro", all(abs(c["c_bb"] - 1 / 3) < 1e-12 for c in Cm if c["mode"] == "full"))
check("micro norm: pose side 1/8 per pose micro", all(abs(c["c_p"] - 1 / 8) < 1e-12 for c in Cm[:8]))
# guards
for bad, kw in (("text frozen", dict(modes=["full"] * 9 + ["frozen"])), ("no full micro", dict(modes=["frozen"] * 10, seq=["pose"] * 9 + ["speech"]))):
    try:
        S8.coefs(kw.get("seq", SEQ), kw["modes"], BB, TW)
        check(f"guard {bad}", False)
    except AssertionError:
        check(f"guard {bad}", True)

# ---- streams
r1 = np.arange(100, 117)                                      # epoch-1 order of 17 rows


def ep(e):
    return np.arange(1000 * e, 1000 * e + 17)[::-1]


st, mk = S8.pose_stream(r1, 12, ep, 8 * 3)
check("pose stream: epoch-1 rest then epoch 2", st[:5].tolist() == [112, 113, 114, 115, 116] and st[5:22].tolist() == ep(2).tolist() and st[22] == ep(3)[0]
      and mk == [(1, 0), (2, 5), (3, 22)])
st2, mk2 = S8.pose_stream(r1, 17 + 4, ep, 8)                  # PPM already in its 2nd pass: epoch 2 directly
check("pose stream: PPM past epoch 1 -> epoch 2 directly", st2.tolist() == ep(2)[:8].tolist() and mk2 == [(2, 0)])
rows = np.arange(50)
ss, sm = S8.speech_stream(rows, 47, rows[5:], 7, 3, 10)
perm = rows[5:][np.random.default_rng(np.random.SeedSequence([7, 3, 0x5E2])).permutation(45)]
check("speech stream: rest of the stream then a seeded epoch", ss[:3].tolist() == [47, 48, 49] and ss[3:].tolist() == perm[:7].tolist() and sm == [(1, 0), (2, 3)])
check("speech epoch 2 is a permutation of the epoch set", sorted(perm.tolist()) == list(range(5, 50)))
check("consumed rows: PPM 2 speech/step until 16762, then 1", S8.consumed_rows("speech", 16762, 16762, ["pose", "speech", "speech", "text"], SEQ) == 33524
      and S8.consumed_rows("speech", 16770, 16762, ["pose", "speech", "speech", "text"], SEQ) == 33532)
check("consumed rows: pose 1/step then 8", S8.consumed_rows("pose", 16764, 16762, ["pose", "speech", "speech", "text"], SEQ) == 16762 + 16)
check("seq_at", S8.seq_at(5, 10, ["a"], ["b"]) == ["a"] and S8.seq_at(10, 10, ["a"], ["b"]) == ["b"])
# resume continuity: a stream built at a later start = the tail of the stream built at the switch (pure function of from_step / seeds)
a, _ = S8.pose_stream(r1, 12, ep, 80)
check("pose stream deterministic (resume = same stream)", np.array_equal(a, S8.pose_stream(r1, 12, ep, 80)[0]))


# ---- autograd bridge (= model_ppm.GradScale) on a toy trunk -> hn -> two towers graph
class GS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s):
        ctx.s = float(s)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.s, None


torch.manual_seed(0)
W = torch.randn(4, 4, requires_grad=True); Ts = torch.randn(4, 4, requires_grad=True); Tp = torch.randn(4, 4, requires_grad=True)
x = torch.randn(3, 4)


def losses(hn_s, hn_p, hn):
    return (hn ** 2).sum(), ((hn_s @ Ts).sin()).sum(), ((hn_p @ Tp).cos()).sum()


cb, cs, cp = 0.25, 1 / 32, 1 / 24
hn = x @ W
text, ls, lp = losses(GS.apply(hn, cb / cs), GS.apply(hn, cb / cp), hn)
torch.autograd.backward([cb * text + cs * ls + cp * lp])
gW, gTs, gTp = W.grad.clone(), Ts.grad.clone(), Tp.grad.clone()
for t in (W, Ts, Tp):
    t.grad = None
hn = x @ W
text, ls, lp = losses(hn, hn, hn)
torch.autograd.backward([cb * (text + ls + lp)])               # reference: the trunk gets c_bb x everything
gW_ref = W.grad.clone()
for t in (W, Ts, Tp):
    t.grad = None
hn = (x @ W).detach()
text, ls, lp = losses(hn, hn, hn)
torch.autograd.backward([cs * ls + cp * lp])
check("bridge: trunk grad = c_bb x (text + both towers' losses)", torch.allclose(gW, gW_ref, rtol=1e-6, atol=1e-7))
check("bridge: tower grads = c_side x their loss gradient", torch.allclose(gTs, Ts.grad, rtol=1e-6) and torch.allclose(gTp, Tp.grad, rtol=1e-6))
print(f"{sum(ok)}/{len(ok)} passed", flush=True)
sys.exit(0 if all(ok) else 1)
