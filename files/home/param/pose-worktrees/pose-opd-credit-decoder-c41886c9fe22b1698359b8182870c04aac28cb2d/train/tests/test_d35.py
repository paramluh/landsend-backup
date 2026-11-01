"""D35 unit test (ONE GPU, synthetic, no checkpoints): the pose head's backbone-gradient scaling model_ppm.posehead_in.

usage: python tests/test_d35.py <code dir>
A tiny synthetic 'backbone' (Linear, H = 5120 as prod) -> hn; the REAL model_ppm.pose_head_ce on a real Ctx (3 packed segments) with a real
zero-init-free random head (Linear 5120 -> 2048), labels with -100 holes and per-position weights; plus a stand-in 'tower' loss on hn so the
backbone also gets non-head gradient. Checks:
  (1) s = None and s = 1: loss and every gradient bit-identical to the pre-D35 code path (head reads hn directly);
  (2) s = 0: backbone gradient from the head loss = exactly 0 (total backbone grad == tower-only grad, bitwise); head grads bit-identical;
  (3) forward loss bit-identical for every s in {None, 1, 0.75, 0.5, 0.123, 0};
  (4) intermediate s: backbone head-gradient = s x reference (allclose), head grads bit-identical;
  (5) the trainer's schedule posehead_bbgrad (FROM 5717, RAMP 250) at a few steps."""
import os
import sys

CODE = sys.argv[1]
for p in (os.path.join(CODE, "vendor"), os.path.join(CODE, "lib"), CODE):
    sys.path.insert(0, p)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import model_ppm as M  # noqa: E402

dev = torch.device("cuda", 0)
H, NIN = 5120, 256
segs = [300, 517, 211]
N = sum(segs)
ctx = M.Ctx(segs, dev)
g = torch.Generator(device=dev).manual_seed(35)
x = torch.randn(N, NIN, device=dev, generator=g)
lab = torch.randint(0, M.CBS, (N,), device=dev, generator=g)
lab[torch.rand(N, device=dev, generator=g) < 0.2] = -100
wpos = torch.rand(N, device=dev, generator=g) * (lab >= 0)
wt_tower = torch.randn(N, H, device=dev, generator=g)


def run(s, head_on=True, amp=True):
    torch.manual_seed(0)
    bb = nn.Linear(NIN, H).to(dev)
    head = nn.Linear(H, M.CBS, bias=True).to(dev)            # random (not zero) so the head's input gradient is non-trivial
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        hn = bb(x)
        tower = (hn.float() * wt_tower).mean()                # stand-in for the tower's loss on hn
        loss = tower
        if head_on:
            hin = hn if s == "pre" else M.posehead_in(hn, s)
            l1, s1, n1, k1 = M.pose_head_ce(head, hin, ctx, lab, wpos)
            loss = loss + 1.0 * l1
        else:
            l1 = torch.zeros(())
    loss.backward()
    return dict(loss=loss.detach().clone(), l1=l1.detach().clone(), bbw=bb.weight.grad.clone(), bbb=bb.bias.grad.clone(),
                hw=head.weight.grad.clone() if head.weight.grad is not None else None,
                hb=head.bias.grad.clone() if head.bias.grad is not None else None)


eq = torch.equal
ref = run("pre")                                              # the current (pre-D35) code path: head reads hn
tow = run(None, head_on=False)                                # backbone grad from the tower stand-in only
res = {}
ok = True


def chk(name, c):
    global ok
    ok &= bool(c)
    print(f"{'PASS' if c else 'FAIL'}  {name}")


for s in (None, 1.0, 0.75, 0.5, 0.123, 0.0):
    r = run(s)
    res[s] = r
    chk(f"s={s}: forward loss bit-identical to pre-D35 (head CE {float(r['l1']):.6f}, total {float(r['loss']):.6f})",
        eq(r["loss"], ref["loss"]) and eq(r["l1"], ref["l1"]))
    chk(f"s={s}: head weight/bias grads bit-identical to pre-D35", eq(r["hw"], ref["hw"]) and eq(r["hb"], ref["hb"]))
for s in (None, 1.0):
    chk(f"s={s}: backbone grads bit-identical to pre-D35", eq(res[s]["bbw"], ref["bbw"]) and eq(res[s]["bbb"], ref["bbb"]))
r0 = res[0.0]
chk("s=0: backbone grad == tower-only grad bitwise (zero head contribution)", eq(r0["bbw"], tow["bbw"]) and eq(r0["bbb"], tow["bbb"]))
hd_ref = ref["bbw"] - tow["bbw"]
print(f"      |head part of backbone grad| at s=1: {float(hd_ref.norm()):.4e}  (tower part {float(tow['bbw'].norm()):.4e})")
for s in (0.75, 0.5, 0.123):                                  # bf16 autocast (prod): the scaled hn-gradient is rounded to bf16 (2^-9 rel)
    hd = res[s]["bbw"] - tow["bbw"]
    rel = float((hd - s * hd_ref).norm() / (s * hd_ref).norm())
    chk(f"s={s} (bf16 autocast): head part of backbone grad = s x reference (rel err {rel:.2e}, bf16 rounding bound 4e-3)", rel < 4e-3)
ref32, tow32 = run("pre", amp=False), run(None, head_on=False, amp=False)   # fp32: the scaling itself, without bf16 rounding
hd32 = ref32["bbw"] - tow32["bbw"]
for s in (1.0, 0.75, 0.5, 0.123, 0.0):
    r = run(s, amp=False)
    hd = r["bbw"] - tow32["bbw"]
    rel = float((hd - s * hd32).norm() / hd32.norm())
    chk(f"s={s} (fp32): loss bit-identical, head grads bit-identical, backbone head-part = s x reference (rel err {rel:.2e})",
        eq(r["loss"], ref32["loss"]) and eq(r["hw"], ref32["hw"]) and eq(r["hb"], ref32["hb"]) and rel < 1e-5)

# (5) the trainer's schedule (import train_ppm with the prod env: CFG only, no dist init)
os.environ.update({"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"})
import train_ppm as T  # noqa: E402
want = {5716: None, 5717: 1.0, 5718: 0.996, 5842: 0.5, 5966: 0.004, 5967: 0.0, 6500: 0.0}
got = {k: T.posehead_bbgrad(k) for k in want}
print(f"      schedule FROM {T.CFG['pose_head_detach_from']} RAMP {T.CFG['pose_head_detach_ramp']}: {got}")
chk("schedule: None before FROM, 1.0 at FROM, linear, 0 from FROM + RAMP",
    all((w is None and got[k] is None) or (w is not None and got[k] is not None and abs(got[k] - w) < 1e-9) for k, w in want.items()))
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
