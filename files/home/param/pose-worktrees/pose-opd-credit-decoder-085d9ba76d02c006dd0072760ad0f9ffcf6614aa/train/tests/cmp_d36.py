"""compare tests/test_d36_ckpt.py phase records: python cmp_d36.py <root> A B [--head-removed]
  plain           A and B must be BIT-IDENTICAL in every recorded quantity (stats, losses, LRs, grad / weight digests, group norms).
  --head-removed  B = PPM_POSEHEAD_OFF=1, A = knob unset with D35 full detach (s = 0): every NON-head quantity must match; the head's own
                  entries (its 3 pose-stats slots, its grads / gn / LR group, the pose micro's head loss term) are the only allowed
                  differences. Digest mismatches fall back to the dumped per-rank tensors (max |diff| / max |A|, fp tolerance 1e-6)."""
import json
import sys

import torch

R, A, B = sys.argv[1], sys.argv[2], sys.argv[3]
HR = "--head-removed" in sys.argv
a, b = json.load(open(f"{R}/{A}.json")), json.load(open(f"{R}/{B}.json"))
HK = ("pose_head.weight", "pose_head.bias")
bad, notes = [], []
TOL = 1e-6


def tensor_cmp(kind, names):
    """max relative |diff| per tensor from the dumps (all ranks' local shards)."""
    out = {}
    for r in range(8):
        try:
            ta = torch.load(f"{R}/dump-{A}/{kind}-r{r}.pt"); tb = torch.load(f"{R}/dump-{B}/{kind}-r{r}.pt")
        except FileNotFoundError:
            continue
        for n in names:
            if n in ta and n in tb:
                d = float((ta[n].double() - tb[n].double()).abs().max()) if ta[n].numel() else 0.0
                s = float(ta[n].double().abs().max()) if ta[n].numel() else 0.0
                out[n] = max(out.get(n, 0.0), d / max(s, 1e-30))
    return out


for sa, sb in zip(a["steps"], b["steps"]):
    st = sa["step"]
    if HR and sa.get("bbgrad_s") != 0.0:                          # A must be D35 FULLY detached (s = 0), as prod since step 5967
        bad.append((st, "A is not fully detached", sa.get("bbgrad_s")))
    notes.append(f"step {st} D35 s: A {sa.get('bbgrad_s')} B {sb.get('bbgrad_s')}")
    ga = [g for g in sa["groups"] if not (HR and g == "pose_head")]
    if ga != sb["groups"]:
        bad.append((st, "groups", sa["groups"], sb["groups"]))
    for k in ("lr_used", "lr_next", "gn"):
        da = {g: v for g, v in sa[k].items() if not (HR and g == "pose_head")}
        if da != sb[k]:
            bad.append((st, k, {g: (da.get(g), sb[k].get(g)) for g in set(da) | set(sb[k]) if da.get(g) != sb[k].get(g)}))
    for i, (ma, mb) in enumerate(zip(sa["micros"], sb["micros"])):
        va, vb = list(ma["v"]), list(mb["v"])
        if HR and ma["kind"] == "pose":                           # the head's 3 slots sit after the pose light cb stats, before D33's
            k0 = 10 + 10 + 3 + 4                                  # len(TK) + 10 + len(CB_S) + len(CB_P)
            notes.append(f"step {st} pose micro: head slots removed from A (rank 0 sum {va[k0]:.4f}, n {va[k0 + 1]:.0f}, hits {va[k0 + 2]:.0f}) -> "
                         f"head CE (rank 0) {va[k0] / max(va[k0 + 1], 1):.4f}; loss A {ma['loss']:.6f} B {mb['loss']:.6f} (A - B = head term {ma['loss'] - mb['loss']:+.6f})")
            del va[k0:k0 + 3]
            sa_sum = list(ma["sum"]); del sa_sum[k0:k0 + 3]
            if sa_sum != mb["sum"]:
                bad.append((st, i, "all-rank stats sums differ"))
            ka = {k: v for k, v in ma["summ"].items() if "pose_head" not in k}
            if ka != mb["summ"] or any("pose_head" in k for k in mb["summ"]):
                bad.append((st, i, "summarize differs", [k for k in set(ka) | set(mb["summ"]) if ka.get(k) != mb["summ"].get(k)][:6]))
        else:
            if ma["loss"] != mb["loss"]:
                bad.append((st, i, ma["kind"], "loss", ma["loss"], mb["loss"]))
            if ma["summ"] != mb["summ"]:
                bad.append((st, i, ma["kind"], "summarize differs"))
        if va != vb:
            bad.append((st, i, ma["kind"], "stats differ", [(j, x, y) for j, (x, y) in enumerate(zip(va, vb)) if x != y][:4], len(va), len(vb)))
    for k, dump in (("grad", "grad"), ("w_after", "w")):
        na = {n for n in sa[k] if not (HR and n in HK)}
        nb = set(sb[k]) - (set(HK) if k == "w_after" and HR else set())
        if na != nb:
            bad.append((st, k, "names differ", sorted(na ^ nb)[:6]))
        diff = sorted(n for n in na & nb if sa[k][n] != sb[k][n])
        if diff:
            rel = tensor_cmp(dump, diff)
            over = {n: r_ for n, r_ in rel.items() if r_ > TOL}
            (bad if over or len(rel) < len(diff) else notes).append(
                (st, k, f"{len(diff)} tensors not bit-identical; max rel diff {max(rel.values()) if rel else None}", sorted(over)[:6]))
        notes.append(f"step {st} {k}: {len(na & nb)} tensors compared, bit-identical {len(na & nb) - len(diff)}")
    if HR:
        notes.append(f"step {st} B: head grads None {sb['head_grad_none']}; groups {sb['groups']}")
print("\n".join(str(x) for x in notes))
print(f"RESULT {A} vs {B}{' (head removed in B)' if HR else ''}:", "MATCH" if not bad else "DIFFERENCES")
for x in bad:
    print("  ", x)
sys.exit(1 if bad else 0)
