"""compare two tests/test_d32.py grads outputs: python cmp_d32.py A.json B.json [--head]  (B may have the D32 head; A must not)."""
import json
import math
import sys

a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
head = "--head" in sys.argv
LN = math.log(2048)
bad = []
for s, (sa, sb) in enumerate(zip(a["steps"], b["steps"])):
    for i, (ma, mb) in enumerate(zip(sa["micros"], sb["micros"])):
        va, vb = ma["v"], mb["v"]
        n = len(va)
        if vb[:n] != va:
            bad.append((s + 1, i, ma["kind"], "stats differ", [(k, x, y) for k, (x, y) in enumerate(zip(va, vb)) if x != y][:4]))
        extra = vb[n:]
        if head and ma["kind"] == "pose":
            ce = extra[0] / extra[1]
            print(f"step {s + 1} micro {i} (pose): head CE {ce:.7f} (ln 2048 = {LN:.7f}, diff {ce - LN:+.2e}) over {int(extra[1])} positions; top1 {extra[2] / extra[1]:.4f}; "
                  f"micro loss {ma['loss']:.7f} -> {mb['loss']:.7f} (+{mb['loss'] - ma['loss']:.7f})")
        elif ma["loss"] != mb["loss"] or extra:
            bad.append((s + 1, i, ma["kind"], "loss / stats", ma["loss"], mb["loss"], extra))
    for k in ("grad", "w_after"):
        ka, kb = set(sa[k]), set(sb[k])
        new = kb - ka
        if new and not (head and all(n.startswith("pose_head.") for n in new)):
            bad.append((s + 1, k, "unexpected new params", sorted(new)))
        if ka - kb:
            bad.append((s + 1, k, "missing", sorted(ka - kb)[:4]))
        diff = [n for n in ka & kb if sa[k][n] != sb[k][n]]
        if diff:
            bad.append((s + 1, k, f"{len(diff)} tensors differ", sorted(diff)[:6]))
        print(f"step {s + 1} {k}: {len(ka & kb)} common tensors, bit-identical: {len(ka & kb) - len(diff)}; new in B: {sorted(new)}")
    ga = {g: v for g, v in sa["gn"].items()}
    gb = {g: v for g, v in sb["gn"].items()}
    dg = {g: (ga[g], gb[g]) for g in ga if ga[g] != gb.get(g)}
    if dg:
        bad.append((s + 1, "gn differ", dg))
    print(f"step {s + 1} gn A {ga}\n       gn B {gb}")
    if head:
        print(f"step {s + 1} head: grad norms {{w: {sb['grad_norm'].get('pose_head.weight')}, b: {sb['grad_norm'].get('pose_head.bias')}}}, "
              f"head-input grad |max| {sb.get('head_input_grad_absmax')} ({sb.get('n_head_input_grads')} micro(s)); LR used {sb['lr_used']} next {sb['lr_next']}")
print("RESULT:", "BIT-IDENTICAL (except the head's own entries)" if not bad and head else ("BIT-IDENTICAL" if not bad else "DIFFERENCES"))
for x in bad:
    print("  ", x)
