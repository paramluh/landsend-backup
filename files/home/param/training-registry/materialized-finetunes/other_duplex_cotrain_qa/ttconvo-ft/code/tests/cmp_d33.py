"""compare two tests/test_d33.py grads outputs: python cmp_d33.py A.json B.json [--bb]
A = the reference (old code or new code with the link off); B = the run under test. --bb: B has the D33 link (its decp_bb.* entries,
its 12 extra pose stats (training micros: zeros) and its 'decp_bb' group are allowed extras; everything else must be bit-identical)."""
import json
import sys

a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
bb = "--bb" in sys.argv
bad = []
for s, (sa, sb) in enumerate(zip(a["steps"], b["steps"])):
    for i, (ma, mb) in enumerate(zip(sa["micros"], sb["micros"])):
        va, vb = ma["v"], mb["v"]
        n = len(va)
        if vb[:n] != va:
            bad.append((s + 1, i, ma["kind"], "stats differ", [(k, x, y) for k, (x, y) in enumerate(zip(va, vb)) if x != y][:4]))
        extra = vb[n:]
        if extra and not (bb and ma["kind"] == "pose" and all(x == 0 for x in extra)):
            bad.append((s + 1, i, ma["kind"], "unexpected extra stats", extra))
        if ma["loss"] != mb["loss"]:
            bad.append((s + 1, i, ma["kind"], "loss", ma["loss"], mb["loss"]))
    for k in ("grad", "w_after"):
        ka, kb = set(sa[k]), set(sb[k])
        new = kb - ka
        if new and not (bb and all(x.startswith("decp_bb.") for x in new)):
            bad.append((s + 1, k, "unexpected new tensors", sorted(new)))
        if ka - kb:
            bad.append((s + 1, k, "missing", sorted(ka - kb)[:4]))
        diff = [x for x in ka & kb if sa[k][x] != sb[k][x]]
        if diff:
            bad.append((s + 1, k, f"{len(diff)} tensors differ", sorted(diff)[:6]))
        nd = sum(1 for x in ka & kb if x.startswith("dec_"))
        print(f"step {s + 1} {k}: {len(ka & kb)} common tensors ({nd} decoder), bit-identical: {len(ka & kb) - len(diff)}; new in B: {sorted(new)}")
    ga, gb = sa["gn"], sb["gn"]
    dg = {g: (ga[g], gb.get(g)) for g in ga if ga[g] != gb.get(g)}
    if dg:
        bad.append((s + 1, "gn differ", dg))
    if set(gb) - set(ga) - ({"decp_bb"} if bb else set()):
        bad.append((s + 1, "unexpected groups", sorted(set(gb) - set(ga))))
    if sa.get("dec_micros") != sb.get("dec_micros"):
        bad.append((s + 1, "decoder micro counters differ", sa.get("dec_micros"), sb.get("dec_micros")))
    if bb:
        print(f"step {s + 1} B link: grad |W| {sb['grad_norm'].get('decp_bb.weight')}, grad |b| {sb['grad_norm'].get('decp_bb.bias')}, "
              f"input-grad |max| {sb.get('bb_input_grad_absmax')} ({sb.get('n_bb_hooks')} hook(s)), output-grad norm max {sb.get('bb_output_grad_norm_max')}, "
              f"|W| {sb.get('bb_weight_norm_before')} -> {sb.get('bb_weight_norm_after')}, gn_decp_bb {gb.get('decp_bb')}, LR used {sb['lr_used'][-1]} "
              f"next {sb['lr_next'][-1]}")
print("RESULT:", ("BIT-IDENTICAL (except the link's own entries)" if bb else "BIT-IDENTICAL") if not bad else "DIFFERENCES")
for x in bad:
    print("  ", x)
