"""FRESH-RESUME FLIP CHECKS (10-21; data_ppm FLIP_BIT, chat_ppm.eob_to_user, train_ppm CFG['fresh']). Run on a training host (node-local
s3-rows + txtdup-full), CPU only:  python check_flip.py <out.txt>
  1. speech: 3 training rows (voice clips / persona present): flipped vs unflipped vs raw store words; B(flipped) == A(unflipped) on the
     conversation (lane ids, Mimi codes, cb0 labels, targets); prefix clip / persona data sides follow the flip.
  2. Gemma txtdup: 3 train packs: flipped lanes == unflipped lanes swapped, per conversation.
  3. chat: 3 train packs: B(flipped) == the user lane + <|im_end|> after every user turn (count == user messages of the raw conversation),
     A(flipped) == the assistant lane with its <|im_end|> removed.
  4. evals + unflipped training items: new code == the pre-fresh code (.bak-pre-fresh) bit for bit."""
import importlib.util
import json
import os
import sys

import numpy as np

C = "/data/home/amartyavarma/ppm/code-811"
sys.path.insert(0, f"{C}/lib"); sys.path.insert(0, C)
import data_ppm as D  # noqa: E402
import chat_ppm as CH  # noqa: E402

OUT = open(sys.argv[1], "w") if len(sys.argv) > 1 else sys.stdout


def P_(*a):
    print(*a, file=OUT, flush=True)


def load_old(name, path):
    import importlib.machinery
    ld = importlib.machinery.SourceFileLoader(name, path)
    s = importlib.util.spec_from_loader(name, ld)
    m = importlib.util.module_from_spec(s); ld.exec_module(m)
    return m


DO = load_old("data_ppm_old", f"{C}/data_ppm.py.bak-pre-fresh")
CO = load_old("chat_ppm_old", f"{C}/chat_ppm.py.bak-pre-fresh")
cfg = D.default_cfg(seed=20261011, sp_cap=2000, persona_sides="both", crop_conv=True, txt_cap=7500, chat_side="assistant", chat_eob=True)
tok = D._tok_init()


def dec(ids, n=40):
    ids = [int(x) for x in ids if int(x) < D.CTRL_LO or int(x) == CH.EOB][:n]
    return tok.decode(ids).replace("\n", " ")[:220]


def same(a, b):
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, np.ndarray):
            if not (isinstance(y, np.ndarray) and x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y)):
                return False
        elif x != y:
            return False
    return True


fails = []


def check(cond, msg):
    P_(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


# ------------------------------------------------------------------ 1. speech
root = "/scratch/amartyavarma/s3-rows"
ds = D.MixedSpeechDS(root, None, [], cfg, 0)
node, st, sib, ps = ds._open(0)
host = node["node"]
P_(f"== host store {host}: {len(node['length']):,} rows")
picked, r = [], 0
full = np.flatnonzero(node["length"] == 7500)
for r in full[:4000].tolist():
    row = st[r]
    if D.h64(f"ppm-voice|{cfg['seed']}|{row['row_key']}") % 3 != 0:
        continue
    try:
        pers = ps.get(r, row["row_key"])
    except (FileNotFoundError, OSError):
        pers = None
    if pers is None:
        continue
    m0 = ds.get(D.code_of(0, r))
    pv = json.loads(m0["prov"])
    if len(pv["clips"]) == 2 and len(pv["persona"]) == 2:
        picked.append(r)
    if len(picked) == 3:
        break
P_(f"speech rows picked (voice clips on both channels + both personas): {picked}")
for r in picked:
    row = st[r]
    u = ds.get(D.code_of(0, r)); f = ds.get(D.code_of(0, r) | D.FLIP_BIT)
    old = DO.MixedSpeechDS(root, None, [], cfg, 0).get(DO.code_of(0, r))
    pu, pf = json.loads(u["prov"]), json.loads(f["prov"])
    P_(f"\n-- speech row {r} {row['row_key']}: side_of swap {pu['swap']} (B = data {pu['chan_B_data_side']}); flipped swap {pf['swap']} (B = data {pf['chan_B_data_side']})")
    P_(f"   raw words user     : {' '.join(w['text'] if isinstance(w, dict) else str(w) for w in list(row['words_user'])[:25])[:200]}")
    P_(f"   raw words assistant: {' '.join(w['text'] if isinstance(w, dict) else str(w) for w in list(row['words_assistant'])[:25])[:200]}")
    Pu, Pf = u["P"], f["P"]
    P_(f"   unflipped: P {Pu}; clips {[(c['track'], c['data_side']) for c in pu['clips']]}; persona {pu['persona']}")
    P_(f"   flipped  : P {Pf}; clips {[(c['track'], c['data_side']) for c in pf['clips']]}; persona {pf['persona']}")
    P_(f"   unflipped lane A: {dec(u['ids'][0, Pu:])}")
    P_(f"   unflipped lane B: {dec(u['ids'][1, Pu:])}")
    P_(f"   flipped   lane B: {dec(f['ids'][1, Pf:])}")
    check(same(u, old), "unflipped training micro == pre-fresh code (bit for bit)")
    check(pf["swap"] == 1 - pu["swap"] and pf.get("flip") == 1, "flipped swap = 1 - side_of")
    check(Pu == Pf, f"prefix length equal ({Pu}, {Pf})")
    du = {c["track"]: c["data_side"] for c in pu["clips"]}; dfl = {c["track"]: c["data_side"] for c in pf["clips"]}
    check(dfl.get("B") == du.get("A") and dfl.get("A") == du.get("B"), "voice clips follow the flip (B's clip = the new B speaker's)")
    tu = {p["track"]: p["tokens"] for p in pu["persona"]}; tf = {p["track"]: p["tokens"] for p in pf["persona"]}
    check(sorted(tu.values()) == sorted(tf.values()), f"persona blocks follow the flip (tokens per channel {tu} -> {tf})")
    a_u = u["ids"][0, Pu:]; b_f = f["ids"][1, Pf:]
    check(np.array_equal(b_f, a_u) and np.array_equal(f["ids"][0, Pf:], u["ids"][1, Pu:]), "conversation lanes swapped exactly (B_flip == A_unflip, A_flip == B_unflip)")
    check(np.array_equal(f["depth"][1, Pf:], u["depth"][0, Pu:]) and np.array_equal(f["depth"][0, Pf:], u["depth"][1, Pu:]), "Mimi codes swapped exactly")
    lab = u["depth"][0, :, 0].copy(); lab[lab < 0] = -100; lab[:Pu + D.CD] = -100
    check(np.array_equal(f["cb0"][Pf:], lab[Pu:]), "cb0 labels of flipped B == channel A's codes (cb0) of the unflipped row")
    check(np.array_equal(f["y"], u["ids"][0, u["sel"] + 1]) and np.array_equal(f["sel"], u["sel"]), "text targets of flipped B == unflipped A's next tokens")
    check(np.array_equal(f["light"], u["light"]) if Pu == Pf else True, "light frames (100%) equal")
    # B's lane equals the raw store lane of the new B data side
    side = "user" if pf["swap"] else "assistant"
    import data_s3q as DS
    std = DS.channel_stream(row, side, D.CD, 7500, True, 0.0, off=Pf, n_use=min(int(row["n_frames"]), 7500 - Pf))
    check(np.array_equal(f["ids"][1, Pf:], std["ids"][Pf:]) and np.array_equal(f["depth"][1, Pf:], std["depth"][Pf:]),
          f"flipped B lane + codes == raw store channel '{side}' (words_{side}, codes_{side})")

# eval rows (speech + speech_A) unchanged
sidx = D.DS.load_index(root)
splan = D.speech_plan(node, sidx, 20261011, 8, 0, 0.0005, "ppm-v1", 2)
ev_n = [D.SpeechDS(root, splan["eval_rows"], cfg, 0)[i] for i in range(2)]
ev_o = [DO.SpeechDS(root, splan["eval_rows"], cfg, 0)[i] for i in range(2)]
evA_n = [ds.get(D.code_of(0, r), swap=1 - int(m["swap"])) for r, m in zip(splan["eval_rows"], ev_n)]
evA_o = [DO.MixedSpeechDS(root, None, [], cfg, 0).get(DO.code_of(0, r), swap=1 - int(m["swap"])) for r, m in zip(splan["eval_rows"], ev_o)]
check(all(same(a, b) for a, b in zip(ev_n, ev_o)) and all(same(a, b) for a, b in zip(evA_n, evA_o)), "speech eval + speech_A eval micros == pre-fresh code")

# ------------------------------------------------------------------ 2. Gemma txtdup
troot = "/scratch/amartyavarma/ppm/txtdup-full"
tds = D.TextDS(troot, "train", [], cfg); tst = D.TXD.Store(troot)
for p in (1234567 % tst.n_packs("train"), 17012 * 128 + 5, 17100 * 128 + 77):
    p = p % tst.n_packs("train")
    u = D.TextDS(troot, "train", [p], cfg)[0]; f = D.TextDS(troot, "train", [p | D.FLIP_BIT], cfg)[0]
    o = DO.TextDS(troot, "train", [p], cfg)[0]
    pu, pf = json.loads(u["prov"]), json.loads(f["prov"])
    recs = [tst.get(j) for j in tst.pack("train", p)]
    P_(f"\n-- Gemma pack {p}: {len(recs)} conversations; swaps {[x['swap'] for x in pu]} -> {[x['swap'] for x in pf]}")
    r0 = recs[0]
    lanes, n, _ = D.TXD.conv_lanes(r0, 7500, True)
    P_(f"   raw conv {r0['conv_id']} side a (user): {dec(lanes[0])}")
    P_(f"   raw conv {r0['conv_id']} side b (asst): {dec(lanes[1])}")
    P_(f"   unflipped B: {dec(u['ids'][1, :n])}")
    P_(f"   flipped   B: {dec(f['ids'][1, :n])}")
    check(same(u, o), "unflipped Gemma train micro == pre-fresh code")
    check(all(b["swap"] == 1 - a["swap"] for a, b in zip(pu, pf)), "every conversation's swap flipped")
    check(np.array_equal(f["ids"][1], u["ids"][0]) and np.array_equal(f["ids"][0], u["ids"][1]), "lanes swapped exactly (B_flip == A_unflip)")
    wb = lanes[1 - pf[0]["swap"]] if False else lanes[0 if pf[0]["swap"] else 1]
    check(np.array_equal(f["ids"][1, :n], wb), f"flipped B (conv 0) == raw side {'a (user)' if pf[0]['swap'] else 'b (assistant)'} lane")
    check(np.array_equal(f["y"], f["ids"][1, f["sel"] + 1]) and float(f["n_text"]) == float(len(f["sel"])), "targets = flipped B's next tokens")
ev_n = [D.TextDS(troot, "eval", [0, 1], cfg)[i] for i in range(2)]; ev_o = [DO.TextDS(troot, "eval", [0, 1], cfg)[i] for i in range(2)]
check(all(same(a, b) for a, b in zip(ev_n, ev_o)), "Gemma eval micros == pre-fresh code")

# ------------------------------------------------------------------ 3. chat
cpacks = CH.multi_split_packs([2, 3], 7500, 20261011)
mine = cpacks[0::128]
P_(f"\n== chat: {len(cpacks):,} packs (splits 2, 3), rank 0 {len(mine):,}")
rend = {}
for k in (0, len(mine) // 2, len(mine) - 1):
    sp, rows = mine[k]
    u = CH.ChatDS([(sp, rows)], cfg)[0]; f = CH.ChatDS([(sp, rows, True)], cfg)[0]; o = CO.ChatDS([(sp, rows)], cfg)[0]
    pu, pf = json.loads(u["prov"]), json.loads(f["prov"])
    if sp not in rend:
        rend[sp] = CH.Renderer(sp)
    R = rend[sp]
    P_(f"\n-- chat pack (split {sp}, {len(rows)} conversations) rank-0 index {k}: swaps {[x['swap'] for x in pu]} -> {[x['swap'] for x in pf]}; "
       f"user EOBs {[x.get('eob_user') for x in pf]} nofit {[x.get('eob_user_nofit') for x in pf]}")
    check(same(u, o), "unflipped chat train micro == pre-fresh code")
    off = 0
    ok_all = True
    for ci, (j, a, b) in enumerate(zip(rows, pu, pf)):
        row = R(j)
        lanes, P, _ = CH.conv_lanes(row, True)
        L = a["L"]
        seg_u, seg_f = u["ids"][:, off:off + L], f["ids"][:, off:off + L]
        c0 = P + CH.PREPEND
        msgs = R.ds[int(j)]["messages"]
        roles = [m_["role"] for m_ in msgs]
        nuser = sum(1 for i_, r_ in enumerate(roles) if r_ == "user")
        bf = seg_f[1, c0:].copy(); af = seg_f[0, c0:].copy()
        e_ = bf == CH.EOB
        bu = seg_u[0, c0:]; au = seg_u[1, c0:].copy()
        c1 = np.array_equal(np.where(e_, CH.PAD, bf), bu)                       # B_flip minus its EOBs == the user lane
        au2 = np.where(au == CH.EOB, CH.PAD, au)
        c2 = np.array_equal(af, au2)                                             # A_flip == the assistant lane without <|im_end|>
        c3 = bool((np.flatnonzero(e_) > 0).all()) and bool((bu[np.flatnonzero(e_) - 1] != CH.PAD).all())   # EOB right after a user piece
        c4 = int(e_.sum()) + int(b.get("eob_user_nofit", 0)) == nuser or L < len(lanes[0])
        c5 = np.array_equal(seg_f[1, :P], seg_u[1, :P])                          # system text stays on B
        if ci == 0:
            P_(f"   raw roles {roles[:8]}{'...' if len(roles) > 8 else ''}; user msgs {nuser}; EOBs on flipped B {int(e_.sum())}; system P {P}")
            P_(f"   raw user msg 0     : {msgs[[i_ for i_, r_ in enumerate(roles) if r_ == 'user'][0]]['content'][:160]!r}")
            P_(f"   unflipped B (asst) : {dec(seg_u[1, c0:])}")
            P_(f"   flipped   B (user) : {dec(seg_f[1, c0:])}")
            P_(f"   flipped   A (asst) : {dec(seg_f[0, c0:])}")
        if not (c1 and c2 and c3 and c4 and c5):
            ok_all = False
            P_(f"   conv {j}: user-lane {c1} asst-lane {c2} eob-after-piece {c3} eob-count {int(e_.sum())}+{b.get('eob_user_nofit')} vs user msgs {nuser} ({c4}) sys {c5}")
        off += L
    check(ok_all, "every conversation: B_flip = user lane + <|im_end|> after each user turn (count = user messages), A_flip = assistant lane w/o <|im_end|>, system on B")
    check(np.array_equal(f["y"], f["ids"][1, f["sel"] + 1]) and bool(f["eob"].sum() > 0) and not bool(f["nw"].any()), "targets = flipped B's next tokens incl. <|im_end|>")
ctb0, _, cho = CH.table(2)
cev = [(2, p) for p in CH.pack_plan(cho, ctb0, 7500, 0, shuffle=False)]
ev_n = [CH.ChatDS([cev[i]], cfg)[0] for i in range(2)]; ev_o = [CO.ChatDS([cev[i]], cfg)[0] for i in range(2)]
check(all(same(a, b) for a, b in zip(ev_n, ev_o)), "chat eval micros == pre-fresh code")

P_(f"\nRESULT: {'ALL OK' if not fails else f'{len(fails)} FAIL(S)'}")
for x in fails:
    P_("  FAIL: " + x)
