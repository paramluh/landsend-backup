"""aloe-uncond-ft (10-16, UNCONDITIONAL channel B = job 22091 with channel A EMPTY): env QQA_A_EMPTY (default 0 = 22091 / 22291 unchanged).
QQA_A_EMPTY=1 (with QQA_POSE_B=1): SAME stores / items / plan / split / rank sharding / B targets / B pose; the ONLY change is channel A:
  SYSTEM BLOCK  22091's prefix rule run unchanged (same voice draw, both clips chosen, BOTH persona blocks built and cropped to 22091's
                budget), then every channel-A segment (A's voice clip, A's persona block) is DROPPED before sysprompt.build_prefix: B's clip
                and B's persona block are token-identical to 22091's; header / footer on B as always; nothing left -> PPM's empty block (P 5).
  CONVERSATION  channel A at EVERY position = PPM's one-sided-row representation (lib/pose_s3q.pose_micro, PPM's pose-only TikTok rows:
                ids_A = PAD, depth_A = -1, present_A = 0; model_ppm.Inp: e = sum_c present_c (emb_c(ids_c) + aud_c(depth_c)) -> A adds
                exactly nothing): no partner words, no partner Mimi, no A voice clip, no A persona.
  B             lane / Mimi / pose / every target (y word pad eob nw cb0 light pcb0 plight) = 22091's, shifted only by the shorter prefix.
aloe-nopose-ft (10-16, the no-pose ablation of job 22091): env QQA_POSE_B (default 1 = 22091 unchanged). QQA_POSE_B=0: SAME stores,
SAME items (the qualifying B tracks: pose_index b_ok selection unchanged), SAME plan / split / rank sharding, SAME speech micro; the ONLY change:
the micro carries NO pose (no pdepth / pcb0 / plight), kind 'speech' (= 21980's data_aloe micro: B = Mimi + words, A = partner's words + Mimi),
own counts = (text targets, cb0 labels, light frames).
aloe-pose-ft DATA (10-16): 21980's data_aloe with POSE ON CHANNEL B. CPU only (DataLoader workers, CPU tests).

STORES: the 8 aloe-pose-ft stores p0..p7 (build_poseB_rows.py: s3-duplex-rows-v1 + pose add-on), staged unchanged to every training node's
/scratch/amartyavarma/aloe-pose-ft/stores/pK (digests asserted against STORES.json written after the build).
ITEMS (user spec 10-16): one item per QUALIFYING B track (pose ok + no crossbleed into B) per row: (store k, node row r, swap s) with
  swap 0: model channel B = data 'assistant' (SPEAKER_00), A = data 'user'      (iff pose_index b_ok[r, 1])
  swap 1: model channel B = data 'user' (SPEAKER_01),     A = data 'assistant'  (iff pose_index b_ok[r, 0])
  A conversation whose two tracks both qualify gives both orientations; A is conditioning only (its words + Mimi, no pose, no loss).
MICRO = data_ppm.speech_micro(row, r, sib, pers, cfg, rank, swap) UNCHANGED (21980's speech-row maths: PPM sysprompt rule with both
  speakers' personas each on its own model channel, voice clips on 1/3 of rows, NEW_WORD timing targets, CD = 2, no conversation crop)
  + channel B's POSE exactly as PPM's pose rows carry it (lib/pose_s3q.segment_arrays / pose_micro):
    pdepth [N, 32]  B's pose codes of row frame f at position P + f + pose_delay (pose_delay = PPM's cfg 'pose_delay' = CD = 2: the same
                    positions as B's Mimi frame f), codebook offsets cb x 2048, -1 elsewhere (system prompt, the first CD positions)
    pcb0 [N]        pose cb0 label where pdepth >= 0 (-100 none)
    plight [Lp]     the pose 0.6B frames: pose_s3q.light_pick over those positions with PPM's pose_light_frac 1.0 (= all of them) and
                    PPM's seed words (light_seed, h64(row_key), 1, 0x9053)
  kind = 'pose' (model_ppm.PPM.forward: + inp.pose(pdepth) summed into the backbone input, speech tower on depth[1], pose tower on pdepth,
  decp_bb link; the trainer adds the pose cb0 loss and the pose decoder nanosteps).
PLAN (identical on every rank): train items permuted with seed 20261011; one item per rank per step (step t, rank R -> item t x WORLD + R);
  the last step's spare ranks run DUMMY micros (zero weights). EVAL: every eval item, round-robin over ranks (item i -> rank i % WORLD).
aloe-pose-mono-8020 (10-18): env ALOE_DATA=aloe-mono (default v2pure = 23051 / 24198 unchanged): the v2pure machinery (64 rows of <= 7,500
frames per step, every item its own attention segment, channel A empty) over the ALOE HANDOVER POSE stores of job 22091 (ALOE_STORE_DIR =
aloe-pose-ft/stores): one MONOPLEX item per qualifying track (pose ok, no crossbleed into it) = (store, row, swap) exactly as 22091 / 22317
select them; plus the 10-18 prompt rule (env ALOE_PERSONA_RULE / ALOE_PERSONA_P10 / ALOE_PERSONA_TAG, ALOE_DRAW_KEY=item): voice clip and
persona text drawn independently per item (key row_key|data side). Section at the end of the file."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import data_qqa as Q  # noqa: E402

D = Q.D
PS = D.PS
NST = 8
# aloe-v2pure-ft (10-17): the release-v2 stores (build_v2pure_rows.py), read in place from Weka (no staging: ~700 GB, 7,500-frame padded rows)
STORE_DIR = os.environ.get("ALOE_STORE_DIR", "/data/home/amartyavarma/ppm/aloe-v2pure-ft/stores")
STORES = [f"{STORE_DIR}/p{k}" for k in range(NST)]
SRC_STORES = list(STORES)
DIGESTS_FILE = f"{STORE_DIR}/STORES.json"
PLAN_SEED = 20261011
BUDGET = 2000 + 7500 + D.CD                       # sp_cap + the longest row + CD
POSE_SEED_TAG = 0x9053
POSE_B = os.environ.get("QQA_POSE_B", "1") == "1"   # aloe-nopose-ft: 0 = no pose anywhere (module doc)
assert os.environ.get("QQA_POSE_B", "1") in ("0", "1")
A_EMPTY = os.environ.get("QQA_A_EMPTY", "0") == "1"   # aloe-uncond-ft: 1 = channel A empty (module doc)
assert os.environ.get("QQA_A_EMPTY", "0") in ("0", "1")


def aloe_cfg():
    """22317 / 23051's data cfg, plus (aloe-v2pure-clip80, 10-18) the system-prompt rule from env; unset env = 23051's cfg exactly:
    ALOE_PERSONA_SIDES  persona text on which channels ("both" = 23051; "none" = no persona text at all)
    ALOE_VOICE_RULE     "mod" (23051 / PPM: voice clip drawn iff h64("ppm-voice|<seed>|<row_key>") % voice_mod == 0, i.e. 1/3)
                        "p10" (10-18 rule: drawn iff h64("<voice_tag>-voice|<seed>|<row_key>") % 10 < voice_p10; an independent draw:
                        its own tag, shared with no other draw). The clip itself = data_ppm.voice_clips unchanged (same choice as 23051)."""
    c = Q.qqa_cfg()
    ps = os.environ.get("ALOE_PERSONA_SIDES")
    if ps is not None:
        assert ps in ("both", "A", "B", "none"), ps
        c["persona_sides"] = ps
    rule = os.environ.get("ALOE_VOICE_RULE", "mod")
    assert rule in ("mod", "p10"), rule
    if rule == "p10":
        p10, tag = int(os.environ["ALOE_VOICE_P10"]), os.environ["ALOE_VOICE_TAG"]
        assert 0 <= p10 <= 10 and tag and "|" not in tag, (p10, tag)
        del c["voice_mod"]                                        # unused under p10 (kept out of the cfg so it cannot be misread)
        c.update(voice_rule="p10", voice_p10=p10, voice_tag=tag)
    # aloe-pose-mono-8020 (10-18): persona text drawn per item (ALOE_PERSONA_RULE p10: h64("<tag>-persona|<seed>|<key>") % 10 < p10; unset =
    # persona on every item that has one, 23051 / 22317); ALOE_DRAW_KEY item: both draws keyed on row_key|<B's data side> (one draw per
    # monoplex item; unset = row_key, 23051 / 24198)
    pr = os.environ.get("ALOE_PERSONA_RULE")
    if pr is not None:
        assert pr == "p10", pr
        pp10, ptag = int(os.environ["ALOE_PERSONA_P10"]), os.environ["ALOE_PERSONA_TAG"]
        assert 0 <= pp10 <= 10 and ptag and "|" not in ptag, (pp10, ptag)
        c.update(persona_rule="p10", persona_p10=pp10, persona_tag=ptag)
    dk = os.environ.get("ALOE_DRAW_KEY")
    if dk is not None:
        assert dk == "item", dk
        c["draw_key"] = "item"
    return c


def _dkey(row_key, side, cfg):
    """the draw key: row_key (23051 / 24198) or row_key|<B's data side> (ALOE_DRAW_KEY item: one draw per monoplex item)."""
    if cfg.get("draw_key") == "item":
        assert side in ("user", "assistant"), ("per-item draw needs B's data side", side)
        return f"{row_key}|{side}"
    return row_key


def voice_draw(row_key, cfg, side=None):
    """does this item's system block get a voice clip drawn? (aloe_cfg doc; no 'voice_rule' key = 23051's / PPM's rule, the same hash)"""
    if "voice_rule" not in cfg:
        return D.h64(f"ppm-voice|{int(cfg['seed'])}|{row_key}") % int(cfg["voice_mod"]) == 0
    assert cfg["voice_rule"] == "p10", cfg["voice_rule"]
    return D.h64(f"{cfg['voice_tag']}-voice|{int(cfg['seed'])}|{_dkey(row_key, side, cfg)}") % 10 < int(cfg["voice_p10"])


def persona_draw(row_key, cfg, side=None):
    """aloe-pose-mono-8020: does this item get persona text (if the sidecar has one for B)? no 'persona_rule' key = every item (23051 / 22317)."""
    if "persona_rule" not in cfg:
        return True
    assert cfg["persona_rule"] == "p10", cfg["persona_rule"]
    return D.h64(f"{cfg['persona_tag']}-persona|{int(cfg['seed'])}|{_dkey(row_key, side, cfg)}") % 10 < int(cfg["persona_p10"])


def store_digests():
    return json.load(open(DIGESTS_FILE))


def splits(roots=STORES):
    """-> {'train': [(k, r, swap)], 'eval': [...]} from each store's rows.parquet + pose_index.npz (b_ok per side)."""
    import pyarrow.parquet as pq
    out = {"train": [], "eval": []}
    nrows = {"train": 0, "eval": 0}
    for k, root in enumerate(roots):
        t = pq.read_table(f"{root}/rows.parquet", columns=["row", "split", "n_frames"]).to_pydict()
        assert t["row"] == list(range(len(t["row"]))), root
        z = np.load(f"{root}/pose_index.npz")
        b_ok, sp = z["b_ok"], z["split"]
        assert len(b_ok) == len(t["row"]) and [("eval" if x else "train") for x in sp] == t["split"], root
        for r, s in zip(t["row"], t["split"]):
            nrows[s] += 1
            for sw in (0, 1):                                     # swap 0 -> B = assistant (b_ok col 1); swap 1 -> B = user (col 0)
                if b_ok[r, 1 - sw]:
                    out[s].append((k, int(r), sw))
    return out, nrows


def make_plan(world, roots=STORES):
    sp, nrows = splits(roots)
    items = list(sp["train"])
    perm = np.random.default_rng(PLAN_SEED).permutation(len(items))
    items = [items[i] for i in perm]
    steps = -(-len(items) // world)
    return {"items": items, "steps": steps, "eval": sp["eval"], "n_train_rows": nrows["train"], "n_eval_rows": nrows["eval"]}


def rank_items(plan, rank, world):
    it = plan["items"]
    tr = [it[t * world + rank] if t * world + rank < len(it) else None for t in range(plan["steps"])]
    return tr, plan["eval"][rank::world]


def pose_arrays(m, row, pose_b, cfg):
    """B's pose (store frame grid [32, 7500], -1 outside the row) -> pdepth / pcb0 / plight on the micro's positions (module doc)."""
    P, n, N = int(m["P"]), int(row["n_frames"]), len(m["pid"])
    pd = int(cfg["pose_delay"])
    assert pd == D.CD, ("pose delay != the speech code delay (PPM: both CD)", pd, D.CD)
    off = np.arange(D.DEPTH, dtype=np.int64) * D.CBS
    pdepth = np.full((N, D.DEPTH), -1, np.int64)
    pos = P + pd + np.arange(n)
    p = np.asarray(pose_b[:, :n]).T.astype(np.int64)
    assert (p >= 0).all() and (p < D.CBS).all(), ("B pose codes missing / out of range on a qualifying track", row["row_key"])
    pdepth[pos] = p + off
    # alignment with B's Mimi: frame f of the conversation sits at P + CD + f in depth[1] (the conversation part of the micro)
    db = np.asarray(m["depth"][1])
    conv_m = db[P:, 0] >= 0
    assert np.array_equal(np.flatnonzero(conv_m) + P, pos), ("B Mimi / pose positions differ", row["row_key"])
    assert np.array_equal(db[pos] - off, np.asarray(row[f"codes_{'user' if m['swap'] else 'assistant'}"])[:, :n].T.astype(np.int64)), "B side"
    pcb0 = np.where(pdepth[:, 0] >= 0, pdepth[:, 0], -100).astype(np.int64)
    plight = PS.light_pick(np.flatnonzero(pdepth[:, 0] >= 0), float(cfg["pose_light_frac"]),
                           (cfg["light_seed"], PS.h64(row["row_key"]), 1, POSE_SEED_TAG)).astype(np.int64)
    return pdepth, pcb0, plight


def podcast_prefix_a_empty(row, swap, sib, pers, cfg):
    """= data_ppm.podcast_prefix (22091's rule, line for line: voice draw, both clips, both persona blocks, the crop to 22091's budget), then
    every channel-A segment is DROPPED before the prefix is built (module doc). -> (sysprompt.build_prefix dict, provenance)."""
    sw = int(swap)
    bside = "user" if sw else "assistant"                            # B's data side (swap 0 -> assistant)
    want_voice = voice_draw(row["row_key"], cfg, bside)              # aloe-v2pure-clip80: the run's rule (aloe_cfg); 23051's cfg -> its hash
    want_pers = persona_draw(row["row_key"], cfg, bside)             # aloe-pose-mono-8020: the persona draw (23051's cfg -> always)
    segs, pv = [], {"voice_drawn": int(want_voice), "persona_drawn": int(want_pers)}
    clips = D.voice_clips(row, sib, cfg) if want_voice else {0: None, 1: None}
    for c in (0, 1):                                                 # model channel A then B; channel c holds data track c ^ swap
        cl = clips[c ^ sw]
        if cl is not None:
            segs.append({"kind": "speech", "track": c, **cl})
    fixed = 5 + sum(int(s["L"]) + 5 for s in segs)                   # 22091's budget (A's clip included) -> B's persona crop unchanged
    assert fixed <= int(cfg["sp_cap"]), ("voice clips alone exceed the system-prompt cap", fixed)
    allow = {"both": (0, 1), "B": (1,), "A": (0,), "none": ()}[cfg["persona_sides"]] if want_pers else ()
    blocks, chans = [], []
    if pers is not None and allow:
        D._tok_init()
        for c in (0, 1):
            if c not in allow:
                continue
            av = D.PT.avail(pers.get(D.PER.SIDES[c ^ sw]))
            keys = [k for k in D.PT.SECTIONS if k in av]
            if keys:
                try:
                    ids = D.PT.encode(D.PT.block_text(av, keys))
                except AssertionError as e:
                    print(f"[data] persona block skipped row {row['row_key']} side {D.PER.SIDES[c ^ sw]}: {e}", file=sys.stderr, flush=True)
                    pv["persona_skipped"] = pv.get("persona_skipped", 0) + 1
                    continue
                blocks.append([D.PT.SOP] + ids + [D.PT.EOP]); chans.append(c)
    blocks2, cut = D.crop_blocks(blocks, int(cfg["sp_cap"]) - fixed)
    segs += [{"kind": "text", "track": c, "ids": b2} for c, b2 in zip(chans, blocks2) if b2 is not None]
    pv["dropped_A"] = [s["kind"] for s in segs if int(s["track"]) == 0]
    segs = [s for s in segs if int(s["track"]) == 1]                 # aloe-uncond: channel A empty -> A's clip / persona dropped
    pv["clips"] = [{"track": "B", "data_side": s["side"], "src_key": s["src_key"], "src_chunk": s["src_chunk"],
                    "frames": [s["w0"], s["w0"] + s["L"]], "cov": round(s["cov"], 3)} for s in segs if s["kind"] == "speech"]
    pv.update(persona=[{"track": "B", "tokens": len(s["ids"])} for s in segs if s["kind"] == "text"], persona_cut=int(cut))
    if not segs:                                                     # EMPTY system block: header + footer only (PPM's rule)
        segs = [{"kind": "text", "track": 1, "ids": []}]
    spx = D.SPR.build_prefix(row, segs, {"new_word": cfg["new_word"]})
    assert 5 <= spx["P"] <= int(cfg["sp_cap"]), ("system prompt length", spx["P"])
    assert spx["present"][0].sum() == 0 and (spx["ids"][0] == D.PAD_ID).all() and (spx["depth"][0] < 0).all(), "A owns no prefix position"
    pv["P"] = int(spx["P"])
    return spx, pv


def speech_micro_a_empty(row, node_row, sib, pers, cfg, rank, swap):
    """= data_ppm.speech_micro (22091's) with podcast_prefix_a_empty and channel A = PPM's one-sided representation at every position."""
    sw = int(swap)
    spx, pv = podcast_prefix_a_empty(row, sw, sib, pers, cfg)
    P, n = int(spx["P"]), int(row["n_frames"])
    rf = P + n if not cfg["crop_conv"] else 7500
    dc = {"new_word": cfg["new_word"], "eob_gap_s": cfg["eob_gap_s"], "wpid": cfg["wpid"], "light_seed": cfg["light_seed"], "row_frames": rf,
          "frame_light": True, "no_crop": not cfg["crop_conv"]}
    a = D.DS.row_arrays(row, dc, rank, int(node_row), "ppm", float(cfg["light_frac"]), sp=spx, swap=sw)
    sel, y, word, pad, eob, nw, cb0, light = D._b_only(a)
    T = len(a["pid"])
    ids, depth, present = a["ids"].copy(), a["depth"].copy(), a["present"].copy()
    ids[0], depth[0], present[0] = D.PAD_ID, -1, 0.0                 # PPM one-sided rows (pose_s3q.pose_micro): ids_A PAD, depth_A -1, present_A 0
    out = {"kind": "speech", "ids": ids, "depth": depth, "present": present, "pid": a["pid"], "segs": [T],
           "sel": sel, "y": y, "word": word, "pad": pad, "eob": eob, "nw": nw, "n_text": float(len(sel)), "cb0": cb0, "light": light,
           "P": P, "swap": sw}
    assert (cb0[:P + D.CD] < 0).all() and (light >= P + D.CD).all() and (sel >= P).all()
    assert T == (7500 if cfg["crop_conv"] else P + n) + D.CD, ("row length", T, P, n)
    assert np.array_equal(y, ids[1, sel + 1]), "B targets = B's own lane"
    pv.update(row_key=row["row_key"], swap=sw, chan_B_data_side=("user" if sw else "assistant"), chan_A="empty", n_frames=n, T=T)
    out["prov"] = json.dumps(pv)
    return out


class Stores:
    def __init__(self, roots=STORES):
        self.roots, self.s = list(roots), {}

    def open(self, k):
        if k not in self.s:
            root = self.roots[k]
            node = D.R3.load_node(root)
            dg = store_digests()[f"p{k}"]
            assert node["digest"] == dg["node_digest"], ("staged store differs from the build", root, node["digest"])
            st = D.R3.Rows(node)
            name = node["shards"][0]
            assert len(node["shards"]) == 1
            pz = {s: np.load(f"{root}/{name}.pose_{s}.npy", mmap_mode="r") for s in ("user", "assistant")}
            z = np.load(f"{root}/pose_index.npz")
            self.s[k] = (node, st, D.SPR.SibIndex(node, st), D.PER.PersonaSidecar(node), pz, z["b_ok"])
        return self.s[k]

    def conv(self, item, cfg, rank):
        k, r, sw = item
        node, st, sib, ps, pz, b_ok = self.open(int(k))
        row = st[int(r)]
        D.R3.check_row(row)
        bside = "user" if int(sw) else "assistant"
        assert b_ok[int(r), 0 if bside == "user" else 1], ("item's B side does not qualify", item)
        try:
            pers = ps.get(int(r), row["row_key"])
        except (FileNotFoundError, OSError):
            pers = None
        if A_EMPTY:
            m = speech_micro_a_empty(row, int(r), sib, pers, cfg, rank, swap=int(sw))
        else:
            m = D.speech_micro(row, int(r), sib, pers, cfg, rank, swap=int(sw))
        T = len(m["pid"])
        assert m["segs"] == [T] and np.array_equal(m["pid"], np.arange(T)) and m["swap"] == int(sw)
        assert T == int(m["P"]) + int(row["n_frames"]) + D.CD <= BUDGET, ("segment length", T, m["P"], row["n_frames"])
        if POSE_B:
            m["pdepth"], m["pcb0"], m["plight"] = pose_arrays(m, row, pz[bside][int(r)], cfg)
        m["row_key"] = row["row_key"]
        m["item"] = [int(k), int(r), int(sw)]
        return m


def own_counts(m):
    """(text targets, cb0 labels, light frames, pose cb0 labels, pose light frames) of one conversation."""
    if not POSE_B:                                                # no-pose ablation: 21980's (text, cb0, light) counts
        return Q.own_counts(m)
    return Q.own_counts(m) + [float((np.asarray(m["pcb0"]) >= 0).sum()), float(len(m["plight"]))]


def as_micro(m, dummy=False):
    """one conversation -> the trainer's micro (data_qqa.pack of one = its own segment) + the pose arrays; dummy: zero counts."""
    out = Q.pack([m], budget=BUDGET)
    if POSE_B:
        out["kind"] = "pose"
        out["pdepth"], out["pcb0"], out["plight"] = m["pdepth"], m["pcb0"], m["plight"]
    else:
        assert out["kind"] == "speech" and "pdepth" not in m, "no-pose micro must be a plain speech micro"
    out["own"] = np.asarray([own_counts(m)], np.float64)
    out["swap"] = int(m["swap"])
    out["item"] = json.dumps(m["item"])
    if dummy:
        out["dummy"] = 1
        out["own"] = np.zeros_like(out["own"])
    return out


class AloeDS:
    def __init__(self, items, cfg, rank, fallback, singles_at=0, roots=STORES):
        self.st, self.items, self.cfg, self.rank, self.fb, self.singles_at = Stores(roots), list(items), dict(cfg), int(rank), fallback, singles_at

    def __len__(self):
        return len(self.items)

    def __getitem__(self, t):
        it = self.items[int(t)]
        m = self.st.conv(it if it is not None else self.fb, self.cfg, self.rank)
        out = as_micro(m, dummy=it is None)
        out["step"] = int(t)
        if int(t) == self.singles_at:
            out["singles"] = [as_micro(m)]
        return out


def eval_micros(items, cfg, rank, roots=STORES):
    st = Stores(roots)
    return [as_micro(st.conv(it, cfg, rank)) for it in items]


# ================================================================================================ aloe-v2pure-ft (10-17)
# DATA = the Aloe pose-speech-vad release v2 (the 8B SAM pose v2pure fine-tune's data): ONE store row per active-speaker clip (median 12.6 s),
# B = the clip's active speaker, A absent (QQA_A_EMPTY=1: never reaches the model). Everything per clip is 22317's micro unchanged
# (speech_micro_a_empty + pose_arrays: B's system block = header / footer + B's voice clip from a later clip of the same recording on 1/3 of
# clips, B lane / Mimi / pose / every target).
# ROWS PER STEP (22317: 64 rows of <= 7,500 frames per step): the permuted clips (PLAN_SEED) are cut, in order, into ROWS of <= 7,500
# conversation frames (and <= BUDGET positions incl. each clip's own system block, upper-bounded at plan time), 64 rows per step; one epoch.
# Every clip is its OWN attention segment (data_qqa.pack: FA3 varlen block-diagonal, positions restart per clip; nothing crosses a clip).
# MICROS (maths-neutral distribution): a step's clips are spread over WORLD x R packed micros (LPT on positions, R = the fewest rounds with mean
# <= ALOE_MICRO_TOK positions, every micro <= BUDGET); the trainer forwards the R micros of a rank in rounds and normalises every term by the
# STEP's token totals over all ranks and rounds (tokens_step, as 22317), and the 0.6B decoders take their K nanostep updates per STEP over
# all of the step's light frames (joint over rounds), so the update = 22317's one-row-per-rank step over the same 64 rows.
# EVAL = every held-out clip (the 8B's held-out recordings, blake2b8('aloe-pose-v1|' + recording_id) % 10000 < 100), packed per swap into
# micros of <= QQA_EVAL_LIGHT_MAX frames so the decoder eval scores every frame (= per-clip evaluation; clips longer than that alone, subsampled
# as 22317's long rows), round-robin over ranks.
ROW_CAP = 7500
MONO = os.environ.get("ALOE_DATA", "v2pure") == "aloe-mono"      # aloe-pose-mono-8020 (module doc); default = 23051 / 24198 unchanged
assert os.environ.get("ALOE_DATA", "v2pure") in ("v2pure", "aloe-mono")
ROWS_PER_STEP = int(os.environ.get("ALOE_ROWS_PER_STEP", "64"))
MICRO_TOK = int(os.environ.get("ALOE_MICRO_TOK", "6000"))
EVAL_PACK = int(os.environ.get("QQA_EVAL_LIGHT_MAX", "2048"))
MIN_CLIP = 2                                                      # the PPM row builder's floor (data_s3q.row_arrays asserts >= 2 frames)


def want_voice(row_key, cfg, side=None):
    """= podcast_prefix_a_empty's voice draw (the same function), used only to upper-bound a clip's positions at plan time."""
    return voice_draw(row_key, cfg, side)


# aloe-v2pure-clip80 (10-18): the ROWS (which clips form each step's 64 rows, hence the 200 steps) and the EVAL packing are cut with 23051's
# position bound (its 1/3 voice draw), so they are 23051's exactly; this run's own draw only sets the per-clip bound used to spread a step's
# clips over ranks x rounds (memory only, maths-neutral) and is asserted <= BUDGET on every eval micro. For 23051's cfg both bounds coincide
# and the plan is 23051's byte for byte.
ROWS_RULE = {"seed": 20261011, "voice_mod": 3, "clip_max": 125}   # = 23051's data cfg seed / voice_mod / clip_max


def rows_digest(pl):
    import hashlib
    return hashlib.sha256(json.dumps([pl["items"], pl["rows"], pl["eval_micros"]]).encode()).hexdigest()[:16]


def tok_ub(n, row_key, cfg, side=None):
    """upper bound of a clip micro's positions T = P + n + CD: P = 5 (empty block) or 5 + (L + 5) with B's voice clip L <= clip_max.
    aloe-pose-mono-8020 (MONO): an item with persona text drawn -> P <= sp_cap (asserted in podcast_prefix_a_empty)."""
    if MONO and cfg["persona_sides"] in ("both", "B") and persona_draw(row_key, cfg, side):
        return int(n) + D.CD + int(cfg["sp_cap"])
    return int(n) + D.CD + 5 + ((int(cfg["clip_max"]) + 5) if want_voice(row_key, cfg, side) else 0)


def tok_ub_any(n):
    """aloe-pose-mono-8020: a bound valid under ANY prompt rule (P <= sp_cap): the eval packing, shared by the run's eval and the step-0 eval
    under 23051's rule (persona on every item)."""
    return int(n) + D.CD + 2000


def splits_v2(roots=STORES):
    """-> {'train': [(k, r, swap, n, row_key)], 'eval': [...]}; exactly one qualifying B side per v2pure row (asserted)."""
    if MONO and os.environ.get("ALOE_MONO_ITEMS"):                      # aloe-pose-rosie (10-27): one speaker's items (build_items.py), same tuples
        j = json.load(open(os.environ["ALOE_MONO_ITEMS"]))
        return {s: [(int(k), int(r), int(sw), int(n), str(key)) for k, r, sw, n, key in j[s]] for s in ("train", "eval")}
    import pyarrow.parquet as pq
    out = {"train": [], "eval": []}
    for k, root in enumerate(roots):
        t = pq.read_table(f"{root}/rows.parquet", columns=["row", "split", "n_frames", "row_key"]).to_pydict()
        assert t["row"] == list(range(len(t["row"]))), root
        z = np.load(f"{root}/pose_index.npz")
        b_ok, sp = z["b_ok"], z["split"]
        assert len(b_ok) == len(t["row"]) and [("eval" if x else "train") for x in sp] == t["split"], root
        if MONO:                                                  # aloe-pose-mono-8020: one item per qualifying track, 22091's order / rule
            for r, s, n, key in zip(t["row"], t["split"], t["n_frames"], t["row_key"]):
                for sw in (0, 1):                                 # swap 0 -> B = assistant (b_ok col 1); swap 1 -> B = user (col 0)
                    if b_ok[r, 1 - sw]:
                        if int(n) < MIN_CLIP:
                            out["skipped_" + s] = out.get("skipped_" + s, 0) + 1
                            continue
                        out[s].append((k, int(r), sw, int(n), key))
            continue
        assert (b_ok.sum(1) == 1).all(), ("v2pure: exactly one B side per clip row", root)
        for r, s, n, key in zip(t["row"], t["split"], t["n_frames"], t["row_key"]):
            sw = 0 if b_ok[r, 1] else 1                           # swap 0 -> B = assistant (speaker_1), swap 1 -> B = user (speaker_2)
            if int(n) < MIN_CLIP:                                 # 1-frame clips (1 train, 1 eval): data_s3q.row_arrays needs >= 2 frames
                out["skipped_" + s] = out.get("skipped_" + s, 0) + 1
                continue
            out[s].append((k, int(r), sw, int(n), key))
    return out


def _lpt(toks, nb):
    import heapq
    h = [(0, b) for b in range(nb)]
    bins = [[] for _ in range(nb)]
    load = [0] * nb
    for i in sorted(range(len(toks)), key=lambda i: (-toks[i], i)):
        l_, b = heapq.heappop(h)
        bins[b].append(i); load[b] = l_ + toks[i]
        heapq.heappush(h, (load[b], b))
    return bins, load


def make_plan_v2(world, cfg, roots=STORES):
    """the identical-on-every-rank plan (module section doc). -> dict(items, rows, steps, step_micros[t][rank] = [[item idx per round]], R[t],
    eval_micros (list of [item idx]), eval_items, counts)."""
    sp = splits_v2(roots)
    tr = sp["train"]
    assert int(cfg["seed"]) == ROWS_RULE["seed"] and int(cfg["clip_max"]) == ROWS_RULE["clip_max"], ("data cfg != 23051's seed / clip_max", cfg)
    EP = int(os.environ.get("ALOE_EPOCHS", "1"))                     # aloe-pose-rosie (10-27): E epochs = E seeded permutations (seed + e), rows cut per epoch
    perm = np.concatenate([np.random.default_rng(PLAN_SEED + e).permutation(len(tr)) for e in range(EP)])   # epoch 0 = the 1-epoch plan's order
    items = [tr[i][:3] for i in perm]
    ns = [tr[i][3] for i in perm]
    sd = ["user" if tr[i][2] else "assistant" for i in perm]           # B's data side (per-item draws)
    tk = [tok_ub(tr[i][3], tr[i][4], cfg, sd[j]) for j, i in enumerate(perm)]   # this run's bound (micro distribution)
    if MONO:                                                          # aloe-pose-mono-8020: rows cut with this run's own bound (no reference rows)
        assert int(cfg["sp_cap"]) == 2000, ("tok_ub_any assumes sp_cap 2000", cfg["sp_cap"])
        tkr = list(tk)
    else:
        tkr = [tok_ub(tr[i][3], tr[i][4], ROWS_RULE) for i in perm]   # 23051's bound (rows: = 23051's rows)
    rows, cur, cf, ct = [], [], 0, 0
    for i in range(len(items)):
        assert ns[i] <= ROW_CAP and tk[i] <= BUDGET and tkr[i] <= BUDGET, ("clip larger than a row", items[i], ns[i], tk[i], tkr[i])
        if cur and (cf + ns[i] > ROW_CAP or ct + tkr[i] > BUDGET or (i > 0 and i % len(tr) == 0)):   # (an epoch boundary also ends a row)
            rows.append(cur); cur, cf, ct = [], 0, 0
        cur.append(i); cf += ns[i]; ct += tkr[i]
    if cur:
        rows.append(cur)
    steps = -(-len(rows) // ROWS_PER_STEP)
    step_micros, R_of, max_tok = [], [], 0
    for t in range(steps):
        if MONO and ROWS_PER_STEP <= world:
            # aloe-pose-mono-8020 MICRO LAYOUT 'row_per_rank' (maths-neutral, = 22317's one row per rank per step): the long Aloe items (median
            # 4,630 frames, 37% full 7,500-frame rows) would leave most of LPT's WORLD x R bins empty (dummy micros = wasted forwards); every
            # row already fits one micro (<= BUDGET positions by construction), so rank j's single micro = the step's row j (spare ranks of the
            # last step: dummies). tokens_step normalisers / joint decoder nanosteps unchanged -> the same update as any layout.
            rr = rows[t * ROWS_PER_STEP:(t + 1) * ROWS_PER_STEP]
            per = [[sorted(rr[j])] if j < len(rr) else [[]] for j in range(world)]
            step_micros.append(per); R_of.append(1); max_tok = max(max_tok, max(sum(tk[i] for i in rw) for rw in rr))
            continue
        cl = [i for rw in rows[t * ROWS_PER_STEP:(t + 1) * ROWS_PER_STEP] for i in rw]
        tot = sum(tk[i] for i in cl)
        R = max(1, -(-tot // (world * MICRO_TOK)))
        while True:
            bins, load = _lpt([tk[i] for i in cl], world * R)
            if max(load) <= BUDGET:
                break
            R += 1
        order = sorted(range(world * R), key=lambda b: (-load[b], b))       # round r = the next WORLD bins by load (balanced rounds)
        per = [[[] for _ in range(R)] for _ in range(world)]
        for j, b in enumerate(order):
            per[j % world][j // world] = sorted(cl[x] for x in bins[b])
        step_micros.append(per); R_of.append(R); max_tok = max(max_tok, max(load))
    ev = sp["eval"]
    eitems = [e[:3] for e in ev]
    emic = []
    for sw in (0, 1):
        cur, cf, ct = [], 0, 0
        for i, e in enumerate(ev):
            if e[2] != sw:
                continue
            n_, t_ = e[3], (tok_ub_any(e[3]) if MONO else tok_ub(e[3], e[4], ROWS_RULE))   # 23051's eval packing (MONO: any-rule bound)
            if cur and (cf + n_ > EVAL_PACK or ct + t_ > BUDGET):
                emic.append(cur); cur, cf, ct = [], 0, 0
            cur.append(i); cf += n_; ct += t_
        if cur:
            emic.append(cur)
    etk = [sum(tok_ub(ev[i][3], ev[i][4], cfg, "user" if ev[i][2] else "assistant") for i in g) for g in emic]   # this run's draw: every eval micro still fits
    if MONO:
        assert max(sum(tok_ub_any(ev[i][3]) for i in g) for g in emic) <= BUDGET, "eval micro over the budget under the any-rule bound"
    assert max(etk) <= BUDGET, ("eval micro over the position budget under this run's voice draw", max(etk))
    return {"items": items, "n_frames": ns, "tok_ub": tk, "rows": rows, "steps": steps, "step_micros": step_micros, "R": R_of,
            "max_micro_tok_ub": max_tok, "max_eval_tok_ub": max(etk), "eval_items": eitems, "eval_micros": emic, "n_train_items": len(items), "n_eval_items": len(eitems),
            "train_frames": int(sum(ns)), "eval_frames": int(sum(e[3] for e in ev)), "n_rows": len(rows),
            "skipped_lt2": [sp.get("skipped_train", 0), sp.get("skipped_eval", 0)]}


def plan_digest(pl):
    import hashlib
    h = hashlib.sha256()
    h.update(json.dumps([pl["items"], pl["rows"], pl["step_micros"], pl["eval_micros"]]).encode())
    return h.hexdigest()[:16]


def pack_micro(ms, dummy=False):
    """several standalone clip micros -> ONE packed micro (data_qqa.pack: own attention segment per clip) + B's pose arrays offset per clip."""
    out = Q.pack(ms, budget=BUDGET)
    T = [len(m["pid"]) for m in ms]
    o = np.concatenate([[0], np.cumsum(T)[:-1]]).astype(np.int64)
    N = int(sum(T))
    if POSE_B:
        out["kind"] = "pose"
        out["pdepth"] = np.concatenate([m["pdepth"] for m in ms], 0)
        out["pcb0"] = np.concatenate([m["pcb0"] for m in ms])
        out["plight"] = np.concatenate([m["plight"] + ok for m, ok in zip(ms, o)])
        assert out["pdepth"].shape == (N, D.DEPTH) and len(out["pcb0"]) == N
        for m, ok, t in zip(ms, o, T):
            assert (m["plight"] < t).all() and (m["plight"] >= int(m["P"]) + D.CD).all()
    out["own"] = np.asarray([own_counts(m) for m in ms], np.float64)
    sws = {int(m["swap"]) for m in ms}
    out["swap"] = sws.pop() if len(sws) == 1 else -1
    out["item"] = json.dumps([m["item"] for m in ms])
    if dummy:
        out["dummy"] = 1
        out["own"] = np.zeros_like(out["own"])
    return out


class StepDS:
    """one rank's training stream: item s = the list of this rank's R packed micros of plan step start + s (empty micro -> a dummy)."""

    def __init__(self, items, step_lists, cfg, rank, fallback, start=0, singles_at=0, roots=STORES):
        self.st, self.items, self.sl, self.cfg, self.rank = Stores(roots), list(items), step_lists, dict(cfg), int(rank)
        self.fb, self.start, self.singles_at = fallback, int(start), singles_at

    def __len__(self):
        return len(self.sl) - self.start

    def __getitem__(self, i):
        s = self.start + int(i)
        out = []
        for r, grp in enumerate(self.sl[s]):
            if grp:
                ms = [self.st.conv(self.items[j], self.cfg, self.rank) for j in grp]
                b = pack_micro(ms)
                if s == self.singles_at and r == 0:
                    b["singles"] = [as_micro(m) for m in ms]
            else:
                b = pack_micro([self.st.conv(self.fb, self.cfg, self.rank)], dummy=True)
            b["step"] = s
            out.append(b)
        return out


def eval_micros_v2(eitems, groups, cfg, rank):
    st = Stores()
    return [pack_micro([st.conv(eitems[j], cfg, rank) for j in g]) for g in groups]


# ================================================================================================ aloe-v2pure-clip80 (10-18)
PSTAT_KEYS = ("items", "voice_drawn", "voice_valid", "empty_block", "persona_B", "ref_voice_drawn", "ref_voice_valid", "ref_persona_B_avail",
              "ref_empty_block")


def prompt_stats(items, cfg, rank, world):
    """this rank's share (items[rank::world]) of the system-prompt draws, summed (allsum over ranks -> the realised fractions):
    this run (cfg): voice clip drawn, clip valid (drawn and data_ppm.voice_clips found one for channel B), empty system block (no clip and
    no persona text), persona block on B; and 23051's rule on the same items (ROWS_RULE's 1/3 draw, persona 'both': clip drawn, valid,
    B persona text available in the sidecar, empty block). voice_clips does not depend on the draw rule, so 'valid' = drawn x the same clip."""
    st = Stores()
    out = np.zeros(len(PSTAT_KEYS), np.float64)
    for k, r, sw in items[rank::world]:
        node, rows, sib, ps, pz, b_ok = st.open(int(k))
        row = rows[int(r)]
        key = row["row_key"]
        dr, drr = bool(voice_draw(key, cfg)), bool(voice_draw(key, ROWS_RULE))
        ok = (dr or drr) and D.voice_clips(row, sib, cfg)[1 ^ int(sw)] is not None
        try:
            pers = ps.get(int(r), key)
        except (FileNotFoundError, OSError):
            pers = None
        pav = bool(pers is not None and D.PT.avail(pers.get(D.PER.SIDES[1 ^ int(sw)])))
        pb = pav and cfg["persona_sides"] in ("both", "B")
        out += [1, dr, dr and ok, not (dr and ok) and not pb, pb, drr, drr and ok, pav, not (drr and ok) and not pav]
    return out


# ================================================================================================ aloe-pose-mono-8020 (10-18)
PSTAT_MONO_KEYS = ("items", "voice_drawn", "persona_drawn", "clip", "persona", "both", "clip_only", "persona_only", "empty",
                   "ref_clip", "ref_persona", "ref_both", "ref_clip_only", "ref_persona_only", "ref_empty")


def prompt_stats_mono(items, cfg, ref_cfg, rank, world):
    """this rank's share (items[rank::world]) of the REALISED system blocks, summed (allsum over ranks -> fractions), built with the micro's
    own prefix builder (podcast_prefix_a_empty: exactly what the model sees): voice clip drawn, persona drawn (this run's two independent
    draws), clip present (drawn and data_ppm.voice_clips found one for B), persona text present (drawn and B has persona text in the
    sidecar), and the four cells both / clip only / persona only / empty; the same four cells under the reference rule (23051's cfg)."""
    st = Stores()
    out = np.zeros(len(PSTAT_MONO_KEYS), np.float64)
    for k, r, sw in items[rank::world]:
        node, rows, sib, ps, pz, b_ok = st.open(int(k))
        row = rows[int(r)]
        try:
            pers = ps.get(int(r), row["row_key"])
        except (FileNotFoundError, OSError):
            pers = None
        cells = []
        for c_ in (cfg, ref_cfg):
            _, pv = podcast_prefix_a_empty(row, int(sw), sib, pers, c_)
            cl, pe = len(pv["clips"]) > 0, len(pv["persona"]) > 0
            cells.append((pv["voice_drawn"], pv["persona_drawn"], cl, pe))
        (vd, pd, cl, pe), (_, _, rcl, rpe) = cells
        out += [1, vd, pd, cl, pe, cl and pe, cl and not pe, pe and not cl, not cl and not pe,
                rcl, rpe, rcl and rpe, rcl and not rpe, rpe and not rcl, not rcl and not rpe]
    return out
