"""aloe-ft-32-samsep PERSONA DATA (10-24, user via main ~07:40 PT: "prep being able to add some text string as a persona to the start of some of
the rows with some fraction of rows"; user spec update 10-24: 50 % of TRAIN rows, prose only, independent of the voice clip) =
data_aloe_v2vox.py (the data path of job 29345, imported UNCHANGED; data_aloe_v2 / data_ppm / data_qqa / data_aloe stay byte-identical and
are sha-asserted by the trainer) + a persona text block in the system prompt of a seeded share of (row, orientation) TRAIN items.

KNOBS (env, read at import; identical on every rank and DataLoader worker):
  ALOE_PERSONA_P         TARGET share of ALL train items that carry a persona (default 0 = job 29345's recipe exactly: every micro is built by
                         the unchanged v2vox code path, bit-identical, tests/identity_p0.py). Only ELIGIBLE items can carry one (the side's
                         sidecar status ok and at least one kept section with non-empty prose; ~82 % of prompts have persona prose), so an
                         eligible item draws with p_eff = P / eligible_share (eligible_share over the plan's train items, computed from the
                         stores at start: persona_setup); P x N_train items then carry one in expectation. p_eff > 1 -> abort.
  ALOE_PERSONA_SIDE      B (default: the trained side's speaker, block on channel B) | A (channel A's speaker, block on channel A)
  ALOE_PERSONA_SECTIONS  keep-list (comma) of sidecar prose sections, default 'persona' (user 10-24: ONLY the persona prose, never the
                         starting context). Several kept sections are joined with a blank line, in PPM's section order, with NO labels.
  ALOE_PERSONA_TEXT      optional: a fixed string, or the path of a UTF-8 text file (stripped), used INSTEAD of every row's sidecar prose
                         (every item is then eligible).
  ALOE_PERSONA_SEED      the draw's seed (default 20261011)
  ALOE_PERSONA_EVAL      0 (default): held-out eval micros WITHOUT persona (= job 29345's eval, bit-identical; curves comparable) | 1: the eval
                         items draw with the same rule (eligible_share / p_eff of the TRAIN items)
DRAW  per (row_key, orientation) item, its OWN hash stream: drawn iff eligible and h64('aloe-persona|<seed>|<row_key>|<swap>') % 10^6 < p_eff x 10^6.
      The B voice-clip draw (data_aloe_v2.want_voice: h64('aloe-v2-voice|20261011|<row_key>|<swap>') % 10 < 3) is untouched and independent.
RENDER = data_ppm.podcast_prefix's persona slot (the block on the speaker's own model channel; channel c holds data side SIDES[c ^ swap]):
      <|im_start|> system \\n  [B voice clip]  <sop> tok(prose) <eop>  <|im_end|> \\n          (the clip only on the 30 % voice items)
      <sop> 151680 / <eop> 151681 = PPM's persona wrapper ids; tok = Qwen3-32B (data_ppm._tok_init; persona_text_s3q.encode: plain text ids,
      a prose holding a special token -> no block for that item, logged, as PPM). PPM's own block text is
      persona_text_s3q.block_text = "\\n\\n".join(HEADER + "\\n" + prose) with HEADER "Persona" / "Starting context" / "Earlier conversation
      memory": those label words are NOT emitted here (user 10-24) - the block holds ONLY the prose as written.
      NEVER TRUNCATED: a block that would not fit sp_cap 2,000 - (header + footer + clip) asserts (PPM's crop_blocks is not used; the
      plan-wide length report, tests/persona_stats.py, shows every item fits). Prompt-only: no target / cb0 label / light frame in the prefix
      (sel >= P, cb0[:P + CD] < 0, light >= P + CD asserted), so no loss on the persona. The conversation is never cropped (crop_conv False,
      as the recipe): T = P + n + CD <= BUDGET 9,502 (asserted by data_qqa.pack).
NO BLOCK (item not drawn / not eligible / persona off for eval): the micro is the UNCHANGED v2vox code path's (data_aloe_v2vox.Stores.conv).
CHANNEL A in the conversation: emptied exactly as data_aloe_v2vox.speech_micro_v2vox (PAD ids, depth -1, present 0 at p >= P, vin (P, n));
      with SIDE B the prefix carries nothing on A (asserted, as v2vox)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import data_aloe_v2vox as V  # noqa: E402   (job 29345's data module, unchanged copy; sha asserted by the trainer)

A2, D, Q = V.A2, V.D, V.Q
PER, PT = D.PER, D.PT
STORES, SRC_STORES, BUDGET = V.STORES, V.SRC_STORES, V.BUDGET
FLAC_ROOT, INDEX, SAMSEP, RAWGAIN_PLAN_DIGEST, GAIN_MODE, RAWGAIN = V.FLAC_ROOT, V.INDEX, V.SAMSEP, V.RAWGAIN_PLAN_DIGEST, V.GAIN_MODE, V.RAWGAIN
aloe_cfg, eval_cfg, splits, make_plan, rank_items = V.aloe_cfg, V.eval_cfg, V.splits, V.make_plan, V.rank_items
rawgain, a_gain, vox_index, a_entry, node_items = V.rawgain, V.a_gain, V.vox_index, V.a_entry, V.node_items


def _text_knob(v):
    if not v:
        return None
    if os.path.isfile(v):
        v = open(v, encoding="utf-8").read()
    v = v.strip()
    assert v, "ALOE_PERSONA_TEXT is empty"
    return v


P_TARGET = float(os.environ.get("ALOE_PERSONA_P", "0") or 0)
SIDE = os.environ.get("ALOE_PERSONA_SIDE", "B") or "B"
SECTIONS = tuple(s for s in (os.environ.get("ALOE_PERSONA_SECTIONS", "persona") or "persona").split(",") if s)
TEXT = _text_knob(os.environ.get("ALOE_PERSONA_TEXT", ""))
SEED = int(os.environ.get("ALOE_PERSONA_SEED", "20261011") or 20261011)
EVAL_ON = (os.environ.get("ALOE_PERSONA_EVAL", "0") or "0") == "1"
assert 0.0 <= P_TARGET <= 1.0, ("ALOE_PERSONA_P must be in [0, 1]", P_TARGET)
assert SIDE in ("A", "B"), ("ALOE_PERSONA_SIDE must be A or B", SIDE)
assert SECTIONS and all(s in PT.SECTIONS for s in SECTIONS) and len(set(SECTIONS)) == len(SECTIONS), ("ALOE_PERSONA_SECTIONS", SECTIONS, PT.SECTIONS)
KEEP = tuple(k for k in PT.SECTIONS if k in SECTIONS)            # PPM's section order
CH = 1 if SIDE == "B" else 0                                     # model channel of the block
ON = P_TARGET > 0.0
_SETUP = {}


def cfg_view():
    """the resolved persona config (trainer CFG['persona'], logged + diffed)."""
    out = {"p_target": P_TARGET, "side": SIDE, "sections": list(KEEP), "labels": "none (prose only)", "seed": SEED, "eval": EVAL_ON,
           "text": None if TEXT is None else {"chars": len(TEXT), "sha256_16": __import__("hashlib").sha256(TEXT.encode()).hexdigest()[:16]},
           "draw": "eligible and h64('aloe-persona|seed|row_key|swap') % 1e6 < p_eff x 1e6", "slot": "<|im_start|>system\\n [clip B] <sop> prose <eop> <|im_end|>\\n"}
    if _SETUP:
        out.update({k: _SETUP[k] for k in ("eligible_share", "p_eff", "n_train_items", "n_eligible")})
    return out


# ------------------------------------------------------------------------------------------------ eligibility + draw
def prose_of(pers, swap):
    """-> the block's text for this item's side (None: no kept section with prose / status not ok). Prose only, no labels."""
    if TEXT is not None:
        return TEXT
    av = PT.avail((pers or {}).get(PER.SIDES[CH ^ int(swap)]))
    keys = [k for k in KEEP if k in av]
    if not keys:
        return None
    return "\n\n".join(av[k] for k in keys)


def hash_u(row_key, swap):
    return D.h64(f"aloe-persona|{SEED}|{row_key}|{int(swap)}") % 1000000


def _sidecar_table(root):
    """{node row: {side: fields}} of one store's sidecar (whole-file read; rows in node order, row_key asserted)."""
    import pyarrow as pa
    import pyarrow.ipc as ipc
    node = D.R3.load_node(root)
    out = []
    for s in node["shards"]:
        t = ipc.open_file(pa.memory_map(PER.sidecar_path(node["flat_root"], s), "r")).read_all()
        cols = {c: t.column(c).to_pylist() for c in t.column_names}
        for i in range(t.num_rows):
            out.append({"row_key": cols["row_key"][i], **{sd: {f: cols[f"{sd}_{f}"][i] for f in PER.FIELDS} for sd in PER.SIDES}})
    return node, out


def eligible_items(items, roots=None):
    """-> list of bools aligned with items: the item's side has prose under the knobs (TEXT: always)."""
    if TEXT is not None:
        return [True] * len(items)
    roots = roots or STORES
    tabs = {}
    out = []
    for k, r, sw in items:
        if k not in tabs:
            tabs[k] = _sidecar_table(roots[int(k)])[1]
        out.append(prose_of(tabs[k][int(r)], sw) is not None)
    return out


def persona_setup(plan=None, roots=None):
    """eligible share over the plan's TRAIN items (world-independent: every train first-chunk row x 2 orientations) and p_eff = P / share.
    Called by the trainer before the DataLoaders exist (forked workers inherit it); lazily by any process that draws without it."""
    if _SETUP or not ON:
        return dict(_SETUP)
    plan = plan or make_plan(64, roots=roots or STORES)
    el = eligible_items(plan["items"], roots)
    n, ne = len(el), int(sum(el))
    share = ne / max(n, 1)
    p_eff = P_TARGET / share if share > 0 else float("inf")
    assert p_eff <= 1.0, ("ALOE_PERSONA_P above the eligible share: cannot reach the target", P_TARGET, share)
    _SETUP.update(eligible_share=share, p_eff=p_eff, n_train_items=n, n_eligible=ne)
    return dict(_SETUP)


def drawn(row_key, swap):
    if not ON:
        return False
    if not _SETUP:
        persona_setup()
    return hash_u(row_key, swap) < _SETUP["p_eff"] * 1000000


# ------------------------------------------------------------------------------------------------ prefix + micro
def prefix_persona(row, swap, sib, text, cfg):
    """= data_aloe_v2.prefix_v2 (B's own clip on the seeded 30 %, PPM voice_clips rule) + the persona block in PPM's persona slot (after the
    clips, before the footer) on model channel CH. -> (sysprompt.build_prefix dict, provenance)."""
    sw = int(swap)
    wv = A2.want_voice(row["row_key"], sw, cfg["voice_seed"])
    pv = {"voice_drawn": int(wv)}
    segs = []
    if wv:
        cl = D.voice_clips(row, sib, cfg)[1 ^ sw]                   # model channel B (c = 1) holds data track 1 ^ swap (podcast_prefix rule)
        if cl is not None:
            segs.append({"kind": "speech", "track": 1, **cl})
    pv["clips"] = [{"track": "AB"[s["track"]], "data_side": s["side"], "src_key": s["src_key"], "src_chunk": s["src_chunk"],
                    "frames": [s["w0"], s["w0"] + s["L"]], "cov": round(s["cov"], 3)} for s in segs]
    pv["voice_clip"] = int(bool(segs))
    fixed = 5 + sum(int(s["L"]) + 5 for s in segs)
    assert fixed <= int(cfg["sp_cap"]), ("voice clip exceeds the system-prompt cap", fixed)
    D._tok_init()
    ids = PT.encode(text)                                          # AssertionError on a special / control id (caller: no block, logged)
    blk = [PT.SOP] + ids + [PT.EOP]
    budget = int(cfg["sp_cap"]) - fixed
    assert len(blk) <= budget, ("persona block does not fit the system-prompt cap: never truncated (report: tests/persona_stats.py)",
                                row["row_key"], sw, len(blk), budget)
    segs.append({"kind": "text", "track": CH, "ids": blk})
    pv.update(persona=[{"track": "AB"[CH], "tokens": len(blk)}], persona_cut=0)
    spx = D.SPR.build_prefix(row, segs, {"new_word": cfg["new_word"]})
    assert 5 < spx["P"] <= int(cfg["sp_cap"]), ("system prompt length", spx["P"])
    if CH == 1:
        assert (spx["ids"][0] == D.PAD_ID).all() and not (np.asarray(spx["depth"])[0] >= 0).any(), "channel A must carry nothing in the prefix"
    pv["P"] = int(spx["P"])
    return spx, pv


def speech_micro_persona(row, node_row, sib, text, cfg, rank, swap):
    """= data_aloe_v2.speech_micro_v2 with prefix_persona in place of prefix_v2, then channel A emptied in the conversation exactly as
    data_aloe_v2vox.speech_micro_v2vox. Every line below the prefix = those two functions' (same calls, same asserts)."""
    sw = int(swap)
    spx, pv = prefix_persona(row, sw, sib, text, cfg)
    P, n = int(spx["P"]), int(row["n_frames"])
    rf = P + n if not cfg["crop_conv"] else 7500
    dc = {"new_word": cfg["new_word"], "eob_gap_s": cfg["eob_gap_s"], "wpid": cfg["wpid"], "light_seed": cfg["light_seed"], "row_frames": rf,
          "frame_light": True, "no_crop": not cfg["crop_conv"]}
    a = D.DS.row_arrays(row, dc, rank, int(node_row), "ppm", float(cfg["light_frac"]), sp=spx, swap=sw)
    sel, y, word, pad, eob, nw, cb0, light = D._b_only(a)
    T = len(a["pid"])
    m = {"kind": "speech", "ids": a["ids"], "depth": a["depth"], "present": a["present"], "pid": a["pid"], "segs": [T],
         "sel": sel, "y": y, "word": word, "pad": pad, "eob": eob, "nw": nw, "n_text": float(len(sel)), "cb0": cb0, "light": light,
         "P": P, "swap": sw}
    assert (cb0[:P + D.CD] < 0).all() and (light >= P + D.CD).all() and (sel >= P).all()      # prompt-only: no loss on the persona
    assert T == (7500 if cfg["crop_conv"] else P + n) + D.CD, ("row length", T, P, n)
    pv.update(row_key=row["row_key"], swap=sw, chan_B_data_side=("user" if sw else "assistant"), n_frames=n, T=T, n_light=int(len(light)))
    m["prov"] = json.dumps(pv)
    m["voice_clip"] = pv["voice_clip"]
    m["voice_drawn"] = pv["voice_drawn"]
    # ---- = data_aloe_v2vox.speech_micro_v2vox (channel A emptied in the conversation; Voxtral latents at P .. P + n - 1)
    assert (m["depth"][0, :P] < 0).all() and (CH == 0 or (m["ids"][0, :P] == D.PAD_ID).all()), "the prefix carries no A codes (and no A text with side B)"
    m["ids"] = m["ids"].copy(); m["depth"] = m["depth"].copy(); m["present"] = m["present"].copy()
    m["ids"][0, P:] = D.PAD_ID
    m["depth"][0] = -1
    m["present"][0, P:] = 0.0
    m["vin"] = (P, n)
    pv = json.loads(m["prov"]); pv["chan_A"] = "voxtral_latents_only"; m["prov"] = json.dumps(pv)
    return m


class Stores(V.Stores):
    """= data_aloe_v2vox.Stores + the persona sidecar. conv(persona=None): the module default (training draw); False: off (= v2vox exactly)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ps = {}

    def sidecar(self, k):
        if k not in self.ps:
            node = self.open(k)[0]
            self.ps[k] = PER.PersonaSidecar(node)
        return self.ps[k]

    def persona_text(self, item):
        """-> (drawn, text or None) of a (k, r, swap) item under the knobs."""
        k, r, sw = (int(x) for x in item)
        node, st, sib = self.open(k)
        row_key = _row_key(st, r)
        if not drawn(row_key, sw):
            return 0, None
        if TEXT is not None:
            return 1, TEXT
        pers = self.sidecar(k).get(r, row_key)
        return 1, prose_of(pers, sw)

    def conv(self, item, cfg, rank, audio=True, persona=None):
        use = ON if persona is None else (bool(persona) and ON)
        dr, text = self.persona_text(item) if use else (0, None)
        if text is None:                                             # = job 29345's micro, its own code path
            m = super().conv(item, cfg, rank, audio=audio)
            m["persona_drawn"], m["persona_block"], m["persona_tokens"] = int(dr), 0, 0
            return m
        import audio_s4q as AU
        import rowcache_vox as RC
        k, r, sw = item
        node, st, sib = self.open(int(k))
        row = st[int(r)]
        D.R3.check_row(row)
        assert int(row["chunk_index"]) == 0, ("v2: first chunks only", row["row_key"], row["chunk_index"])
        try:
            m = speech_micro_persona(row, int(r), sib, text, cfg, rank, int(sw))
        except AssertionError as e:
            if "special / control id" not in str(e):
                raise
            print(f"[data] persona block skipped row {row['row_key']} swap {sw}: {e}", file=sys.stderr, flush=True)
            m = super().conv(item, cfg, rank, audio=audio)
            m["persona_drawn"], m["persona_block"], m["persona_tokens"] = int(dr), 0, 0
            return m
        # ---- = data_aloe_v2vox.Stores.conv below the micro (segment asserts, A entry / gain, A waveform)
        T = len(m["pid"])
        assert m["segs"] == [T] and np.array_equal(m["pid"], np.arange(T)) and m["swap"] == int(sw)
        assert T == int(m["P"]) + int(row["n_frames"]) + D.CD <= BUDGET, ("segment length", T, m["P"], row["n_frames"])
        aside, a = a_entry(item, row)
        g_db = a_gain(a)
        m["aref"] = {"row_key": row["row_key"], "side": aside, "flac": a["flac"], "gain_db": g_db, "frame0": int(a["frame0"]),
                     "n": int(a["n_frames"]), "kind": a["kind"], "gain_mode": GAIN_MODE, "gain_mimi_db": float(a["gain_db"])}
        if audio:
            path = f"{FLAC_ROOT}/{a['flac']}"
            if a["kind"] == "gcs":
                assert os.path.getsize(path) == int(a["flac_bytes"]), ("A FLAC missing / wrong size on this node", path)
            else:
                assert a["kind"] == "mimidec" and int(a["audio_frame0"]) == 0 and os.path.getsize(path) > 0, ("A mimidec file", path)
            m["wav"] = RC.row_audio({"flac": path, "frame0": int(a["audio_frame0"]), "n_frames": int(a["n_frames"]), "row_key": row["row_key"]},
                                    g_db)
            assert m["wav"].dtype == np.float32 and m["wav"].shape == (int(a["n_frames"]) * AU.SAMPLES_PER_FRAME_24K,)
        m["row_key"] = row["row_key"]
        m["item"] = [int(k), int(r), int(sw)]
        m["persona_drawn"], m["persona_block"] = 1, 1
        m["persona_tokens"] = int(json.loads(m["prov"])["persona"][0]["tokens"])
        return m


def _row_key(st, r):
    """the row_key of node row r from the words table without materialising the row (codes / pieces)."""
    k, rr = int(st.node["shard_of"][r]), int(st.node["row_of"][r])
    st._open(k)
    return st.tb[k].column("row_key")[rr].as_py()


def as_micro(m, dummy=False):
    out = V.as_micro(m, dummy=dummy)                                 # job 29345's micro (v2vox -> v2 -> data_qqa.pack)
    for k_ in ("persona_drawn", "persona_block", "persona_tokens"):
        out[k_] = int(m.get(k_, 0)) if not dummy else 0
    return out


class AccDS(V.AccDS):
    """= data_aloe_v2vox.AccDS (one physical rank's stream of logical ranks' micros) with this module's Stores / as_micro."""

    def __getitem__(self, i):
        if self.st is None:
            self.st = Stores()
        t, g = divmod(int(i), self.G)
        L = self.lr[g]
        pad = L is None
        it = None if pad else self.rows[g][t]
        rk = self.lr[0] if pad else L
        m = self.st.conv(it if it is not None else self.fb, self.cfg, rk)
        out = as_micro(m, dummy=it is None)
        out["step"], out["slot"], out["lrank"], out["padslot"] = t, g, (-1 if pad else int(L)), int(pad)
        if t == self.singles_at:
            s1 = as_micro(m)
            s1.pop("wav", None)
            out["singles"] = [s1]
        return out


def eval_micros(items, cfg, rank):
    st = Stores()
    return [as_micro(st.conv(it, cfg, rank, persona=EVAL_ON)) for it in items]


# ------------------------------------------------------------------------------------------------ checks + plan report (trainer start, tests)
_SHIFT_KEYS = ("y", "word", "pad", "eob", "nw")


def shift_check(m0, m1):
    """m0 = the no-persona micro (v2vox path), m1 = the same item with the persona knobs. No block: m1 == m0 on every array. Block: the
    conversation part of m1 = m0's shifted by d = P1 - P0 (ids / depth / present / cb0 / targets / light / vin), the prefix = m0's header +
    clip, then <sop> .. <eop> on channel CH only, then m0's footer; no target / label / light frame in the prefix."""
    P0, P1 = int(m0["P"]), int(m1["P"])
    if not int(m1.get("persona_block", 0)):
        return P0 == P1 and all(np.array_equal(np.asarray(m0[k]), np.asarray(m1[k])) for k in
                                ("ids", "depth", "present", "pid", "sel", "cb0", "light") + _SHIFT_KEYS) and tuple(m0["vin"]) == tuple(m1["vin"])
    d = P1 - P0
    ok = d > 2 and tuple(m1["vin"]) == (P1, int(m0["vin"][1]))
    ok = ok and all(np.array_equal(m1[k][:, P1:], m0[k][:, P0:]) for k in ("ids", "depth", "present"))
    ok = ok and np.array_equal(m1["sel"] - P1, m0["sel"] - P0) and np.array_equal(m1["light"] - P1, m0["light"] - P0)
    ok = ok and all(np.array_equal(np.asarray(m1[k]), np.asarray(m0[k])) for k in _SHIFT_KEYS)
    ok = ok and np.array_equal(m1["cb0"][P1:], m0["cb0"][P0:]) and bool((m1["cb0"][:P1 + D.CD] < 0).all())
    h = P0 - 2                                                       # m0 = header + [clip] + footer (2 positions: <|im_end|> \n)
    ok = ok and all(np.array_equal(m1[k][:, :h], m0[k][:, :h]) and np.array_equal(m1[k][:, P1 - 2:P1], m0[k][:, P0 - 2:P0]) for k in ("ids", "depth", "present"))
    blk = m1["ids"][CH, h:P1 - 2]
    ok = ok and int(blk[0]) == PT.SOP and int(blk[-1]) == PT.EOP and bool((blk[1:-1] < PT.CTRL_LO).all())
    ok = ok and bool((m1["present"][CH, h:P1 - 2] == 1).all() and (m1["present"][1 - CH, h:P1 - 2] == 0).all() and (m1["depth"][:, h:P1 - 2] < 0).all())
    return bool(ok)


def plan_report(items, cfg):
    """per item (k, r, swap): [k, r, swap, row_key, eligible, drawn, voice_drawn, block tokens, n_frames, P upper bound, overflow, skipped].
    Exact budget check: block <= sp_cap - 5 - clip (clip L + 5 <= clip_max + 5; computed exactly only when the bound does not already fit)."""
    st_ = Stores()
    cap, cmax = int(cfg["sp_cap"]), int(cfg["clip_max"])
    out = []
    for it in items:
        k, r, sw = (int(x) for x in it)
        node, rows, sib = st_.open(k)
        rk = _row_key(rows, r)
        n = int(node["length"][r])
        if TEXT is not None:
            text = TEXT
        else:
            text = prose_of(st_.sidecar(k).get(r, rk), sw)
        el = text is not None
        dr = bool(el and ON and drawn(rk, sw))
        vd = bool(A2.want_voice(rk, sw, cfg["voice_seed"]))
        ntok, skip, over, Pu = 0, 0, 0, 5
        if dr:
            D._tok_init()
            try:
                ntok = len(PT.encode(text)) + 2
            except AssertionError:
                skip, ntok = 1, 0
            clip = cmax + 5 if vd else 0
            if vd and ntok and 5 + clip + ntok > cap:                # exact clip length only when the bound does not fit
                cl = D.voice_clips(rows[r], sib, cfg)[1 ^ sw]
                clip = int(cl["L"]) + 5 if cl is not None else 0
            Pu = 5 + clip + ntok
            over = int(ntok > 0 and Pu > cap)
        else:
            Pu = 5 + (cmax + 5 if vd else 0)
        out.append([k, r, sw, rk, int(el), int(dr), int(vd), ntok, n, Pu, over, skip])
    return out


def summarize_report(rows, n_items):
    a = np.asarray([[x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11]] for x in rows], np.int64).reshape(-1, 8)
    el, dr, vd, nt, n, Pu, ov, sk = (a[:, i] for i in range(8))
    N = max(len(a), 1)
    q = {"pv": int(((dr == 1) & (vd == 1)).sum()), "p_only": int(((dr == 1) & (vd == 0)).sum()), "v_only": int(((dr == 0) & (vd == 1)).sum()),
         "neither": int(((dr == 0) & (vd == 0)).sum())}
    tk = nt[dr == 1]
    Tu = Pu + n + D.CD
    rep = {"n_items": int(len(a)), "plan_items": int(n_items), "eligible": int(el.sum()), "eligible_share": float(el.sum() / N),
           "p_target": P_TARGET, "p_eff": _SETUP.get("p_eff", 0.0), "drawn": int(dr.sum()), "realized_share": float(dr.sum() / N),
           "realized_share_of_eligible": float(dr.sum() / max(el.sum(), 1)), "voice_drawn": int(vd.sum()), "voice_share": float(vd.sum() / N),
           "2x2_counts": q, "2x2_shares": {k: v / N for k, v in q.items()},
           "block_tokens": {"mean": float(tk.mean()) if len(tk) else 0.0, "median": float(np.median(tk)) if len(tk) else 0.0,
                            "p99": float(np.percentile(tk, 99)) if len(tk) else 0.0, "max": int(tk.max()) if len(tk) else 0},
           "T_upper_max": int(Tu.max()) if len(a) else 0, "T_upper_max_no_persona": int(Tu[dr == 0].max()) if (dr == 0).any() else 0,
           "n_T_over_7502": int((Tu > 7502).sum()), "n_T_over_7502_persona": int(((Tu > 7502) & (dr == 1)).sum()), "budget": BUDGET,
           "skipped_special_token": [x[3] + f"|{x[2]}" for x in rows if x[11]],
           "overflow": [{"item": x[:3], "row_key": x[3], "block_tokens": x[7], "P": x[9]} for x in rows if x[10]]}
    s2 = rep["2x2_shares"]
    rep["text"] = (f"train items {rep['n_items']:,} (plan {n_items:,}): eligible (side {SIDE} prose in {list(KEEP)}{', fixed text' if TEXT else ''}) "
                   f"{rep['eligible']:,} = {rep['eligible_share']:.4f}; target {P_TARGET:g} -> p_eff {rep['p_eff']:.6f}; persona drawn {rep['drawn']:,} = "
                   f"{rep['realized_share']:.4f} of all train items ({rep['realized_share_of_eligible']:.4f} of eligible); voice-clip draw "
                   f"{rep['voice_share']:.4f}; 2x2 shares both {s2['pv']:.4f} / persona only {s2['p_only']:.4f} / clip only {s2['v_only']:.4f} / neither "
                   f"{s2['neither']:.4f}; block tokens (<sop> .. <eop>) mean {rep['block_tokens']['mean']:.1f} median {rep['block_tokens']['median']:.0f} "
                   f"p99 {rep['block_tokens']['p99']:.0f} max {rep['block_tokens']['max']}; row length T upper bound max {rep['T_upper_max']} "
                   f"(no-persona items {rep['T_upper_max_no_persona']}; budget {BUDGET}); items with T > 7,502: {rep['n_T_over_7502']} "
                   f"({rep['n_T_over_7502_persona']} with persona); OVERFLOW (block > sp_cap - header/footer/clip) {len(rep['overflow'])}; "
                   f"special-token skips {len(rep['skipped_special_token'])}; eval micros: {'same draw' if EVAL_ON else 'NO persona (= job 29345 eval)'}")
    return rep


def render_dump(items, cfg, _rank, path, n_plain=3, n_voice=2, conv_pos=600):
    """decoded rendered rows (prefix + the first conv_pos positions of the B lane) of the first drawn train items in plan order
    (n_plain without the voice-clip draw, n_voice with it) -> path; returns the text. Item i of the plan = logical rank i % 64."""
    D._tok_init()
    tk = PT._TOK[0]
    st_ = Stores()
    pick, a, b = [], 0, 0
    for i, it in enumerate(items):
        k, r, sw = (int(x) for x in it)
        rows = st_.open(k)[1]
        rk = _row_key(rows, r)
        if not drawn(rk, sw):
            continue
        vd = A2.want_voice(rk, sw, cfg["voice_seed"])
        if (vd and b < n_voice) or (not vd and a < n_plain):
            m = st_.conv(it, cfg, i % 64, audio=False)
            if not m["persona_block"]:
                continue
            pick.append((i, it, m)); a += not vd; b += bool(vd)
        if a >= n_plain and b >= n_voice:
            break
    lines = [f"5 rendered TRAIN rows (aloe-ft-32-samsep with-text-block run; target share {P_TARGET:g}, p_eff {_SETUP.get('p_eff', 0.0):.6f}, side {SIDE}, {len(KEEP)} prose section(s), no labels)",
             "prefix = <|im_start|> system \\n [B voice clip] <sop> TEXT <eop> <|im_end|> \\n ; channel A: nothing in the prefix, Voxtral latents only in the conversation", ""]
    for i, it, m in pick:
        P = int(m["P"])
        pv = json.loads(m["prov"])
        lo = int(np.flatnonzero(m["ids"][CH, :P] == PT.SOP)[0]); hi = int(np.flatnonzero(m["ids"][CH, :P] == PT.EOP)[0])
        btxt = tk.decode([int(t) for t in m["ids"][CH, lo + 1:hi]])
        lane = [int(t) for t in m["ids"][1, P:P + conv_pos]]
        words = tk.decode([t for t in lane if t < PT.CTRL_LO])
        lines += [f"===== plan item {i} {list(it)} row {m['row_key']} swap {it[2]} (B = data {pv['chan_B_data_side']}); P {P}; block tokens "
                  f"{int(m['persona_tokens'])}; voice drawn {m['voice_drawn']} clip {m['voice_clip']}; T {len(m['pid'])}",
                  "--- prefix, decoded by channel (lib decode_prefix):", PT.decode_prefix(m["ids"], m["present"], P, tk),
                  "--- the text block alone (between <sop> and <eop>), exactly as tokenised:", btxt,
                  f"--- B lane, first {conv_pos} conversation positions (words only; PAD / NEW_WORD / control slots omitted):", words, ""]
    txt = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(txt)
    return txt
