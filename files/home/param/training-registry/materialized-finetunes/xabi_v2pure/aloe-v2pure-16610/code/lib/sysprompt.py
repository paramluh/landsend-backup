"""STAGE-3 SYSTEM PROMPT (S3_SYSPROMPT=1): a voice-cloning prefix before the conversation. CPU only (DataLoader workers, tests).

LAYOUT (positions 0..P-1 of the row, one position = one 12.5 Hz frame slot, the same per-frame summed format as the conversation):
  header   <|im_start|> system \\n                      (151644, 8948, 198)        through channel B's table (B = the trunk's text-table copy)
  segments in order; each one of
    speech  <sos_c> lane_c(clip frames) PAD PAD PAD <eos_c>   (L + 5 positions)  ONLY track c: track c's lane (its words on their clip
            frames, NEW_WORD / PAD rule) + track c's codes with the conversation's code delay 3 (clip frame f's codes at position
            b0 + 1 + f + 3: the 3 PAD-lane slots hold the delayed codes of the clip's last 3 frames, so no code crosses <eos_c>), through
            THAT channel's tables; the other channel contributes ZERO (present mask 0: neither its text nor its audio table is read).
            <sosa> 151676 / <eosa> 151677 (track A = user), <sosb> 151678 / <eosb> 151679 (track B = assistant).
    text    (future, user's text prompts) a list of token ids through channel `track`'s table (default B), no codes.
  footer   <|im_end|> \\n                                 (151645, 198)              through channel B's table
  The conversation then starts at position P (its frame 0); its tail is cropped so P + conversation <= 7,500 frames.
  NO LOSS on any position < P (no text target, cb0 label -100, no light frame), asserted in data_s3q.row_arrays and the trainer.
NEW IDS 151676..151679: Qwen3's tokenizer emits ids 0..151668; the 1ch lineage uses 151670 PAD, 151671 NEW_WORD, 151672 BOUNDARY (klpre),
  151673 SOA, 151674 EOA, 151675 SOB and skips 151669 on purpose (data_1ch doc). 151676..151679 are unused anywhere in qwen-1ch / sam-graft
  (grep). Input rows: sos* <- SOB row, eos* <- EOB (<|im_end|>) row, in BOTH channel tables and the voice tower's embed (model_s3q).
  They never appear as a target (the prefix is never scored; asserted).
SWAP (text2_s3q): states / tags refer to MODEL channels; model channel c's clip is drawn from the data track it holds (c ^ swap).
SAMPLING (deterministic per row: rng = SeedSequence([S3_SP_SEED, blake2b(row_key)])): state NONE / A / B / BOTH with probabilities
  S3_SP_P = 0.5, 1/6, 1/6, 1/6 (one uniform draw). Then a clip is ALWAYS drawn for track A and then for track B (so a row's A clip is the
  same in the A-only and BOTH variants; eval runs every variant of a row). Clip of track c:
    range  the row's SECOND HALF: frames [lo, n) with lo = 3750 (S3_SP_HALF). Short rows (n - 3750 < S3_SP_CLIP_MIN): S3_SP_SHORT=half
           (default) uses the second half of the row's actual length, lo = n // 2; 'drop' drops the track (A / B -> NONE, BOTH -> the other).
    length L ~ uniform integer [S3_SP_CLIP_MIN, S3_SP_CLIP_MAX] = [25, 375] frames (2..30 s), capped at n - lo.
    window speech coverage of a window = fraction of its frames inside one of track c's words ([start_frame, max(end_frame, start + 1))).
           Candidates = every start in [lo, n - L] with stride S3_SP_STRIDE (4 frames). best = max coverage; the clip is drawn uniformly
           among the candidates with coverage >= max(S3_SP_MIN_COV, S3_SP_TOP_FRAC x best) (0.5, 0.9). If best < S3_SP_MIN_COV, L is
           halved (not below CLIP_MIN) and the search repeated; still below -> the track is infeasible (dropped as for 'drop').
  The clip's audio also occurs in the conversation (the conversation is the full row): intended.
  10-09 LEAK FIX (S3_SP_SRC; default 'own' = everything above, bit-identical): measured on every model, an own-row clip WITH its word lane is
  copied at ~0 CE (5.7% of the scored words; ~0.13 of the clip's ~0.14 word-CE gain). S3_SP_SRC=sibling: sample() is unchanged (same rng
  draws, same states, same fallbacks, same L0 / u), then resolve_sources() replaces each feasible clip's SOURCE, track by track:
    1. an EARLIER chunk of the same recording (rec_key: the row key without its /cNN[/...] part) holding the same speaker (matched by
       <side>_speaker, any side of the sibling row), nearest first: clip window searched over the WHOLE sibling row (lo 0) with the row's own
       L0 / u draws; lane KEPT (past content). src 'sib_earlier'.
    2. else a LATER chunk (nearest first), same search, lane BLANKED (PAD only, no NEW_WORD: audio codes only; its words are future
       content). src 'sib_later'.
    3. else (single-chunk recording, or no sibling window reaches the coverage threshold) the own-row clip as before, flagged mask=True:
       build_prefix() returns mask [(model channel, w0, L)] and the row builders drop every target on the clip-window frames of that channel
       (apply_clip_mask: text targets of frames [w0, w0 + L) + the trailing word-piece run of a word crossing w0 + L, cb0 labels and light
       frames of audio frames [w0, w0 + L)). src 'own_masked'.
  Clip feasibility, hence the prompt-state mix, is decided by the own-row draw exactly as before (tested equal per row).
  clip_src 'foreign' (eval control only): a clip from a DIFFERENT recording (deterministic per row_key), same data side, lane kept, no mask."""
import hashlib
import re

import numpy as np

STATES = ("NONE", "A", "B", "BOTH")
IM_START, IM_END, NL, SYSTEM = 151644, 151645, 198, 8948
SOSA, EOSA, SOSB, EOSB = 151676, 151677, 151678, 151679
NEW_IDS = (SOSA, EOSA, SOSB, EOSB)
SOS, EOS = (SOSA, SOSB), (EOSA, EOSB)
import speech_q as _SPQ                       # noqa: E402  (PPM: TAIL follows the single code-delay knob PPM_CD)
TAIL = _SPQ.CD                               # = code delay CD: PAD-lane slots after a clip holding its last frames' delayed codes


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def default_cfg(**kw):
    c = {"seed": 20261006, "p": (0.5, 1 / 6, 1 / 6, 1 / 6), "half": 3750, "clip_min": 25, "clip_max": 375, "stride": 4, "min_cov": 0.5,
         "top_frac": 0.9, "short": "half", "new_word": True, "row_frames": 7500, "src": "own"}
    c.update(kw)
    return c


def coverage_mask(words, n):
    m = np.zeros(n + 1, np.int64)
    for w in words:
        a = int(w["start_frame"]); b = max(int(w["end_frame"]), a + 1)
        a, b = max(0, a), min(n, b)
        if b > a:
            m[a] += 1; m[b] -= 1
    return (np.cumsum(m)[:n] > 0).astype(np.int64)


def choose_clip(rng, words, n, cfg):
    """-> dict(w0, L, cov, lo) or None (infeasible). Always consumes the same number of rng draws for the length (1)."""
    L0 = int(rng.integers(int(cfg["clip_min"]), int(cfg["clip_max"]) + 1))
    u = float(rng.random())                                    # the window draw (fixed count: determinism across variants)
    lo = int(cfg["half"])
    if n - lo < int(cfg["clip_min"]):
        if cfg["short"] != "half":
            return None
        lo = n // 2
        if n - lo < int(cfg["clip_min"]):
            return None
    cs = np.concatenate([[0], np.cumsum(coverage_mask(words, n))])
    L = min(L0, n - lo)
    while True:
        st = np.arange(lo, n - L + 1, int(cfg["stride"]))
        cov = (cs[st + L] - cs[st]) / float(L)
        best = float(cov.max())
        if best >= float(cfg["min_cov"]):
            ok = st[cov >= max(float(cfg["min_cov"]), float(cfg["top_frac"]) * best)]
            w0 = int(ok[min(int(u * len(ok)), len(ok) - 1)])
            return {"w0": w0, "L": int(L), "L_drawn": L0, "cov": float((cs[w0 + L] - cs[w0]) / L), "lo": lo, "u": u}
        if L <= int(cfg["clip_min"]):
            return None
        L = max(int(cfg["clip_min"]), L // 2)


def sample(row, cfg):
    """-> dict(state_drawn (MODEL-channel terms: A = model channel A), clips {data track 0 user / 1 assistant: clip|None}, state = the
    unswapped mapping's state after fallback; use model_state(smp, swap) for a swapped row)."""
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), h64(row["row_key"]) & 0x7FFFFFFFFFFF]))
    u = float(rng.random())
    cp = np.cumsum(np.asarray(cfg["p"], np.float64)); cp = cp / cp[-1]
    drawn = STATES[int(np.searchsorted(cp, u, side="right"))]
    n = int(row["n_frames"])
    clips = {c: choose_clip(rng, row[f"words_{side}"], n, cfg) for c, side in enumerate(("user", "assistant"))}
    return {"state": fallback(drawn, clips), "state_drawn": drawn, "clips": clips}


def mclips(smp, swap):
    """clips by MODEL channel: channel c holds data track c ^ swap."""
    return {c: smp["clips"][c ^ int(swap)] for c in (0, 1)}


def model_state(smp, swap=0):
    return fallback(smp["state_drawn"], mclips(smp, swap))


def fallback(state, clips):
    want = {"NONE": (), "A": (0,), "B": (1,), "BOTH": (0, 1)}[state]
    have = tuple(c for c in want if clips[c] is not None)
    return {(): "NONE", (0,): "A", (1,): "B", (0, 1): "BOTH"}[have]


def segments(state, smp, swap=0):
    """the system block's segment list (speech clips in MODEL channel A, B order; <sosa>/<eosa> = model channel A, whichever data track it
    holds). Text segments can be appended: {"kind": "text", "ids": [...], "track": 1}."""
    mc = mclips(smp, swap)
    st = fallback(state, mc)
    want = {"NONE": (), "A": (0,), "B": (1,), "BOTH": (0, 1)}[st]
    return [{"kind": "speech", "track": c, "side": ("user", "assistant")[c ^ int(swap)], **mc[c]} for c in want]


def clip_prov(state, smp, row, swap=0):
    return [{"track": "AB"[s["track"]], "data_role": s["side"], "frame_start": s["w0"], "frame_end": s["w0"] + s["L"], "seconds": s["L"] / 12.5,
             "coverage": round(s["cov"], 4), "row_key": s.get("src_key", row["row_key"]),
             **({"src": s["src"], "src_side": s.get("src_side", s["side"]), "src_chunk": s.get("src_chunk"), "lane": bool(not s.get("blank_lane")),
                 "mask": bool(s.get("mask"))} if "src" in s else {})} for s in segments(state, smp, swap)]


def clip_lane(row, side, w0, L, new_word):
    """track `side`'s lane over clip frames [w0, w0 + L): its words whose start frame is in the window, re-timed to the clip (same
    placement rule as the conversation: data_s3q.lane_words), NEW_WORD on the PAD before each word's first piece."""
    import data_s3q as DS
    words, pid, off = row[f"words_{side}"], np.asarray(row[f"piece_ids_{side}"], np.int64), np.asarray(row[f"piece_offsets_{side}"], np.int64)
    js = [j for j, w in enumerate(words) if w0 <= int(w["start_frame"]) < w0 + L]
    ws = [dict(words[j], start=float(words[j]["start"]) - w0 / 12.5) for j in js]
    pp = [pid[off[j]:off[j + 1]] for j in js]
    po = np.concatenate([[0], np.cumsum([len(x) for x in pp])]).astype(np.int64)
    wif = np.zeros(L, np.int64)
    fr, _ = DS.lane_words(ws, np.concatenate(pp) if pp else np.zeros(0, np.int64), po, L, wif)
    return fr, wif, len(js)


def build_prefix(row, segs, cfg, cd=None, depth_n=32, cbs=2048):
    """segments -> {ids [2, P], depth [2, P, 32], present [2, P] fp32, P, layout} (module doc). P = 0 when segs is empty."""
    import speech_q as SPQ
    cd = SPQ.CD if cd is None else int(cd)
    assert cd == TAIL, ("clip tail slots must equal the code delay", cd, TAIL)
    if not segs:
        return {"P": 0}
    PAD = SPQ.PAD_ID
    cols = []                                                   # (owner channel, id, codes [32] or None)

    def tok(c, t):
        cols.append((c, int(t), None))
    for t in (IM_START, SYSTEM, NL):
        tok(1, t)
    layout, masks = [], []
    for sg in segs:
        if sg["kind"] == "speech":
            c, w0, L = int(sg["track"]), int(sg["w0"]), int(sg["L"])
            side = sg.get("side", ("user", "assistant")[c])
            srow, sside = sg.get("src_row", row), sg.get("src_side", side)        # S3_SP_SRC: the clip's SOURCE row / side (default: this row)
            b0 = len(cols)
            if sg.get("blank_lane"):                                    # later sibling: audio codes only (PAD lane, no NEW_WORD)
                fr, wif, nw_ = np.full(L, PAD, np.int64), np.zeros(L, np.int64), 0
            else:
                fr, wif, nw_ = clip_lane(srow, sside, w0, L, cfg["new_word"])
            lane = np.concatenate([[SOS[c]], fr, [PAD] * TAIL])
            w2 = np.concatenate([[0], wif, [0] * TAIL])
            if cfg["new_word"]:
                SPQ.insert_new_word(lane, w2)
            codes = np.asarray(srow[f"codes_{sside}"])[:, w0:w0 + L].astype(np.int64)
            assert codes.shape == (depth_n, L) and int(codes.min()) >= 0, ("clip codes", codes.shape, L)
            for k, t in enumerate(lane):
                f = k - 1 - cd                                  # clip frame whose codes sit at this slot (code delay)
                cols.append((c, int(t), (codes[:, f] if 0 <= f < L else None)))
            tok(c, EOS[c])
            assert sum(x[2] is not None for x in cols[b0:]) == L, "every clip frame's codes inside its block"
            layout.append({"kind": "speech", "track": c, "pos": (b0, len(cols)), "w0": w0, "L": L, "n_words": nw_})
            if sg.get("mask"):
                masks.append((c, w0, L))
        elif sg["kind"] == "text":
            b0 = len(cols)
            for t in sg["ids"]:
                tok(int(sg.get("track", 1)), t)
            layout.append({"kind": "text", "track": int(sg.get("track", 1)), "pos": (b0, len(cols))})
        else:
            raise ValueError(sg["kind"])
    for t in (IM_END, NL):
        tok(1, t)
    P = len(cols)
    ids = np.full((2, P), PAD, np.int64)
    depth = np.full((2, P, depth_n), -1, np.int64)
    present = np.zeros((2, P), np.float32)
    off = np.arange(depth_n, dtype=np.int64) * cbs
    for p, (c, t, cd_) in enumerate(cols):
        ids[c, p] = t; present[c, p] = 1.0
        if cd_ is not None:
            depth[c, p] = cd_ + off
    assert (present.sum(0) == 1).all(), "exactly one channel owns every prefix position"
    assert ((depth >= 0).any(-1) <= present.astype(bool)).all(), "codes only on the owning channel"
    assert depth[:, -1].max() < 0 and ids[1, -2] == IM_END, "the prefix ends with codes-free <|im_end|> \\n"
    out = {"ids": ids, "depth": depth, "present": present, "P": P, "layout": layout}
    if masks:                                                   # S3_SP_SRC own-row fallback: (model channel, w0, L) to drop from the targets
        out["mask"] = masks
    return out


# ------------------------------------------------------------------------------------------------ 10-09 leak fix (S3_SP_SRC; module doc)
def rec_key(row_key):
    """the recording of a row: its key without the chunk part ('/cNN' and anything after it)."""
    k = re.sub(r"/c\d+(/.*)?$", "", row_key)
    if k == row_key:                                            # 10-09 tt-s3 / s3-rows keys '<D>/<podcast>/<episode>/<NN>'
        k = re.sub(r"/\d+$", "", row_key)
    assert k != row_key, ("row key without a /cNN or /NN chunk part", row_key)
    return k


class SibIndex:
    """per node store: recording -> [(chunk_index, node row, user_speaker, assistant_speaker)], from the words.arrow metadata columns
    (node row ids = s3rows.load_node's order: shards sorted, rows in order). rows = an s3rows.Rows of the same node."""

    def __init__(self, node, rows):
        import pyarrow as pa
        import pyarrow.ipc as ipc
        self.rows, self.by_rec, self.keys, self.recs = rows, {}, [], []
        base = 0
        for s in node["shards"]:
            t = ipc.open_file(pa.memory_map(f"{node['flat_root']}/{s}.words.arrow", "r")).read_all()
            cols = {c: t.column(c).to_pylist() for c in ("row_key", "chunk_index", "user_speaker", "assistant_speaker")}
            for r, k in enumerate(cols["row_key"]):
                rk = rec_key(k)
                self.by_rec.setdefault(rk, []).append((int(cols["chunk_index"][r]), base + r, cols["user_speaker"][r], cols["assistant_speaker"][r]))
                self.keys.append(k); self.recs.append(rk)
            base += len(cols["row_key"])
        assert base == len(node["length"]), ("words.arrow rows != node rows", base, len(node["length"]))

    def siblings(self, row, side):
        """-> (earlier, later): [(chunk, node row, sibling side)] holding row's `side` speaker, nearest chunk first; one entry per chunk."""
        spk, ci = row[f"{side}_speaker"], int(row["chunk_index"])
        best = {}
        for c, nr, us, as_ in sorted(self.by_rec.get(rec_key(row["row_key"]), []), key=lambda x: (x[0], self.keys[x[1]])):
            if c == ci or c in best:
                continue
            ss = "user" if us == spk else ("assistant" if as_ == spk else None)
            if ss is not None:
                best[c] = (c, nr, ss)
        ear = sorted((v for c, v in best.items() if c < ci), key=lambda v: -v[0])
        lat = sorted((v for c, v in best.items() if c > ci), key=lambda v: v[0])
        return ear, lat

    def foreign(self, row):
        """a deterministic node row of a DIFFERENT recording (eval control)."""
        n, rk = len(self.keys), rec_key(row["row_key"])
        j = h64(row["row_key"] + "|foreign") % n
        for _ in range(n):
            if self.recs[j] != rk:
                return j
            j = (j + 1) % n
        return None


def window(words, n, L0, u, cfg, lo=0):
    """choose_clip's window rule with GIVEN draws (L0, u) over frames [lo, n) of a source row -> dict(w0, L, cov) or None."""
    if n - lo < int(cfg["clip_min"]):
        return None
    cs = np.concatenate([[0], np.cumsum(coverage_mask(words, n))])
    L = min(int(L0), n - lo)
    while True:
        st = np.arange(lo, n - L + 1, int(cfg["stride"]))
        cov = (cs[st + L] - cs[st]) / float(L)
        best = float(cov.max())
        if best >= float(cfg["min_cov"]):
            ok = st[cov >= max(float(cfg["min_cov"]), float(cfg["top_frac"]) * best)]
            w0 = int(ok[min(int(u * len(ok)), len(ok) - 1)])
            return {"w0": w0, "L": int(L), "cov": float((cs[w0 + L] - cs[w0]) / L)}
        if L <= int(cfg["clip_min"]):
            return None
        L = max(int(cfg["clip_min"]), L // 2)


class BankIndex:
    """10-11 S3_SP_SRC=bank (gen finetune; = stage4/sysprompt.py, ported for the stage-3 user-text comparison): the EXACT voice-bank reference clips that conditioned each generated call. bank_map = json
    {row_key: [A voice_id, B voice_id]} (the generator manifest's voice_row 'bank:<A>|<B>'); bank_index = the generator's frozen index json
    {voices: [{voice_id, codes (.npy Mimi [32, L] local ids), n_frames}]}. Codes are loaded lazily (cached)."""

    def __init__(self, bank_map, bank_index):
        import json
        self.map = json.load(open(bank_map))
        self.path = {v["voice_id"]: v["codes"] for v in json.load(open(bank_index))["voices"]}
        self.cache = {}

    def voices(self, row_key):
        assert row_key in self.map, ("row has no bank voices (S3_SP_BANK_MAP)", row_key)
        return self.map[row_key]

    def codes(self, vid):
        if vid not in self.cache:
            c = np.load(self.path[vid]).astype(np.int64)
            assert c.ndim == 2 and c.shape[0] == 32 and c.min() >= 0 and c.max() < 2048, (vid, c.shape)
            self.cache[vid] = c
        return self.cache[vid]


def bank_sources(smp, row, bank):
    """S3_SP_SRC=bank: EVERY data track's clip = that call's bank reference clip in the AUDIO-ONLY (blank-lane) form = the generator's
    batch_gen/common.py bank_segments (w0 0, the whole clip, PAD lane, no NEW_WORD; codes at the code delay inside the block). The prompt
    state is the DRAWN state (S3_SP_P), never reduced by the own-row feasibility (a bank clip always exists); no target mask (zero overlap
    with the call)."""
    out = {}
    for c, (side, vid) in enumerate(zip(("user", "assistant"), bank.voices(row["row_key"]))):
        cd = bank.codes(vid)
        out[c] = {"w0": 0, "L": int(cd.shape[1]), "cov": 1.0, "L_drawn": int(cd.shape[1]), "src": "bank", "src_row": {f"codes_{side}": cd},
                  "src_side": side, "src_key": f"bank:{vid}", "src_chunk": None, "blank_lane": True}
    return dict(smp, clips=out, state=fallback(smp["state_drawn"], out))


def resolve_sources(smp, row, cfg, sib, mode):
    """S3_SP_SRC (module doc): smp from sample() / data_s4q.sample_s4 (states untouched) -> smp with each feasible clip's source replaced.
    mode 'sibling' (training + eval) | 'foreign' (eval control). A clip without its draws (u: data_s4q's audio-VAD clip of channel A) keeps
    the own-row source with the mask (mode sibling) / is left as is (foreign)."""
    assert mode in ("sibling", "foreign"), mode
    out = {}
    for c, side in enumerate(("user", "assistant")):
        cl = smp["clips"][c]
        if cl is None:
            out[c] = None
            continue
        res = None
        if "u" in cl:
            if mode == "sibling":
                ear, lat = sib.siblings(row, side)
                for kind, cands in (("sib_earlier", ear), ("sib_later", lat)):
                    for ch_, nr, ss in cands:
                        sr = sib.rows[nr]
                        w = window(sr[f"words_{ss}"], int(sr["n_frames"]), cl["L_drawn"], cl["u"], cfg)
                        if w is not None:
                            res = dict(cl, **w, src=kind, src_row=sr, src_side=ss, src_key=sr["row_key"], src_chunk=int(ch_),
                                       blank_lane=(kind == "sib_later"))
                            break
                    if res is not None:
                        break
            else:
                j = sib.foreign(row)
                if j is not None:
                    sr = sib.rows[j]
                    w = window(sr[f"words_{side}"], int(sr["n_frames"]), cl["L_drawn"], cl["u"], cfg)
                    if w is not None:
                        res = dict(cl, **w, src="foreign", src_row=sr, src_side=side, src_key=sr["row_key"], src_chunk=int(sr["chunk_index"]))
        if res is None:
            res = dict(cl, src="own_masked", mask=True)                  # no usable source: the own-row clip, its window masked
        out[c] = res
    return dict(smp, clips=out)


def mask_frames(ids_c, P, nc, w0, L, ctrl_lo):
    """conversation TEXT-target frames dropped for an own-row clip [w0, w0 + L): the window + the trailing run of word-piece tokens that
    continues across w0 + L (the rest of a word started inside the window). -> bool [nc]."""
    m = np.zeros(nc, bool)
    a, b = max(0, w0), min(nc, w0 + L)
    m[a:b] = True
    g = b
    while 0 < g < nc and ids_c[P + g] < ctrl_lo and m[g - 1]:
        m[g] = True; g += 1
    return m


def apply_clip_mask(d, ids_c, P, nc, cd, w0, L, ctrl_lo):
    """drop channel targets on an own-row clip's frames, in place on the row builder's per-channel dict d (sel / y / word / pad / eob / nw,
    cb0 labels [T], light positions): text targets whose TARGET frame (sel + 1 - P) is masked (mask_frames), cb0 labels of audio frames
    [w0, w0 + L) (position P + cd + f), light frames in [w0, w0 + L). Light keeps >= 1 frame (the nearest labelled one outside the window)."""
    mf = mask_frames(ids_c, P, nc, w0, L, ctrl_lo)
    if len(d["sel"]):
        g = d["sel"] + 1 - P
        keep = ~mf[np.clip(g, 0, nc - 1)] | (g >= nc)
        for k in ("sel", "y", "word", "pad", "eob", "nw"):
            d[k] = d[k][keep]
    lab = d["cb0"]
    a, b = P + cd + max(0, w0), min(len(lab), P + cd + w0 + L)
    if b > a:
        lab[a:b] = -100
    if len(d["light"]):
        f = d["light"] - P - cd
        lt = d["light"][(f < w0) | (f >= w0 + L)]
        if not len(lt):
            ok = np.flatnonzero(lab >= 0)
            lt = ok[np.argmin(np.abs(ok - (P + cd + w0)))][None].astype(np.int64)
        d["light"] = lt
    return mf
