"""PPM DATA (pose + podcast + text-duplex), CPU only (DataLoader workers, CPU tests). Thin wrappers over the frozen stage-3 copies in lib/.

USER DECISIONS (2026-10-11; supersede DESIGN.md where they differ):
  * channel B is the ONLY predicted channel. Channel A is conditioning only: its lane + Mimi codes are summed into the backbone input,
    but there is NO loss on A (no A text targets, no A cb0, no A light). Pose rows already carry only B.
  * RANDOM SIDES: per podcast row and per text-duplex conversation, which DATA speaker becomes model channel B is drawn from a
    deterministic hash (side_of(row_key, seed), 50/50). swap = 1 <-> data speaker 'user' sits on channel B. Voice clips and personas
    follow the channels (sysprompt / persona_text_s3q swap rules).
  * system prompt <= SP_CAP (2,000) positions in total (header + clips + persona text + footer); personas of BOTH speakers allowed
    (PPM_PERSONA_SIDES both | B | A | none), the persona text is cropped to fit the budget; voice clips count toward it.

COMMON MICRO (one dict per micro; numpy; the trainer moves it to the GPU):
  kind      'pose' | 'speech' | 'text'
  ids       [2, N] lanes of model channels A, B (text ids)          present [2, N] fp32 (0 = the channel contributes nothing there)
  depth     [2, N, 32] Mimi codes at the code delay (-1 none)       (text: absent -> no audio input)
  pdepth    [N, 32] pose codes at POSE_DELAY on channel B            (pose rows only)
  pid [N] plain RoPE positions (frame index, restarting per segment); segs [..] packed attention segments (sum = N)
  sel [S] scored positions (channel B only) -> y [S] = ids[1, sel + 1]; word / pad / eob / nw [S] class masks; n_text = S (the per-row
            mean's normaliser: R = 1 row per speech / pose micro; a text pack = one mean over the pack's B positions, txtdup's rule)
  cb0 [N] speech cb0 label of channel B at each position (-100 none) ; light [Ls] positions (into N) of B's selected 0.6B frames
  pcb0 [N] pose cb0 label (-100 none) ; plight [Lp] positions of the pose 0.6B frames                (pose rows only)
  prov      json string (provenance)

PODCAST SYSTEM PROMPT (D4 / D11, user spec): per row (deterministic hash):
  voice   on 1/3 of rows (h64('ppm-voice|seed|row_key') % 3 == 0): a clip of data speaker 'user' AND of 'assistant' (3-10 s = 38..125
          frames), each from a LATER chunk of the same recording that lies in the recording's SECOND HALF (chunk_index > this row's and
          chunk_index >= n_chunks / 2; nearest such chunk first; same speaker matched by speaker id; window search = sysprompt.window over
          the whole source chunk), lane BLANKED (audio codes only: sysprompt's sib_later form). A speaker with no such chunk / window gets
          no clip (the other may still have one).
  persona every available persona-sidecar section (persona, starting_context, memory; persona_text_s3q 'all' mode) of the allowed sides,
          one <sop> .. <eop> block per speaker on its model channel.
  order   <|im_start|> system \\n  [clip A] [clip B] [persona A] [persona B]  <|im_end|> \\n   (model-channel order).
  empty   rows with neither: the EMPTY system block <|im_start|> system \\n <|im_end|> \\n (5 positions, channel B).
  cap     total P <= SP_CAP: persona token budget = SP_CAP - (header + footer + clips); blocks cropped (content truncated, <eop> kept),
          a shorter block keeps all its tokens and the leftover goes to the other.
  The conversation is NOT cropped by the prefix (default PPM_CROP_CONV=0): the row is P + n + CD positions long."""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import speech_q as SP  # noqa: E402
import s3rows as R3  # noqa: E402
import data_s3q as DS  # noqa: E402
import pose_s3q as PS  # noqa: E402
import txtdup_s3q as TXD  # noqa: E402
import sysprompt as SPR  # noqa: E402
import persona_s3q as PER  # noqa: E402
import persona_text_s3q as PT  # noqa: E402

PAD_ID, NW_ID, EOB_ID, CTRL_LO, CD, DEPTH, CBS = SP.PAD_ID, SP.NW_ID, SP.EOB_ID, SP.CTRL_LO, SP.CD, SP.DEPTH, SP.CBS
TOKENIZER = "/data/home/amartyavarma/models/Qwen3-32B"
MAX_ID = PT.EOP                     # 151681: the largest input id any PPM row carries (persona <eop>)


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def side_of(key, seed):
    """swap bit of a row / conversation: 1 = data speaker 'user' (speaker 0 / side a) is model channel B."""
    return int(h64(f"ppm-side|{int(seed)}|{key}") & 1)


# 10-21 FRESH-DATA RESUME FLIPS (user request for the 8,1,1 resume from dcp-17012; train_ppm.py CFG['fresh']): a training item whose stream
# entry carries FLIP_BIT is built with its sides FLIPPED (swap = 1 - the row's usual swap): speech (PPM_SPEECH_FLIP: epoch >= 2 rows) = the
# other speaker on the predicted channel B, through everything role-dependent (lanes, Mimi codes, voice clips, persona sidecars, wpid, cb0 /
# light); Gemma text duplex (PPM_TEXT_FLIP) = swap = 1 - side_of('txtdup/<conv_id>'). Chat packs carry the flag as a 3rd tuple element
# (chat_ppm.ChatDS). Eval items never carry it. Unset knobs = no bit anywhere = the previous behaviour exactly.
FLIP_BIT = 1 << 62


def flip_split(c):
    """stream entry -> (entry without FLIP_BIT, flip flag)."""
    c = int(c)
    return c & ~FLIP_BIT, bool(c & FLIP_BIT)


def _tok_init():
    if PT._TOK[0] is None:
        from transformers import AutoTokenizer
        PT._TOK[0] = AutoTokenizer.from_pretrained(TOKENIZER)
    return PT._TOK[0]


def default_cfg(**kw):
    c = {"seed": 20261011, "voice_mod": 3, "clip_min": 38, "clip_max": 125, "stride": 4, "min_cov": 0.5, "top_frac": 0.9,
         "sp_cap": 2000, "persona_sides": "both", "new_word": True, "wpid": "both", "eob_gap_s": 0.0, "light_seed": 20261001,
         "crop_conv": True, "light_frac": 1.0, "pose_light_frac": 1.0, "pose_delay": SP.CD, "txt_cap": 7500,
         "pack_attn": "causal", "chat_side": "assistant", "chat_eob": True}
    c.update(kw)
    return c


# ------------------------------------------------------------------------------------------------ podcast system prompt
def rec_chunks(sib, row):
    """all chunk indices of this row's recording on the node (episodes are node-local: verified on the 19-node manifest)."""
    return sorted({c for c, _, _, _ in sib.by_rec.get(SPR.rec_key(row["row_key"]), [])})


def voice_clips(row, sib, cfg):
    """-> {data track d (0 user, 1 assistant): clip dict (sysprompt segment fields, src_row / blank_lane) | None}, draw info."""
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), h64(row["row_key"]) & 0x7FFFFFFFFFFF, 0x5643]))
    draws = [(int(rng.integers(int(cfg["clip_min"]), int(cfg["clip_max"]) + 1)), float(rng.random())) for _ in range(2)]
    chunks = rec_chunks(sib, row)
    nch = (max(chunks) + 1) if chunks else 0
    ci = int(row["chunk_index"])
    out = {}
    wcfg = {"clip_min": cfg["clip_min"], "stride": cfg["stride"], "min_cov": cfg["min_cov"], "top_frac": cfg["top_frac"]}
    for d, side in enumerate(("user", "assistant")):
        out[d] = None
        _, lat = sib.siblings(row, side)
        L0, u = draws[d]
        for ch_, nr, ss in lat:                                      # nearest later chunk first
            if ch_ <= ci or ch_ < nch / 2.0:
                continue
            sr = sib.rows[nr]
            w = SPR.window(sr[f"words_{ss}"], int(sr["n_frames"]), L0, u, wcfg, lo=0)
            if w is not None:
                out[d] = dict(w, src="later_2nd_half", src_row=sr, src_side=ss, src_key=sr["row_key"], src_chunk=int(ch_), blank_lane=True,
                              side=side, n_chunks=nch)
                break
    return out


def crop_blocks(blocks, budget):
    """blocks: list of id lists ([SOP] + text + [EOP]) -> (list aligned with blocks: cropped block or None (dropped), tokens cut);
    total kept length <= budget; a kept block keeps SOP and EOP."""
    if sum(len(b) for b in blocks) <= budget:
        return list(blocks), 0
    n = len(blocks)
    alloc = [0] * n
    left = budget
    for k, i in enumerate(sorted(range(n), key=lambda i: len(blocks[i]))):   # water-filling: short blocks keep all, the rest share
        alloc[i] = min(len(blocks[i]), left // (n - k))
        left -= alloc[i]
    out, cut = [], 0
    for b, a in zip(blocks, alloc):
        nb = None if a < 3 else b[:a - 1] + [b[-1]]               # < 3: not even one content token -> drop the block
        cut += len(b) - (len(nb) if nb else 0)
        out.append(nb)
    return out, cut


def podcast_prefix(row, swap, sib, pers, cfg):
    """-> (sysprompt.build_prefix dict (P >= 5), provenance)."""
    sw = int(swap)
    want_voice = h64(f"ppm-voice|{int(cfg['seed'])}|{row['row_key']}") % int(cfg["voice_mod"]) == 0
    segs, pv = [], {"voice_drawn": int(want_voice)}
    clips = voice_clips(row, sib, cfg) if want_voice else {0: None, 1: None}
    for c in (0, 1):                                                 # model channel A then B; channel c holds data track c ^ swap
        cl = clips[c ^ sw]
        if cl is not None:
            segs.append({"kind": "speech", "track": c, **cl})
    pv["clips"] = [{"track": "AB"[s["track"]], "data_side": s["side"], "src_key": s["src_key"], "src_chunk": s["src_chunk"],
                    "frames": [s["w0"], s["w0"] + s["L"]], "cov": round(s["cov"], 3)} for s in segs]
    fixed = 5 + sum(int(s["L"]) + 5 for s in segs)
    assert fixed <= int(cfg["sp_cap"]), ("voice clips alone exceed the system-prompt cap", fixed)
    allow = {"both": (0, 1), "B": (1,), "A": (0,), "none": ()}[cfg["persona_sides"]]
    blocks, chans = [], []
    if pers is not None and allow:
        _tok_init()
        for c in (0, 1):
            if c not in allow:
                continue
            av = PT.avail(pers.get(PER.SIDES[c ^ sw]))
            keys = [k for k in PT.SECTIONS if k in av]
            if keys:
                try:                                                 # 10-15 watchdog guard: a sidecar whose prose holds a literal special token
                    ids = PT.encode(PT.block_text(av, keys))         # (e.g. <|endoftext|>) -> that side's persona block is skipped, row kept
                except AssertionError as e:
                    print(f"[data] persona block skipped row {row['row_key']} side {PER.SIDES[c ^ sw]}: {e}", file=sys.stderr, flush=True)
                    pv["persona_skipped"] = pv.get("persona_skipped", 0) + 1
                    continue
                blocks.append([PT.SOP] + ids + [PT.EOP]); chans.append(c)
    blocks2, cut = crop_blocks(blocks, int(cfg["sp_cap"]) - fixed)
    segs_t = [{"kind": "text", "track": c, "ids": b2} for c, b2 in zip(chans, blocks2) if b2 is not None]
    segs += segs_t
    pv.update(persona=[{"track": "AB"[s["track"]], "tokens": len(s["ids"])} for s in segs_t], persona_cut=int(cut))
    if not segs:                                                     # EMPTY system block: header + footer only
        segs = [{"kind": "text", "track": 1, "ids": []}]
    spx = SPR.build_prefix(row, segs, {"new_word": cfg["new_word"]})
    assert 5 <= spx["P"] <= int(cfg["sp_cap"]), ("system prompt length", spx["P"])
    pv["P"] = int(spx["P"])
    return spx, pv


# ------------------------------------------------------------------------------------------------ micros
def _b_only(a, c=1):
    d = a[c]
    return d["sel"], d["y"], d["word"], d["pad"], d["eob"], d["nw"], d["cb0"], d["light"]


def speech_micro(row, node_row, sib, pers, cfg, rank, swap=None, flip=False):
    """one podcast row -> the common micro (channel B active). swap None = the random side (side_of); 0 / 1 forces it (fixed-orientation evals).
    flip (FLIP_BIT, training rows only) = the random side FLIPPED: swap = 1 - side_of (the row's other speaker on B)."""
    assert not (flip and swap is not None), "flip applies to the random side only"
    sw = side_of(row["row_key"], cfg["seed"]) if swap is None else int(swap)
    if flip:
        sw = 1 - sw
    spx, pv = podcast_prefix(row, sw, sib, pers, cfg)
    P, n = int(spx["P"]), int(row["n_frames"])
    rf = P + n if not cfg["crop_conv"] else 7500
    dc = {"new_word": cfg["new_word"], "eob_gap_s": cfg["eob_gap_s"], "wpid": cfg["wpid"], "light_seed": cfg["light_seed"], "row_frames": rf,
          "frame_light": True, "no_crop": not cfg["crop_conv"]}
    a = DS.row_arrays(row, dc, rank, int(node_row), "ppm", float(cfg["light_frac"]), sp=spx, swap=sw)
    sel, y, word, pad, eob, nw, cb0, light = _b_only(a)
    T = len(a["pid"])
    out = {"kind": "speech", "ids": a["ids"], "depth": a["depth"], "present": a["present"], "pid": a["pid"], "segs": [T],
           "sel": sel, "y": y, "word": word, "pad": pad, "eob": eob, "nw": nw, "n_text": float(len(sel)), "cb0": cb0, "light": light,
           "P": P, "swap": sw}
    assert (cb0[:P + CD] < 0).all() and (light >= P + CD).all() and (sel >= P).all()
    assert T == (7500 if cfg["crop_conv"] else P + n) + CD, ("row length", T, P, n)
    pv.update(row_key=row["row_key"], swap=sw, chan_B_data_side=("user" if sw else "assistant"), n_frames=n, T=T)
    if flip:
        pv["flip"] = 1
    out["prov"] = json.dumps(pv)
    return out


def text_micro(recs, cfg, sws=None, flip=False):
    """one text-duplex pack (K conversations, own segments, random side per conversation) -> the common micro (B text CE only).
    sws None = the random side per conversation; a list forces it (fixed-orientation evals). flip (FLIP_BIT, training packs only) = every
    conversation's random side FLIPPED (swap = 1 - side_of)."""
    assert not (flip and sws is not None), "flip applies to the random sides only"
    sws = [side_of(f"txtdup/{r['conv_id']}", cfg["seed"]) ^ int(flip) for r in recs] if sws is None else [int(x) for x in sws]
    m = TXD.micro(recs, sws, {"cap": cfg["txt_cap"], "new_word": cfg["new_word"], "wpid": cfg["wpid"]})
    k = m["ch"] == 1
    N = int(m["T"])
    out = {"kind": "text", "src": "gemma", "ids": m["ids"], "present": np.ones((2, N), np.float32), "pid": m["pid"], "segs": [int(x) for x in m["segs"]],
           "sel": m["sel"][k], "y": m["y"][k], "word": m["word"][k], "pad": m["pad"][k], "eob": m["eob"][k], "nw": m["nw"][k]}
    out["n_text"] = float(k.sum())
    out["prov"] = json.dumps([{"conv_id": r["conv_id"], "swap": s, "n_frames": int(r["n_frames"]), **({"flip": 1} if flip else {})} for r, s in zip(recs, sws)])
    return out


G0_DIM = 8                          # G0COND (pose5 10-27, user change #2): per-clip frame-0 features (g0_feat), pose rows only
G0_FIELDS = ("frame0_root", "frame0_width", "orig_h", "orig_w", "frame0_ok")
G0_STATS = {"clips": 0, "invalid": 0, "missing_field": 0}            # this process's counts (pose_micro_ppm under g0cond; trainer logs them)


def g0_problem(clip):
    """'' if g0_feat can use the clip, else why not ('missing field <f>' / 'frame0_ok false' / 'bad value ...'); audit NIT: never silent."""
    miss = [k for k in G0_FIELDS if k not in clip or clip[k] is None]
    if miss:
        return "missing field " + ",".join(miss)
    return "" if g0_feat(clip)[1] else ("frame0_ok false" if not clip["frame0_ok"] else
                                        f"bad value root {clip['frame0_root']} w {clip['frame0_width']} H {clip['orig_h']} W {clip['orig_w']}")


def g0_feat(clip):
    """G0COND: one pose clip's frame-0 conditioning features (computed in float64, stored float32) -> (f [G0_DIM], ok).
    r = frame0_root (y, x; image-normalised: y / orig_h, x / orig_w), w = frame0_width (horizontal shoulder width, x / orig_w), H, W =
    orig_h, orig_w (the source video's pixel size, parquet pose struct; every clip record of the fresh plans and of PPM's samdata stores):
      f = [r_y, r_x, log w, -r_y / w, (1 - r_y) / w, -r_x / w, (1 - r_x) / w, log(H / W)]
    (f[3:7] = the distances from the root to the top / bottom / left / right frame edges in shoulder widths = pose_s3q.global_features'
    edge term e; f[7] = the aspect ratio, log so portrait / landscape are symmetric). Invalid frame 0 (frame0_ok false, a non-finite value,
    w <= 0, H <= 0 or W <= 0) -> f = 0 (ok False): with g_proj bias-free the clip's SLOT input is then exactly today's."""
    try:
        ry, rx = (float(x) for x in clip["frame0_root"])
        w, H, W = float(clip["frame0_width"]), float(clip["orig_h"]), float(clip["orig_w"])
        ok = bool(clip.get("frame0_ok", True)) and all(np.isfinite([ry, rx, w, H, W])) and w > 0 and H > 0 and W > 0
    except (KeyError, TypeError, ValueError):
        ok = False
    if not ok:
        return np.zeros(G0_DIM, np.float32), False
    f = np.array([ry, rx, np.log(w), -ry / w, (1.0 - ry) / w, -rx / w, (1.0 - rx) / w, np.log(H / W)], np.float64)
    return f.astype(np.float32), True


def pose_micro_ppm(prow, cfg, rank):
    """one pose row (channel B speaker) -> the common micro. G0COND (cfg 'g0cond' truthy; the trainer passes it to the pose datasets only
    when PPM_G0COND=1, never in CFG['data']): + g0pos [K] int64 = each clip's SLOT position (pose_s3q's gpos: its segment's first position,
    unchanged by packed()) and g0f [K, G0_DIM] float32 = g0_feat of that clip; absent = the micro exactly as before."""
    m = PS.pose_micro([prow], {"new_word": cfg["new_word"], "pose_delay": cfg["pose_delay"], "light_seed": cfg["light_seed"]}, rank,
                      float(cfg["light_frac"]), float(cfg["pose_light_frac"]))
    assert (m["ch"] == 1).all()
    out = {"kind": "pose", "ids": m["ids"], "depth": m["depth"], "present": m["present"], "pdepth": m["pdepth"], "pid": m["pid"],
           "segs": [int(x) for x in m["segs"]], "sel": m["sel"], "y": m["y"], "word": m["word"], "pad": m["pad"], "eob": m["eob"],
           "nw": m["nw"], "n_text": float(len(m["sel"])), "cb0": m["cb0"], "light": m["light"], "pcb0": m["pcb0"], "plight": m["plight"],
           "prov": m["prov"]}
    if cfg.get("g0cond"):
        fs = [g0_feat(c) for c in prow["clips"]]
        for k_, (c, (_, ok_)) in enumerate(zip(prow["clips"], fs)):  # LOUD on any unusable clip (expected never: 0 in plans 1-3 / npl5 / samdata)
            G0_STATS["clips"] += 1
            if not ok_:
                why = g0_problem(c)
                G0_STATS["invalid"] += 1; G0_STATS["missing_field"] += int(why.startswith("missing"))
                print(f"[g0cond] WARNING clip {k_} of row {prow.get('row_key')}: {why or 'invalid'} -> f = 0 (SLOT input as today); "
                      f"process counts {G0_STATS}", file=sys.stderr, flush=True)
        assert len(fs) == len(m["gpos"]) and (np.asarray(prow["kind"])[m["gpos"]] == PS.KIND_SLOT).all(), "G0COND: clip SLOT positions"
        out["g0pos"] = np.asarray(m["gpos"], np.int64)
        out["g0f"] = np.stack([f for f, _ in fs]).astype(np.float32)
    return out


def flatten_segments(m):
    """user decision 10-11: packed rows (pose clips, Gemma packs, chat packs) attend PLAIN CAUSAL over the whole row: one segment, positions
    0..N-1 continuous (loss masking unchanged). The towers' one-frame shifts then also run across clip / conversation boundaries."""
    N = int(sum(m["segs"]))
    m["segs"], m["pid"] = [N], np.arange(N, dtype=np.int64)
    return m


def packed(m, cfg):
    return flatten_segments(m) if cfg.get("pack_attn", "causal") == "causal" else m


# ------------------------------------------------------------------------------------------------ plans + datasets
def speech_plan(node, idx, seed, lw, local, eval_frac, eval_salt, n_eval):
    """stage-3 plan (episode-hash eval, seeded stripe) restricted to FULL 7,500-frame rows (train and eval)."""
    p = None
    for mult in (4, 3, 2, 1):                                     # ask for spare eval rows (some are not FULL); fewer if the node has few
        try:
            p = DS.plan(node, idx, seed, lw, local, 1, mult * max(n_eval, 1), eval_frac, eval_salt, row_exclude=None)
            break
        except AssertionError:
            continue
    assert p is not None, ("not enough eval rows on this node even for 1x", node["node"], n_eval, lw)
    full = node["length"] == 7500
    p["rows"] = p["rows"][full[p["rows"]]]
    p["eval_rows"] = [i for i in p["eval_rows"] if full[i]][:n_eval]
    assert len(p["eval_rows"]) == n_eval, ("not enough FULL eval rows for this rank", len(p["eval_rows"]), n_eval)
    return p


class SpeechDS:
    def __init__(self, root, rows, cfg, rank):
        self.root, self.rows, self.cfg, self.rank = root, [int(r) for r in rows], dict(cfg), int(rank)
        self.node = self.st = self.sib = self.ps = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.st is None:
            self.node = R3.load_node(self.root)
            self.st = R3.Rows(self.node)
            self.sib = SPR.SibIndex(self.node, self.st)
            self.ps = PER.PersonaSidecar(self.node)
        r, fl = flip_split(self.rows[i])                             # FLIP_BIT (fresh resume): sides flipped
        row = self.st[r]
        R3.check_row(row)
        try:
            pers = self.ps.get(r, row["row_key"])
        except (FileNotFoundError, OSError):
            pers = None
        return speech_micro(row, r, self.sib, pers, self.cfg, self.rank, flip=fl)


class PoseDS:
    def __init__(self, node, rows, cfg, rank):
        self.node, self.rows, self.cfg, self.rank, self.st = node, [int(r) for r in rows], dict(cfg), int(rank), None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.st is None:
            self.st = PS.PoseRows(self.node)
        return packed(pose_micro_ppm(self.st[self.rows[i]], self.cfg, self.rank), self.cfg)


class TextDS:
    def __init__(self, root, split, packs, cfg):
        self.root, self.split, self.packs, self.cfg, self.st = root, split, [int(p) for p in packs], dict(cfg), None

    def __len__(self):
        return len(self.packs)

    def __getitem__(self, i):
        if self.st is None:
            self.st = TXD.Store(self.root)
        p, fl = flip_split(self.packs[i])                            # FLIP_BIT (fresh resume, train packs only): sides flipped
        assert not (fl and self.split != "train"), "eval packs are never flipped"
        ids = self.st.pack(self.split, p)
        return packed(text_micro([self.st.get(j) for j in ids], self.cfg, flip=fl), self.cfg)


# ------------------------------------------------------------------------------------------------ D2: extra s3-rows slices (all FULL rows)
# The 20 s3-rows slices hold 4.45M FULL rows; the 16 training nodes carry 16 of them. D2 (user decision 10-12): the run sees ALL of them.
# plan_extra.py splits the 4 other slices (slinky-4, -7, -15, -18) over the training hosts (/scratch/amartyavarma/s3-rows-extra/<slice>:
# every slice's NODE.json / rows.parquet / *.DONE / *.index.npz + the codes / words / persona files of the shards the host needs; a
# recording never spans shards, so a host holding a row's shard holds its whole recording = every voice-clip sibling).
# A speech row is addressed by a CODE = slot << 40 | node row: slot 0 = the host's own slice (PPM_SPEECH_ROOT), slot k >= 1 = the host's
# extra slot k - 1. An extra row is built by the SAME speech_micro with the SAME node row id as on its own node (persona sidecar, voice clips
# from the second half of the same recording, CD, random B side, 7,500 crop, light seed): bit-identical (tests/extra_digest.py).
SLOT_SHIFT = 40


def code_of(slot, row):
    return (int(slot) << SLOT_SHIFT) | int(row)


def split_code(c):
    return int(c) >> SLOT_SHIFT, int(c) & ((1 << SLOT_SHIFT) - 1)


def load_extra_plan(path):
    P = json.load(open(path))
    assert P.get("format") == "ppm-extra-plan-v1", P.get("format")
    return P


class PartSibIndex(SPR.SibIndex):
    """SibIndex of a PARTIAL copy of a slice: words.arrow read only for the shards present, node row ids = the full slice's numbering
    (base offsets from NODE.json rows), so siblings / rec_chunks of any row in a present shard equal the native node's answers (a
    recording lies inside one shard: plan_extra asserts the host holds every shard of each of its rows' recordings)."""

    def __init__(self, node, rows, present):
        import pyarrow as pa
        import pyarrow.ipc as ipc
        self.rows, self.by_rec, self.keys, self.recs = rows, {}, {}, {}
        present, base = set(present), 0
        for s, n in zip(node["shards"], node["meta"]["rows"]):
            if s in present:
                t = ipc.open_file(pa.memory_map(f"{node['flat_root']}/{s}.words.arrow", "r")).read_all()
                cols = {c: t.column(c).to_pylist() for c in ("row_key", "chunk_index", "user_speaker", "assistant_speaker")}
                assert len(cols["row_key"]) == n, (s, len(cols["row_key"]), n)
                for r, k in enumerate(cols["row_key"]):
                    rk = rec_key_(k)
                    self.by_rec.setdefault(rk, []).append((int(cols["chunk_index"][r]), base + r, cols["user_speaker"][r], cols["assistant_speaker"][r]))
                    self.keys[base + r] = k; self.recs[base + r] = rk
            base += int(n)
        assert base == len(node["length"]), ("NODE.json rows != node rows", base, len(node["length"]))

    def foreign(self, row):
        raise NotImplementedError("foreign rows are an eval control of the native store only")


def rec_key_(k):
    return SPR.rec_key(k)


class MixedSpeechDS:
    """the rank's speech rows as CODES (above). host_plan = load_extra_plan(..)['hosts'][host] (None: native only)."""

    def __init__(self, root, host_plan, rows, cfg, rank):
        self.root, self.hp, self.rows, self.cfg, self.rank = root, host_plan, [int(r) for r in rows], dict(cfg), int(rank)
        self.src = {}

    def __len__(self):
        return len(self.rows)

    def _open(self, slot):
        if slot not in self.src:
            if slot == 0:
                node = R3.load_node(self.root); st = R3.Rows(node); sib = SPR.SibIndex(node, st)
            else:
                sl = self.hp["slots"][slot - 1]
                node = R3.load_node(sl["root"])
                assert node["node"] == sl["node"], ("extra slot holds another slice", sl["root"], node["node"], sl["node"])
                st = R3.Rows(node); sib = PartSibIndex(node, st, sl["shards_present"])
            self.src[slot] = (node, st, sib, PER.PersonaSidecar(node))
        return self.src[slot]

    def get(self, code, swap=None):
        code, fl = flip_split(code)                                  # FLIP_BIT (fresh resume, training rows only): sides flipped
        slot, r = split_code(code)
        node, st, sib, ps = self._open(slot)
        row = st[r]
        R3.check_row(row)
        try:
            pers = ps.get(r, row["row_key"])
        except (FileNotFoundError, OSError):
            pers = None
        return speech_micro(row, r, sib, pers, self.cfg, self.rank, swap=swap, flip=fl)

    def __getitem__(self, i):
        return self.get(self.rows[i])


def extra_episode_keys(sl):
    """-> (node rows, episode keys) of an extra slot's slice (rows.parquet on the host)."""
    import pyarrow.parquet as pq
    t = pq.read_table(f"{sl['root']}/rows.parquet", columns=["row", "dataset", "podcast_id", "episode_id"]).to_pydict()
    assert t["row"] == list(range(len(t["row"])))
    return [DS.episode_key(a, b, c) for a, b, c in zip(t["dataset"], t["podcast_id"], t["episode_id"])]


def d2_stream(nat, sidx, hplan, local, rank, s0, nsm, salts, eval_frac, seed):
    """THIS rank's D2 speech stream (codes): rows [0, s0 x nsm) = the native stream as consumed so far (unchanged, in order); then the
    rank's remaining native rows + its planned extra rows, minus every row of an eval episode under `salts` (holdout from s0 on), shuffled
    (seed, rank, 0xD2, s0). -> (codes int64, stats)."""
    nat = np.asarray(nat, np.int64)
    assert len(nat) == hplan["native"][local], ("native rows != the plan's native count", len(nat), hplan["native"][local])
    head, tail = nat[:s0 * nsm], nat[s0 * nsm:]
    ext = np.asarray([code_of(k + 1, r) for k, r in hplan["ranks"][local]], np.int64)
    xs = {"head": int(len(head)), "native": int(len(nat)), "extra": int(len(ext)), "seen_holdout": 0, "drop_native": 0, "drop_extra": 0}
    if salts:
        def hold(keys):
            return np.array([any(DS.is_eval_episode(k, eval_frac, sl) for sl in salts) for k in keys], bool)
        hn = hold([DS.episode_key(a, b, c) for a, b, c in zip(sidx["dataset"], sidx["podcast_id"], sidx["episode_id"])])
        xs["seen_holdout"] = int(hn[head].sum())
        keep = ~hn[tail]; xs["drop_native"] = int((~keep).sum()); tail = tail[keep]
        he = {k + 1: hold(extra_episode_keys(sl)) for k, sl in enumerate(hplan["slots"])}
        keep = np.array([not he[split_code(c)[0]][split_code(c)[1]] for c in ext], bool) if len(ext) else np.zeros(0, bool)
        xs["drop_extra"] = int((~keep).sum()); ext = ext[keep]
    xs["extra_kept"] = int(len(ext))
    mix = np.concatenate([tail, ext])
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(rank), 0xD2, int(s0)]))
    return np.concatenate([head, mix[rng.permutation(len(mix))]]), xs
