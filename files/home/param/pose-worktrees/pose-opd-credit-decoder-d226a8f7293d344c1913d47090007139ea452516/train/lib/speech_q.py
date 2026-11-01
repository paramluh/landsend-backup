"""QWEN stage-2 SPEECH data: tts-pretrain-128m utterances packed into SINGLE-CHANNEL rows of exactly ROW_FRAMES (default 7,500) Mimi frames
(12.5 Hz, 32 codebooks x 2048). COPY of mistral-distill-bf16/stage2_1ch/speech_1ch.py (store, plan, pack rule, layout, targets unchanged)
with the Qwen 1ch ids and two additions: the mix-RoPE WORD positions (wpid) and the MERGED micro (N rows packed varlen in one micro).
CPU only, no torch.distributed.

IDS (qwen-1ch data_1ch / render_q): PAD 151670, SOB 151675, EOB = <|im_end|> 151645 (stage 1's EOB, the assistant turn's last lane-B
  token), SOA / EOA 151673 / 151674 never on a speech row (asserted). A WORD piece = any Qwen tokenizer id < 151643 (<|endoftext|>, the
  first control id): Qwen ids are dense from 0, so Mistral's ">= 1000" rule is replaced by "< CTRL_LO"; words are tokenised with
  split_special_tokens=True (a literal "<|im_end|>" in a transcript is plain text) and every piece id is asserted < CTRL_LO.
PER-UTTERANCE LAYOUT (piece of n audio frames at row frame s; it costs n + 2 frames) = speech_1ch's:
  frame s SOB | frames s+1..s+n audio (word pieces on their true frames, Qwen lane rule; PAD elsewhere) | frame s+n+1 EOB | next SOB ...
  Frames >= U are the PAD tail (no loss). depth: codes of audio frame f at position f + CD (CD = 3).
POSITIONS (the trunk's mix-RoPE, stage 1's rule mirrored; items_q.row_word_positions):
  pid  [T]  frame position 0..T-1 (plain RoPE's), per row.
  wpid [T]  WORD position = #lane-B tokens in frames 0..p INCLUSIVE, lane-B token = a word piece or EOB (stage 1: EOB <|im_end|> is a
            lane-B token and advances; SOB / PAD / tail repeat). No system prefix on a speech row (P = 0): frame 0 (SOB) sits at 0, the
            first word piece at 1 (stage 1: prepended frame at P, first assistant token at P + 1).
TARGETS (the trunk at p predicts ids[p + 1], p < U - 1): word (< CTRL_LO) / pad / eob / sob (after an EOB: input-only by default). The
  trainer forms the weights on the GPU from the class masks (word x W_TEXT_CE + ramp(step) x (pad x W_TIM_CE + eob x W_EOB_CE)) / n_scored
  of the ROW: the TIMING ramp (timing_ramp_steps) lives in the trainer, the data is step-independent.
MERGED MICRO (klpre speed.merge_speech, user 10-02: 2 rows of 7,500 per GPU in one micro): merge_rows() concatenates N rows' streams;
  segs = [T] * N (FA3 varlen: block-diagonal causal), pid / wpid restart per row, sel / light offsets shifted, rid = the row of every
  scored position / light frame. Every loss term is each ROW's own mean, summed over the rows (== N one-row micros; klpre merge_unequal)."""
import hashlib
import json
import math
import os

import numpy as np

PAD_ID, EOS_ID, SOB_ID, CTRL_LO = 151670, 151645, 151675, 151643  # qwen-1ch PAD, <|im_end|> (= EOB), SOB; first Qwen control id
EOB_ID = EOS_ID
SOA_ID, EOA_ID = 151673, 151674                                  # data_1ch's user-turn markers (never on a speech row; asserted)
NW_ID = 151671                                                   # NEW_WORD (klpre tts_pack / s2model NW_ID; user 10-02: speech rows carry it)
DEPTH, CBS, FPS, ROW_FRAMES = 32, 2048, 12.5, 7500
CD = int(os.environ.get("PPM_CD", "2"))                         # PPM 10-11 (user): code delay = ONE knob PPM_CD (default 2) for Mimi AND pose
LIGHT_FRAC = 0.25
WRAP = 2                                                         # frames added per utterance: SOB + EOB


# ------------------------------------------------------------------------------------------------ flat store (speech_m, unchanged)
def node_number(host):
    """slinky-N -> N; a SHARED store (NODE.json node = a non-numeric name, e.g. 'weka-tts128m') -> 0."""
    t = str(host).split("-")[-1]
    return int(t) if t.isdigit() else 0


def load_node(flat_root):
    """-> dict(shards, length int32 [N], shard_of int16 [N], row_of int32 [N], digest, node, flat_root) from NODE.json + index.npz."""
    meta = json.load(open(f"{flat_root}/NODE.json"))
    shards = list(meta["shards"])
    assert shards == sorted(shards), shards
    L, S, R = [], [], []
    for k, s in enumerate(shards):
        d = json.load(open(f"{flat_root}/{s}.DONE"))
        z = np.load(f"{flat_root}/{s}.index.npz")
        ln = z["length"].astype(np.int32)
        assert len(ln) == d["rows"] == meta["rows"][k] and int(ln.sum()) == d["frames"], (s, len(ln), d["rows"])
        L.append(ln); S.append(np.full(len(ln), k, dtype=np.int16)); R.append(np.arange(len(ln), dtype=np.int32))
    h = hashlib.blake2b(digest_size=16)
    for s, dg in zip(shards, meta["ids_blake2b"]):
        h.update(s.encode()); h.update(dg.encode())
    return {"shards": shards, "length": np.concatenate(L), "shard_of": np.concatenate(S), "row_of": np.concatenate(R),
            "digest": h.hexdigest(), "node": meta["node"], "flat_root": flat_root}




# ------------------------------------------------------------------------------------------------ plan (Qwen tts_pack, 1ch costs)
def pack(order, length, row_frames, n_rows=None):
    """Qwen tts_pack.pack's "close" rule with the 1ch cost: EVERY piece costs L + 2 frames (its SOB + EOB). A piece that does not fit is
    DISCARDED (never split, never seen) and the row closes. -> (rows, consumed incl. discards, discarded, dropped tail pieces, row starts)."""
    rows, starts, cur, used, disc, i0 = [], [], [], 0, 0, 0
    ln = length[order]
    i, n = 0, len(order)
    while i < n:
        L = int(ln[i])
        assert 0 < L and L + WRAP <= row_frames, (L, row_frames)
        if not cur:
            i0 = i
        cost = L + WRAP
        if used + cost <= row_frames:
            cur.append(int(order[i])); used += cost; i += 1
            continue
        disc += 1; i += 1                                   # the misfit piece: discarded, never split, never seen
        rows.append(np.asarray(cur, dtype=np.int64)); starts.append(i0); cur, used = [], 0
        if n_rows is not None and len(rows) == n_rows:
            return rows, i, disc, 0, starts
    return rows, i, disc, len(cur), starts


def row_used(length, row):
    return int(length[row].sum()) + WRAP * len(row)


def plan(node, seed, lworld, lrank, row_frames=ROW_FRAMES, n_eval_rows_per_rank=1):
    """THIS rank's plan (speech_m.plan: same permutation, eval split and stripes; 1ch pack costs). Digest tagged '1ch'."""
    N = len(node["length"])
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), node_number(node["node"])]))
    perm = rng.permutation(N).astype(np.int64)
    n_ev = lworld * int(n_eval_rows_per_rank)
    ev_rows, E, ev_disc, _, _ = pack(perm, node["length"], row_frames, n_rows=n_ev)
    assert len(ev_rows) == n_ev, (len(ev_rows), n_ev)
    stripe = perm[E:][lrank::lworld]
    rows, consumed, disc, tail, starts = pack(stripe, node["length"], row_frames)
    fill = np.array([row_used(node["length"], r) for r in rows], dtype=np.int64) if rows else np.zeros(0, np.int64)
    h = hashlib.blake2b(digest_size=16)
    h.update(b"1ch-sob-eob"); h.update(perm[:E].tobytes()); h.update(stripe.tobytes()); h.update(np.asarray(starts, dtype=np.int64).tobytes())
    stats = {"pieces_node": N, "eval_pieces": int(E), "eval_rows": n_ev, "eval_discarded": int(ev_disc), "stripe_pieces": int(len(stripe)),
             "rows": len(rows), "pieces_in_rows": int(sum(len(r) for r in rows)), "discarded": int(disc), "tail_dropped_pieces": int(tail),
             "fill_mean": float(fill.mean() / row_frames) if len(fill) else 0.0, "fill_min": float(fill.min() / row_frames) if len(fill) else 0.0}
    return {"rows": rows, "starts": np.asarray(starts, dtype=np.int64), "eval_rows": ev_rows[lrank::lworld], "eval_rows_all": ev_rows, "digest": h.hexdigest(),
            "stats": stats, "row_frames": int(row_frames), "seed": int(seed), "node_digest": node["digest"], "consumed": int(consumed)}


# ------------------------------------------------------------------------------------------------ text: words -> pieces -> frames
def word_pieces(tok, words):
    """Qwen speech_data.word_pieces (the whole word list as ONE string " w1 w2 ..." with offsets; a piece belongs to the word whose span
    holds its start offset) on the Qwen3 tokenizer. split_special_tokens: a literal "<|im_end|>" is plain text, never a control id;
    every id is asserted < CTRL_LO (fail loudly)."""
    text, spans = "", []
    for i, w in enumerate(words):
        piece = w["text"] if i == 0 else " " + w["text"]
        spans.append((len(text), len(text) + len(piece))); text += piece
    if not spans:
        return []
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False, split_special_tokens=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    assert all(int(t) < CTRL_LO for t in ids), ("a speech word tokenised to a control id (>= 151643)", [t for t in ids if t >= CTRL_LO][:5])
    out = [[] for _ in spans]; wi = 0
    for j, (ts, _) in enumerate(offs):
        while wi < len(spans) and ts >= spans[wi][1]:
            wi += 1
        if wi < len(spans) and spans[wi][0] <= ts:
            out[wi].append(int(ids[j]))
    return out


def lane(words, piece_ids, piece_offsets, n, wif=None):
    """Qwen collate.lane: true frame, no lead (floor(start * 12.5)), later pieces on the following frames, never before write + 1."""
    frames = np.full(n, PAD_ID, dtype=np.int64); write = -1
    for w, (s, e) in zip(words, zip(piece_offsets[:-1], piece_offsets[1:])):
        pos = max(int(math.floor(float(w["start"]) * FPS + 1e-6)), write + 1)
        for k, tk in enumerate(piece_ids[s:e]):
            if pos >= n:
                break
            frames[pos] = int(tk); write = pos
            if wif is not None and k == 0:
                wif[pos] = 1
            pos += 1
    return frames




def insert_new_word(ids, wif):
    """klpre tts_pack.insert_new_word VERBATIM: for every word-initial frame f (wif[f] == 1) with f > 0 and ids[f - 1] == PAD:
    ids[f - 1] := NW_ID. In place; returns the count. (1ch: the frame before an utterance's first word is SOB or PAD; SOB stays SOB, a
    word glued to the previous one gets none, the EOB / SOB seam frames are never PAD so never NEW_WORD.)"""
    f = np.nonzero(wif)[0]; f = f[f > 0]; f = f[ids[f - 1] == PAD_ID]
    ids[f - 1] = NW_ID
    return int(len(f))


def row_streams(row, cd=CD, row_frames=ROW_FRAMES, new_word=True):
    """a packed row (load_row item) -> numpy streams ids [T], depth [T, 32], lw [T] + counts (module docstring: the per-frame layout).
    new_word (user 10-02, klpre's rule): the PAD frame right before a word's first piece becomes NEW_WORD (input AND timing target); it
    REPLACES a PAD frame, so the frame budget (n + 2 per utterance) and the row content are unchanged."""
    F = int(row_frames); T = F + cd; U = int(row["used"])
    wif = np.zeros(T, dtype=np.int64)
    assert 0 < U <= F, (U, F)
    ids = np.full(T, PAD_ID, dtype=np.int64)
    depth = np.full((T, DEPTH), -1, dtype=np.int64)
    dropped, s_expect = 0, 0
    off = np.arange(DEPTH, dtype=np.int64) * CBS
    for k, p in enumerate(row["pieces"]):
        s, n = int(p["start"]), int(p["T"])
        assert s == s_expect and int(p["offset"]) == s + 1, ("utterances must be back to back: SOB, n audio frames, EOB", k, s, s_expect)
        ids[s] = SOB_ID
        w = np.zeros(n, dtype=np.int64)
        fr = lane(p["words"], p["piece_ids"], p["piece_offsets"], n, w)
        assert not np.isin(fr, (SOB_ID, EOB_ID, SOA_ID, EOA_ID, NW_ID)).any()
        ids[s + 1:s + 1 + n] = fr
        wif[s + 1:s + 1 + n] = w
        ids[s + 1 + n] = EOB_ID
        dropped += int(len(p["piece_ids"]) - int((fr != PAD_ID).sum()))
        c = np.asarray(p["codes"], dtype=np.int64)
        assert c.shape == (DEPTH, n) and int(c.min()) >= 0 and int(c.max()) < CBS, (c.shape, n)
        depth[cd + s + 1:cd + s + 1 + n] = c.T + off             # audio frame f at position f + cd
        s_expect = s + n + WRAP
    assert s_expect == U, ("utterances do not end at the used frame count", s_expect, U)
    assert int((ids == SOB_ID).sum()) == int((ids == EOB_ID).sum()) == len(row["pieces"]) and ids[0] == SOB_ID and ids[U - 1] == EOB_ID
    nnw = insert_new_word(ids, wif) if new_word else 0
    lw = np.zeros(T, dtype=np.float32); lw[:U] = 1
    return {"ids": ids, "depth": depth, "lw": lw, "used": U, "text_dropped": dropped, "n_utt": len(row["pieces"]), "n_new_word": nnw}


def speech_targets(st, w_text, w_tim, w_eob=None, sob_target=False):
    """-> sel [n] scored positions p, y [n] = ids[p + 1], class masks word / pad / eob / sob, tim = pad | sob(if scored), wt [n] fp32 =
    class weight / n (see the module docstring)."""
    w_eob = float(w_tim if w_eob is None else w_eob)
    ids, lw = st["ids"], st["lw"]
    keep = lw[1:] > 0
    yall = ids[1:]
    if not sob_target:
        keep = keep & (yall != SOB_ID)                         # SOB after an EOB: input-only (packing artefact)
    sel = np.nonzero(keep)[0].astype(np.int64)
    y = ids[sel + 1]
    word, pad, eob, sob, nw = y < CTRL_LO, y == PAD_ID, y == EOB_ID, y == SOB_ID, y == NW_ID
    assert bool((word | pad | eob | sob | nw).all()), ("a scored speech target is not a word piece / PAD / NEW_WORD / EOB / SOB", np.unique(y[~(word | pad | eob | sob | nw)])[:8])
    assert sob_target or not sob.any()
    n = len(sel); assert n > 0
    tim = pad | sob
    wt = ((w_text * word + w_tim * (tim | nw) + w_eob * eob) / n).astype(np.float32)   # NEW_WORD at W_TIM here (reference; trainer: W_NW_CE)
    return {"sel": sel, "y": y, "word": word, "tim": tim, "pad": pad, "eob": eob, "sob": sob, "nw": nw, "wt": wt, "n_keep": n}


def cb0_labels(depth):
    lab = depth[:, 0].copy(); lab[lab < 0] = -100
    return lab


def light_select(depth, frac, seed_words):
    """positions with audio (cb0 and cb1 present), a seeded frac subset (Qwen: keep = max(1, int(n * frac))), ascending."""
    valid = np.nonzero((depth[:, 0] >= 0) & (depth[:, 1] >= 0))[0]
    keep = max(1, int(len(valid) * float(frac)))
    rng = np.random.default_rng(np.random.SeedSequence([int(x) for x in seed_words]))
    return np.sort(rng.permutation(valid)[:keep]).astype(np.int64)



def word_positions(ids):
    """mix-RoPE WORD positions of a speech row (module doc): inclusive count of lane-B tokens (word pieces < CTRL_LO, and EOB)."""
    ids = np.asarray(ids, np.int64)
    lb = (ids < CTRL_LO) | (ids == EOB_ID)
    return np.cumsum(lb).astype(np.int64)


def check_positions(ids, pid, wpid):
    """the rule, asserted: pid = 0..T-1; wpid starts at [ids[0] is lane B], steps by exactly 1 on every word / EOB and by 0 elsewhere."""
    ids, pid, wpid = (np.asarray(x, np.int64) for x in (ids, pid, wpid))
    lb = (ids < CTRL_LO) | (ids == EOB_ID)
    assert np.array_equal(pid, np.arange(len(ids))), "frame positions are not 0..T-1"
    d = np.diff(np.concatenate([[0], wpid]))
    assert np.array_equal(d, lb.astype(np.int64)), "word positions do not advance exactly on the word / EOB tokens"
    assert not lb[np.isin(ids, (PAD_ID, SOB_ID, SOA_ID, EOA_ID, NW_ID))].any()      # PAD / markers / NEW_WORD repeat
    return True


def micro_arrays(st, w_text, w_tim, w_eob, sob_target, light_frac, seed_words):
    """everything one speech row needs, numpy (the trainer's DataLoader worker builds it). wt = the UNRAMPED weights (reference / eval);
    the trainer recomputes the training weights from word / pad / eob / sob + n_keep with its timing ramp."""
    tg = speech_targets(st, w_text, w_tim, w_eob, sob_target)
    T = len(st["ids"])
    wpid = word_positions(st["ids"])
    return {"ids": st["ids"], "depth": st["depth"], "pid": np.arange(T, dtype=np.int64), "wpid": wpid, "sel": tg["sel"], "y": tg["y"],
            "wt": tg["wt"], "word": tg["word"], "tim": tg["tim"], "pad": tg["pad"], "eob": tg["eob"], "sob": tg["sob"], "nw": tg["nw"], "n_keep": tg["n_keep"],
            "cb0": cb0_labels(st["depth"]), "light": light_select(st["depth"], light_frac, seed_words),
            "used": st["used"], "text_dropped": st["text_dropped"], "n_utt": st["n_utt"], "n_new_word": st["n_new_word"]}


def merge_rows(items):
    """N one-row micro_arrays -> ONE merged micro (module doc). Per-row normalisers travel along: n_keep_r (scored positions), n_cb0_r
    (audio positions with a cb0 label), n_light_r (selected light frames); rid / cb0_rid / light_rid map every term to its row."""
    T = len(items[0]["ids"])
    assert all(len(it["ids"]) == T for it in items), "merged rows must have one length (ROW_FRAMES + CD)"
    for it in items:                                               # the mix-RoPE position rule, asserted on every row
        check_positions(it["ids"], it["pid"], it["wpid"])
    out = {"segs": [T] * len(items), "n_rows": len(items)}
    for k in ("ids", "depth", "pid", "wpid", "cb0"):
        out[k] = np.concatenate([it[k] for it in items])
    for k in ("y", "wt", "word", "tim", "pad", "eob", "sob", "nw"):
        out[k] = np.concatenate([it[k] for it in items])
    out["sel"] = np.concatenate([it["sel"] + r * T for r, it in enumerate(items)])
    out["light"] = np.concatenate([it["light"] + r * T for r, it in enumerate(items)])
    out["rid"] = np.concatenate([np.full(len(it["sel"]), r, np.int64) for r, it in enumerate(items)])
    out["cb0_rid"] = np.repeat(np.arange(len(items), dtype=np.int64), T)
    out["light_rid"] = np.concatenate([np.full(len(it["light"]), r, np.int64) for r, it in enumerate(items)])
    out["n_keep_r"] = np.array([it["n_keep"] for it in items], np.float32)
    out["n_cb0_r"] = np.array([int((it["cb0"] >= 0).sum()) for it in items], np.float32)
    out["n_light_r"] = np.array([len(it["light"]) for it in items], np.float32)
    assert (out["n_cb0_r"] > 0).all() and (out["n_light_r"] > 0).all()
    for k in ("used", "text_dropped", "n_utt", "n_new_word"):
        out[k] = int(sum(it[k] for it in items))
    return out


# ------------------------------------------------------------------------------------------------ piece access + dataset
class Pieces:
    """lazy per-worker access to the node's flat shards (mmap codes + mmap Arrow words); Qwen tts_pack.Pieces."""

    def __init__(self, node):
        self.node, self.codes, self.voff, self.words = node, {}, {}, {}

    def _open(self, k):
        if k not in self.codes:
            import pyarrow as pa
            import pyarrow.ipc as ipc
            s = self.node["shards"][k]; root = self.node["flat_root"]
            self.codes[k] = np.load(f"{root}/{s}.codes.npy", mmap_mode="r")
            self.voff[k] = np.load(f"{root}/{s}.index.npz")["voff"]
            self.words[k] = ipc.open_file(pa.memory_map(f"{root}/{s}.words.arrow", "r")).read_all()

    def get(self, pid):
        k, r = int(self.node["shard_of"][pid]), int(self.node["row_of"][pid])
        self._open(k)
        T = int(self.node["length"][pid])
        a, b = int(self.voff[k][r]), int(self.voff[k][r + 1])
        assert b - a == DEPTH * T, (pid, b - a, T)
        codes = np.asarray(self.codes[k][a:b]).reshape(DEPTH, T)
        words = self.words[k].column("words")[r].as_py()
        return codes, words, self.words[k].column("audio_id")[r].as_py()


def load_row(pc, tok, piece_ids, row_frames=ROW_FRAMES):
    """packed row (piece ids) -> {pieces: [{codes, words, piece_ids, piece_offsets, start (SOB frame), offset (first audio frame), T}], used}."""
    pieces, s = [], 0
    for pid in piece_ids:
        codes, words, aid = pc.get(int(pid))
        T = codes.shape[1]
        ws = [{"text": w["word"], "start": float(w["start"]), "end": float(w["end"])} for w in words if 0.0 <= float(w["start"]) < T / FPS]
        pp = word_pieces(tok, ws)
        offs = np.zeros(len(ws) + 1, dtype=np.int32)
        for j, p in enumerate(pp):
            offs[j + 1] = offs[j] + len(p)
        pieces.append({"codes": codes, "words": ws, "piece_ids": np.array([t for p in pp for t in p], dtype=np.int64), "piece_offsets": offs,
                       "start": s, "offset": s + 1, "T": T, "audio_id": aid})
        s += T + WRAP
    assert 0 < s <= row_frames, (s, row_frames)
    return {"pieces": pieces, "used": s}


class SpeechMicros:
    """map-style dataset: item i = merge_rows of rows[i * n : (i + 1) * n] (n = rows per merged micro); row_ids travel per row."""

    def __init__(self, node, rows, row_ids, tokenizer_path, w_text, w_tim, w_eob, sob_target, light_frac, light_seed, lrank, row_frames=ROW_FRAMES, n=2,
                 new_word=True):
        assert len(rows) == len(row_ids) and len(rows) % n == 0, (len(rows), n)
        self.base = SpeechRows(node, rows, row_ids, tokenizer_path, w_text, w_tim, w_eob, sob_target, light_frac, light_seed, lrank, row_frames, new_word)
        self.n = int(n)

    def __len__(self):
        return len(self.base) // self.n

    def __getitem__(self, i):
        its = [self.base[i * self.n + k] for k in range(self.n)]
        out = merge_rows(its)
        out["row_ids"] = [int(it["row_id"]) for it in its]
        out["n_pieces"] = int(sum(it["n_pieces"] for it in its))
        return out


class SpeechRows:
    """map-style dataset: item i = micro_arrays of row rows[i] (+ ids). Tokenizer / mmaps opened lazily per DataLoader worker."""

    def __init__(self, node, rows, row_ids, tokenizer_path, w_text, w_tim, w_eob, sob_target, light_frac, light_seed, lrank, row_frames=ROW_FRAMES, new_word=True):
        self.new_word = bool(new_word)
        assert len(rows) == len(row_ids)
        self.node, self.rows, self.row_ids, self.tok_path = node, rows, list(row_ids), tokenizer_path
        self.w = (float(w_text), float(w_tim), float(w_eob), bool(sob_target)); self.lf, self.ls, self.lrank = float(light_frac), int(light_seed), int(lrank)
        self.F = int(row_frames)
        self.tok = self.pc = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.tok is None:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(self.tok_path); self.pc = Pieces(self.node)
        row = load_row(self.pc, self.tok, self.rows[i], self.F)
        st = row_streams(row, CD, self.F, self.new_word)
        rid = int(self.row_ids[i])
        out = micro_arrays(st, *self.w, self.lf, (self.ls, node_number(self.node["node"]), self.lrank, rid & 0x7FFFFFFF, rid >> 31))
        out.update(row_id=rid, n_pieces=len(row["pieces"]), first_audio_id=row["pieces"][0]["audio_id"])
        return out


# ------------------------------------------------------------------------------------------------ synthetic flat store (tests / smoke)
def write_synthetic_flat(root, node, n_shards=2, pieces_per_shard=200, seed=0, len_range=(38, 375)):
    """a Qwen-format flat store with random codes and random words (ASCII), for the CPU tests and the 1-node smoke. Returns NODE.json."""
    import pyarrow as pa
    import pyarrow.ipc as ipc
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(seed)
    vocab = ["the", "a", "speech", "model", "hello", "world", "mistral", "voice", "tower", "frame", "token", "data", "of", "and", "to"]
    shards, rows, frames, digs = [], [], [], []
    for s in range(n_shards):
        name = f"train-{s:05d}-of-{n_shards:05d}"
        L = rng.integers(len_range[0], len_range[1] + 1, pieces_per_shard).astype(np.int64)
        voff = np.zeros(len(L) + 1, np.int64); voff[1:] = np.cumsum(L * DEPTH)
        codes = rng.integers(0, CBS, int(voff[-1])).astype(np.int16)
        np.save(f"{root}/{name}.codes.npy", codes)
        np.savez(f"{root}/{name}.index.npz", length=L.astype(np.int32), voff=voff)
        aids, wl = [], []
        for i, n in enumerate(L):
            dur = n / FPS; t = float(rng.uniform(0.0, 0.5)); ws = []
            while t < dur - 0.05:
                d = float(rng.uniform(0.15, 0.6))
                ws.append({"word": str(rng.choice(vocab)), "start": t, "end": min(t + d, dur), "probability": 1.0})
                t += d + float(rng.choice([0.0, 0.0, 0.3, 0.8]))
            aids.append(f"syn-{s}-{i}"); wl.append(ws)
        wtype = pa.list_(pa.struct([("word", pa.string()), ("start", pa.float64()), ("end", pa.float64()), ("probability", pa.float64())]))
        tbl = pa.table({"audio_id": pa.array(aids, pa.string()), "words": pa.array(wl, wtype)})
        with pa.OSFile(f"{root}/{name}.words.arrow", "wb") as sink, ipc.new_file(sink, tbl.schema) as w:
            w.write_table(tbl)
        h = hashlib.blake2b(digest_size=16)
        for a in aids:
            h.update(a.encode()); h.update(b"\0")
        h.update(L.astype(np.int32).tobytes())
        d = {"shard": name, "rows": int(len(L)), "frames": int(L.sum()), "ids_blake2b": h.hexdigest(), "synthetic": True}
        json.dump(d, open(f"{root}/{name}.DONE", "w"))
        shards.append(name); rows.append(int(len(L))); frames.append(int(L.sum())); digs.append(h.hexdigest())
    meta = {"node": node, "shards": shards, "rows": rows, "frames": frames, "ids_blake2b": digs, "total_rows": sum(rows),
            "total_frames": sum(frames), "synthetic": True}
    json.dump(meta, open(f"{root}/NODE.json", "w"), indent=1)
    return meta
