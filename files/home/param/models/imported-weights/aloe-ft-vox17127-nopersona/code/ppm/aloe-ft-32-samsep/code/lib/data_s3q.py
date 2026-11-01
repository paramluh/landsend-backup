"""QWEN STAGE-3 (duplex) DATA: the node-local two-channel row store ($ALOE_SCRATCH/s3-rows, stage3/data/rows_build/scripts/
s3rows.py) -> the trainer's merged speech micro. CPU only, no torch.distributed (the trainer's DataLoader workers and the CPU tests
import it).

ROW = one 7,500-frame (600 s, 12.5 Hz) chunk of one podcast episode, TWO channels: 0 = A = 'user' (host / interviewer), 1 = B =
'assistant' (the main talker). Roles exactly as the store labels them. n = n_frames (750..7500; frames >= n are the PAD tail).

PER-CHANNEL STREAM (stage 2's speech_q layout, minus the per-utterance SOB / EOB wrap; code delay CD = 3; T = ROW_FRAMES + CD positions):
  ids_c[p]   p < n: the lane token of frame p = speech_q.lane(words_c, piece_ids_c, piece_offsets_c, n) (word pieces on their true frame
             floor(start x 12.5), later pieces of a word on the following frames, never before the previous piece + 1; pieces that would
             land at >= n are dropped and counted); PAD elsewhere; NEW_WORD (151671) replaces the PAD frame right before every word's
             first piece (speech_q.insert_new_word, klpre's rule) when NEW_WORD=1. p >= n: PAD (tail, no loss).
  depth_c[p] codes of frame p - CD of channel c (global ids cb x 2048 + code), -1 for p < CD and p >= n + CD.
  SOB / EOB: the WLT has NO utterance / segment info (words only: text, start, end, in_overlap, truncated) and rows are continuous
             10-min chunks, so by default NO SOB and NO EOB appear inside a row (S3_EOB_GAP_S = 0). Ablation knob S3_EOB_GAP_S = g > 0:
             EOB (<|im_end|>) on the frame right after a word's last piece when the channel's next word starts >= g s after this word's
             end (a silence-delimited "turn end"; only if that frame is PAD and < n). Never SOB.
TARGETS (per channel, the trunk at position p predicts ids_c[p + 1]): p = 0..n-2 (every frame of the row; no SOB exclusion needed).
  classes word (< 151643) / PAD / NEW_WORD / EOB (gap knob only), asserted exhaustive.
VOICE / LIGHT targets per channel: cb0 of depth_c[p, 0] at every position with audio; light = a seeded LIGHT_FRAC subset of channel c's
  audio positions (seed words (light_seed, node_number, rank, row_id, c)).
WORD POSITIONS (mix-RoPE wpid, stage 1/2's rule generalised): lane-B token = word piece or EOB. S3_WPID 'both': wpid[p] = inclusive count
  over frames 0..p of the frames where EITHER OR BOTH channels carry a lane-B token (+1 per such frame, never +2); 'assistant': the
  assistant ROLE's lane only (stage 2's rule on the main talker; user words repeat). pid = one global frame counter 0..T-1 per row
  (system prefix + conversation).
MERGED MICRO: R rows (S3_ROWS) concatenated: segs [T] x R; ids [2, R*T], depth [2, R*T, 32]; head targets are ordered CHANNEL-MAJOR
  (all of channel 0's scored positions over the R rows, then channel 1's): sel (index into R*T), ch, rid; the TOWER stream is the two
  channels' streams concatenated channel-major (2R segments of T): cb0 [2R*T] (channel 0 rows first), light positions index that stream.
  Per-row per-channel normalisers n_keep / n_cb0 / n_light [R, 2] travel along (every term = each row's own per-channel MEAN, summed).

PLAN (per rank, node-local data; rank -> node = RANK // LW, local = RANK % LW):
  EVAL = whole EPISODES hash-selected (blake2b(salt + 'dataset/podcast_id/episode_id') -> u64 % 10^6 < frac x 10^6; S3_EVAL_FRAC 0.0005,
         S3_EVAL_SALT): the SAME rule on every node, so an eval episode's rows are excluded from training everywhere. The node's eval-episode
         rows sorted by hash(row_key), split by dataset and interleaved A, B, A, B, ...; local rank l takes items [l x E, (l + 1) x E) (E =
         eval rows per rank, default S3_ROWS: with E = 2 every rank evaluates one A row and one B row). eval_sha = blake2b over the node's
         chosen row_keys (the trainer gathers all ranks' keys into the run's eval_rows_sha + an eval manifest with provenance).
  TRAIN = every other row of the node, permuted by a seeded RNG (S3_SEED, crc32 of the NODE.json node name), striped over the node's LW
         ranks (perm[l::LW]); step t consumes the rank's rows [t R, (t + 1) R). Cursor = the step (train_state 'cursors'); the plan digest
         (node digest + stripe) is asserted on resume. epoch = min over ranks of len(rows) // R.
  ROW EXCLUSION (knob, DEFAULT OFF; 10-07): S3_ROW_EXCLUDE=<file> (lines 'row_key<TAB>reason...', '#' comments / a 'row_key' header
         skipped; built by qwen-1ch-data/rowfilter: degenerate-transcript rows) and S3_ROW_EXCLUDE_FROM=<step> (default 0). The stripe is
         computed EXACTLY as above, then its positions >= FROM x R lose every excluded row_key, ORDER PRESERVED; positions < FROM x R (the
         rows a run resumed at step FROM has already consumed) are kept verbatim. So a run resumed at step FROM with FROM = the resume step
         reads every not-yet-seen, not-excluded row of the old plan once, in the old order: no repeat, no skip. Eval rows are NEVER filtered
         (eval set / eval sha unchanged; flagged eval rows are only counted in stats). Off (unset / empty): the plan, its digest and every
         stat are bit-identical to the pre-knob code. On: plan['digest'] is the digest of the FILTERED stripe (it changes on every rank whose
         unconsumed part lost a row); plan['digest_base'] is the unfiltered digest (= the old plan's digest), plan['row_exclude'] = {path, sha,
         from, ...} for the trainer's resume check / train_state."""
_ALOE_SCRATCH = __import__("os").environ["ALOE_SCRATCH"]  # release: node-local caches (aloe.env)
import hashlib
import json
import math
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S3ROWS_DIR = os.path.join(HERE, "data", "rows_build", "scripts")
S2_DIR = os.path.join(os.path.dirname(HERE), "stage2")
for p in (S3ROWS_DIR, S2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
import s3rows as R3  # noqa: E402
import speech_q as SP  # noqa: E402  (stage-2 ids, lane, insert_new_word, light_select; imported, not edited)

PAD_ID, EOB_ID, NW_ID, SOB_ID, CTRL_LO = SP.PAD_ID, SP.EOB_ID, SP.NW_ID, SP.SOB_ID, SP.CTRL_LO
DEPTH, CBS, FPS, CD, ROW_FRAMES = SP.DEPTH, SP.CBS, SP.FPS, SP.CD, R3.ROW_FRAMES
SIDES = R3.SIDES                       # ("user", "assistant") = channel 0 (A), channel 1 (B)
DATA_ROOT = f"{_ALOE_SCRATCH}/s3-rows"
PROV = ("row_key", "dataset", "podcast_id", "episode_id", "chunk_index", "frame_start", "n_frames", "user_speaker", "assistant_speaker")


# ------------------------------------------------------------------------------------------------ hashing / plan
def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def episode_key(dataset, podcast_id, episode_id):
    return f"{dataset}/{podcast_id}/{episode_id}"


def is_eval_episode(ekey, frac, salt):
    return h64(f"{salt}|{ekey}") % 1_000_000 < int(round(float(frac) * 1_000_000))


def load_index(root=DATA_ROOT):
    """node provenance index (rows.parquet): per node row id, row_key, dataset, podcast_id, episode_id, chunk_index, n_frames."""
    import pyarrow.parquet as pq
    t = pq.read_table(f"{root}/rows.parquet", columns=["row", "row_key", "dataset", "podcast_id", "episode_id", "chunk_index", "n_frames"])
    d = {c: t.column(c).to_pylist() for c in t.column_names}
    assert d["row"] == list(range(len(d["row"]))), "rows.parquet 'row' != 0..N-1"
    return d


def load_row_exclude(path):
    """S3_ROW_EXCLUDE file -> (frozenset of row_keys, sha = blake2b-16 over the sorted keys). First tab-separated column = row_key."""
    keys = set()
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln or ln.startswith("#"):
                continue
            k = ln.split("\t", 1)[0].strip()
            if k and k != "row_key":
                keys.add(k)
    h = hashlib.blake2b(digest_size=16)
    for k in sorted(keys):
        h.update(k.encode()); h.update(b"\0")
    return frozenset(keys), h.hexdigest()


def load_eval_list(path):
    """S3_EVAL_LIST file -> (frozenset of episode keys 'dataset/podcast_id/episode_id', sha = blake2b-16 over the sorted keys). '#' lines skipped."""
    keys = set()
    with open(path) as f:
        for ln in f:
            k = ln.strip()
            if k and not k.startswith("#"):
                keys.add(k)
    h = hashlib.blake2b(digest_size=16)
    for k in sorted(keys):
        h.update(k.encode()); h.update(b"\0")
    return frozenset(keys), h.hexdigest()


def row_exclude_env():
    """(path or None, from_step) from S3_ROW_EXCLUDE / S3_ROW_EXCLUDE_FROM (default off)."""
    p = os.environ.get("S3_ROW_EXCLUDE", "").strip()
    return (p or None), int(os.environ.get("S3_ROW_EXCLUDE_FROM", "0") or 0)


def apply_row_exclude(stripe, keys_of, excl, from_pos):
    """stripe positions >= from_pos lose rows whose key is in excl (order preserved); positions < from_pos kept verbatim."""
    head, tail = stripe[:from_pos], stripe[from_pos:]
    keep = np.fromiter((keys_of[int(r)] not in excl for r in tail), bool, len(tail))
    return np.concatenate([head, tail[keep]]).astype(stripe.dtype), int((~keep).sum()), int(sum(keys_of[int(r)] in excl for r in head))


def plan_digest(node, stripe, ev_rows):
    h = hashlib.blake2b(digest_size=16)
    h.update(b"s3-duplex-v1"); h.update(node["digest"].encode()); h.update(stripe.tobytes()); h.update(np.asarray(ev_rows, np.int64).tobytes())
    return h.hexdigest()


def plan(node, idx, seed, lw, local, rows_per_step, n_eval_per_rank, eval_frac, eval_salt, row_exclude="env", row_exclude_from=None, eval_list=None,
         eval_rows_file=None, rank=None):
    """THIS rank's plan (module doc). node = s3rows.load_node(root); idx = load_index(root).
    row_exclude: 'env' (default: S3_ROW_EXCLUDE / S3_ROW_EXCLUDE_FROM), None (off), or a path; row_exclude_from: step (None = env / 0)."""
    N = len(node["length"])
    assert len(idx["row"]) == N, ("rows.parquet rows != NODE.json rows", len(idx["row"]), N)
    assert np.array_equal(np.asarray(idx["n_frames"], np.int32), node["length"]), "rows.parquet n_frames != index.npz length"
    if eval_list is not None:                                          # 10-09 S3_EVAL_LIST: an explicit episode-key set (e.g. a shared split)
        evs = load_eval_list(eval_list)[0] if isinstance(eval_list, str) else frozenset(eval_list)
        ev_ep = np.array([episode_key(a, b, c) in evs for a, b, c in zip(idx["dataset"], idx["podcast_id"], idx["episode_id"])], bool)
    else:
        ev_ep = np.array([is_eval_episode(episode_key(a, b, c), eval_frac, eval_salt)
                          for a, b, c in zip(idx["dataset"], idx["podcast_id"], idx["episode_id"])], bool)
    ev_all = np.flatnonzero(ev_ep)
    tr_all = np.flatnonzero(~ev_ep).astype(np.int64)
    # eval: sorted by hash(row_key), interleaved A / B
    byd = {}
    for i in sorted(ev_all.tolist(), key=lambda i: h64(f"{eval_salt}|row|{idx['row_key'][i]}")):
        byd.setdefault(idx["dataset"][i], []).append(i)
    inter, ks = [], sorted(byd)
    for j in range(max((len(v) for v in byd.values()), default=0)):
        for k in ks:
            if j < len(byd[k]):
                inter.append(byd[k][j])
    E = int(n_eval_per_rank)
    if eval_rows_file is not None:                                     # 10-10 S3_EVAL_ROWS_FILE (opt-in, offline evals): {"world": W, "ranks": [[row_key, ..], ..]}
        ER = json.load(open(eval_rows_file))                           #   = the global rank's eval rows in order (e.g. another run's eval set re-laid)
        pos = {k: i for i, k in enumerate(idx["row_key"])}
        ev_rows = [pos[k] for k in ER["ranks"][int(rank)]]
        assert len(ev_rows) == E and all(ev_ep[i] for i in ev_rows), ("S3_EVAL_ROWS_FILE rows", len(ev_rows), E)
        inter = list(ev_rows) * lw
    else:
        assert len(inter) >= lw * E, ("not enough eval rows on this node", len(inter), lw, E)
        ev_rows = inter[local * E:(local + 1) * E]
    # train: seeded permutation, striped
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(str(node["node"]).encode())]))
    perm = tr_all[rng.permutation(len(tr_all))]
    stripe = perm[local::lw]
    R = int(rows_per_step)
    digest_base = plan_digest(node, stripe, ev_rows)
    if row_exclude == "env":
        rx_path, rx_from = row_exclude_env()
    else:
        rx_path, rx_from = row_exclude, 0
    if row_exclude_from is not None:
        rx_from = int(row_exclude_from)
    rx = None
    if rx_path:
        assert rx_from >= 0, ("S3_ROW_EXCLUDE_FROM must be >= 0", rx_from)
        excl, rx_sha = load_row_exclude(rx_path)
        n0 = len(stripe)
        stripe, n_drop, n_head = apply_row_exclude(stripe, idx["row_key"], excl, min(rx_from * R, n0))
        rx = {"path": os.path.abspath(rx_path), "sha": rx_sha, "from": int(rx_from), "keys_in_file": len(excl), "dropped_stripe": n_drop,
              "kept_consumed": n_head, "stripe_rows_before": n0, "stripe_rows_after": int(len(stripe)),
              "excluded_train_node": int(sum(idx["row_key"][int(i)] in excl for i in tr_all)),
              "flagged_eval_kept": int(sum(idx["row_key"][i] in excl for i in ev_rows))}
    digest = plan_digest(node, stripe, ev_rows)
    hs = hashlib.blake2b(digest_size=16)
    for i in inter[:lw * E]:
        hs.update(idx["row_key"][i].encode()); hs.update(b"\0")
    stats = {"rows_node": N, "eval_episode_rows_node": int(len(ev_all)), "eval_episodes_node": len({episode_key(idx["dataset"][i], idx["podcast_id"][i], idx["episode_id"][i]) for i in ev_all}),
             "eval_by_dataset_node": {k: len(v) for k, v in byd.items()}, "train_rows_node": int(len(tr_all)), "stripe_rows": int(len(stripe)),
             "steps": int(len(stripe) // R), "frames_mean": float(node["length"][stripe].mean()) if len(stripe) else 0.0}
    return {"rows": stripe, "eval_rows": [int(i) for i in ev_rows], "eval_keys": [idx["row_key"][i] for i in ev_rows],
            "eval_node_sha": hs.hexdigest(), "digest": digest, "stats": stats, "node_digest": node["digest"], "node": node["node"],
            **({"digest_base": digest_base, "row_exclude": rx} if rx is not None else {})}


# ------------------------------------------------------------------------------------------------ one row -> per-channel streams
def lane_words(words, piece_ids, piece_offsets, n, wif):
    """speech_q.lane (identical placement) + each word's last written frame (-1 = nothing written)."""
    frames = np.full(n, PAD_ID, dtype=np.int64); write = -1
    last = np.full(len(words), -1, np.int64)
    for j, (w, (s, e)) in enumerate(zip(words, zip(piece_offsets[:-1], piece_offsets[1:]))):
        pos = max(int(math.floor(float(w["start"]) * FPS + 1e-6)), write + 1)
        for k, tk in enumerate(piece_ids[s:e]):
            if pos >= n:
                break
            frames[pos] = int(tk); write = pos; last[j] = pos
            if k == 0:
                wif[pos] = 1
            pos += 1
    return frames, last


def channel_stream(row, side, cd=CD, row_frames=ROW_FRAMES, new_word=True, eob_gap_s=0.0, off=0, n_use=None):
    """channel `side`'s conversation stream. off = P (system-prefix length; the conversation's frame 0 sits at position off), n_use =
    the conversation frames kept (the row's TAIL is cropped by the prefix: n_use = min(n, row_frames - off)). off 0 / n_use None = the
    basic layout exactly."""
    F = int(row_frames); T = F + cd; n_full = int(row["n_frames"])
    n = n_full if n_use is None else int(n_use)
    assert 0 < n <= n_full and off + n <= F, (n, n_full, off, F)
    words, pid, offs = row[f"words_{side}"], np.asarray(row[f"piece_ids_{side}"], np.int64), np.asarray(row[f"piece_offsets_{side}"], np.int64)
    ids = np.full(T, PAD_ID, dtype=np.int64)
    wif = np.zeros(T, dtype=np.int64)
    fr, last = lane_words(words, pid, offs, n, wif[off:off + n])
    ids[off:off + n] = fr
    n_eob = 0
    if eob_gap_s > 0:
        for j in range(len(words)):
            nxt = float(words[j + 1]["start"]) if j + 1 < len(words) else None
            if nxt is not None and nxt - float(words[j]["end"]) >= eob_gap_s and last[j] >= 0:
                q = int(last[j]) + 1
                if q < n and ids[off + q] == PAD_ID:
                    ids[off + q] = EOB_ID; n_eob += 1
    assert not np.isin(ids, (SOB_ID, NW_ID, SP.SOA_ID, SP.EOA_ID)).any()
    if off > 0:
        wif[off] = 0                                     # as the basic layout (frame 0 has no PAD before it): a word at conversation frame 0 gets no NEW_WORD
    nnw = SP.insert_new_word(ids, wif) if new_word else 0
    if off > 0:
        assert ids[off - 1] != NW_ID                     # the PAD prefix of the stream: a NEW_WORD may only land inside the conversation
    c = np.asarray(row[f"codes_{side}"])[:, :n].astype(np.int64)
    assert c.shape == (DEPTH, n) and int(c.min()) >= 0 and int(c.max()) < CBS, (c.shape, n)
    depth = np.full((T, DEPTH), -1, dtype=np.int64)
    depth[off + cd:off + cd + n] = c.T + np.arange(DEPTH, dtype=np.int64) * CBS
    return {"ids": ids, "depth": depth, "n": n, "text_dropped": int(len(pid) - int((fr < CTRL_LO).sum())), "n_new_word": int(nnw),
            "n_eob": n_eob, "n_words": len(words), "n_pieces": int(len(pid))}


def word_positions(ids_u, ids_a, mode, P=0):
    """stage 1's system-prefix rule (items_q.row_word_positions): prefix positions 0..P-1, conversation position p >= P gets
    P + #lane-B tokens in conversation positions P..p (inclusive). P = 0: the basic rule."""
    lb = lambda x: ((x < CTRL_LO) | (x == EOB_ID)).astype(np.int64)   # noqa: E731
    assert mode in ("both", "assistant"), mode
    c = (lb(ids_u[P:]) | lb(ids_a[P:])) if mode == "both" else lb(ids_a[P:])     # both: +1 per frame where EITHER or BOTH tracks have a word
    return np.concatenate([np.arange(P), P + np.cumsum(c)]).astype(np.int64)


def check_positions(ids_u, ids_a, pid, wpid, mode, P=0):
    lb = lambda x: ((x < CTRL_LO) | (x == EOB_ID)).astype(np.int64)   # noqa: E731
    assert np.array_equal(pid, np.arange(len(ids_a))), "frame positions are not 0..T-1"
    assert np.array_equal(wpid[:P], np.arange(P)), "prefix word positions != 0..P-1"
    d = np.diff(np.concatenate([[P], wpid[P:]]))
    want = (lb(ids_u[P:]) | lb(ids_a[P:])) if mode == "both" else lb(ids_a[P:])
    assert np.array_equal(d, want), f"word positions do not advance exactly on the lane-B tokens ({mode})"
    return True


def row_arrays(row, cfg, rank, row_id, node_name, light_frac, sp=None, swap=0):
    """one row -> both channels' streams + targets (numpy). cfg: dict(new_word, eob_gap_s, wpid, light_seed, row_frames).
    sp (S3_SYSPROMPT; None = the basic layout, unchanged): the system prompt of this row = sysprompt.build_prefix(...) output
    {ids [2, P], depth [2, P, 32], present [2, P], P, ...}: positions 0..P-1 = the prefix, the conversation starts at P, its tail cropped."""
    P = 0 if sp is None else int(sp["P"])
    n_full = int(row["n_frames"])
    nc = min(n_full, int(cfg["row_frames"]) - P)
    if cfg.get("no_crop"):                                              # 10-09 S3_SCRIPT: the conversation is never cropped by the prefix
        assert nc == n_full, ("prefix + conversation exceed the row budget (S3_ROW_FRAMES); no cropping allowed", P, n_full, cfg["row_frames"])
    assert nc >= 2, ("conversation too short after the prefix crop", n_full, P)
    std = [channel_stream(row, s, CD, cfg["row_frames"], cfg["new_word"], cfg["eob_gap_s"], off=P, n_use=nc) for s in SIDES]   # data roles
    roles = (1, 0) if swap else (0, 1)                                  # model channel c holds data role roles[c] (text2_s3q swap rule)
    sts = [std[roles[0]], std[roles[1]]]
    T = len(sts[0]["ids"])
    ids = np.stack([s["ids"] for s in sts]); depth = np.stack([s["depth"] for s in sts])
    present = np.ones((2, T), np.float32)
    if P:
        assert (depth[:, :P + CD] == -1).all(), "conversation codes leak into the prefix (or into positions P..P+2)"
        assert (ids[:, :P] == PAD_ID).all()
        ids[:, :P], depth[:, :P], present[:, :P] = sp["ids"], sp["depth"], sp["present"]
        assert (depth[:, P:P + CD] == -1).all(), "prefix codes leak into the conversation"
    iu, ia = ids[roles.index(0)], ids[roles.index(1)]
    wpid = word_positions(iu, ia, cfg["wpid"], P)
    out = {"ids": ids, "depth": depth, "present": present, "pid": np.arange(T, dtype=np.int64), "wpid": wpid, "n": nc, "P": P,
           "roles": roles, "swap": int(swap)}
    for c, s in enumerate(sts):
        dr = roles[c]                                                   # light seeds follow the DATA track (symmetry under swap)
        sel = np.arange(P, P + nc - 1, dtype=np.int64)                  # the conversation only: never a prefix position (no loss there)
        y = ids[c, sel + 1]
        word, pad, eob, nw = y < CTRL_LO, y == PAD_ID, y == EOB_ID, y == NW_ID
        assert bool((word | pad | eob | nw).all()), ("a scored duplex target is not word / PAD / NEW_WORD / EOB", np.unique(y[~(word | pad | eob | nw)])[:8])
        lab = depth[c, :, 0].copy(); lab[lab < 0] = -100
        lab[:P + CD] = -100                                             # no cb0 loss on the prefix (its codes sit at < P)
        if sp is None and not cfg.get("frame_light"):                  # frame_light (S3_SCRIPT): the per-frame rule below with and without a prefix
            light = SP.light_select(depth[c], light_frac, (int(cfg["light_seed"]), SP.node_number(node_name), int(rank), row_id & 0x7FFFFFFF, row_id >> 31, dr))
        else:                                                           # per conversation FRAME (independent of P / crop: the with- vs without-prompt
            u = np.random.default_rng(np.random.SeedSequence([int(cfg["light_seed"]), h64(row["row_key"]) & 0x7FFFFFFF, dr])).random(int(cfg["row_frames"]))
            f = np.flatnonzero(u[:nc] < light_frac)                     #   eval compares the same frames)
            if not len(f):
                f = np.array([int(np.argmin(u[:nc]))])
            light = (P + CD + f).astype(np.int64)
        assert (light >= P + CD).all() and (lab[:P + CD] == -100).all()
        out[c] = {"sel": sel, "y": y, "word": word, "pad": pad, "eob": eob, "nw": nw, "cb0": lab, "light": light,
                  **{k: s[k] for k in ("text_dropped", "n_new_word", "n_eob", "n_words", "n_pieces")}}
    if sp is not None and sp.get("mask"):                               # S3_SP_SRC own-row fallback (sysprompt.apply_clip_mask): drop the clip window's targets
        import sysprompt as SPR
        out["clip_mask"] = []
        for c, w0, L in sp["mask"]:
            mf = SPR.apply_clip_mask(out[c], ids[c], P, nc, CD, int(w0), int(L), CTRL_LO)
            g_ = out[c]["sel"] + 1 - P                                      # 10-09: zero scored targets left inside the masked clip window
            assert not mf[g_[g_ < nc]].any(), ("a scored target inside an own-row clip window", c, w0, L)
            out["clip_mask"].append((int(c), int(w0), int(L), int(mf.sum())))
    return out


def merge(items):
    """R row_arrays -> ONE merged micro (module doc)."""
    T = len(items[0]["pid"]); R = len(items)
    assert all(len(it["pid"]) == T for it in items)
    out = {"segs": [T] * R, "n_rows": R, "T": T}
    out["ids"] = np.concatenate([it["ids"] for it in items], 1)
    out["depth"] = np.concatenate([it["depth"] for it in items], 1)
    out["present"] = np.concatenate([it["present"] for it in items], 1)
    out["P"] = np.array([it["P"] for it in items], np.int64)
    out["nc"] = np.array([it["n"] for it in items], np.int64)
    out["swap"] = np.array([it["swap"] for it in items], np.int64)
    out["roles"] = np.array([it["roles"] for it in items], np.int64)
    out["pid"] = np.concatenate([it["pid"] for it in items])
    out["wpid"] = np.concatenate([it["wpid"] for it in items])
    sel, ch, rid, ys, cls = [], [], [], [], {k: [] for k in ("word", "pad", "eob", "nw")}
    cb0, light, lrid, lch = [], [], [], []
    nk, n0, nl = np.zeros((R, 2), np.float32), np.zeros((R, 2), np.float32), np.zeros((R, 2), np.float32)
    for c in (0, 1):
        for r, it in enumerate(items):
            d = it[c]
            sel.append(d["sel"] + r * T); ch.append(np.full(len(d["sel"]), c, np.int64)); rid.append(np.full(len(d["sel"]), r, np.int64)); ys.append(d["y"])
            for k in cls:
                cls[k].append(d[k])
            cb0.append(d["cb0"])
            light.append(d["light"] + (c * R + r) * T); lrid.append(np.full(len(d["light"]), r, np.int64)); lch.append(np.full(len(d["light"]), c, np.int64))
            nk[r, c] = len(d["sel"]); n0[r, c] = int((d["cb0"] >= 0).sum()); nl[r, c] = len(d["light"])
    out.update(sel=np.concatenate(sel), ch=np.concatenate(ch), rid=np.concatenate(rid), y=np.concatenate(ys), **{k: np.concatenate(v) for k, v in cls.items()},
               cb0=np.concatenate(cb0), light=np.concatenate(light), light_rid=np.concatenate(lrid), light_ch=np.concatenate(lch),
               n_keep=nk, n_cb0=n0, n_light=nl)
    out["cb0_rid"] = np.tile(np.repeat(np.arange(R, dtype=np.int64), T), 2)
    out["cb0_ch"] = np.repeat(np.arange(2, dtype=np.int64), R * T)
    assert (nk > 0).all() and (n0 > 0).all() and (nl > 0).all()
    out["n0_sel"] = int(sum(len(it[0]["sel"]) for it in items))       # first channel-1 head position in sel
    for k in ("text_dropped", "n_new_word", "n_eob", "n_words", "n_pieces"):
        out[k] = np.array([[it[c][k] for c in (0, 1)] for it in items], np.int64)
    out["used"] = int(sum(it["n"] for it in items))
    return out


class S3Micros:
    """map-style dataset: item i = merge of the rank's rows [i R, (i + 1) R); s3rows.Rows opened lazily per DataLoader worker.
    spc (S3_SYSPROMPT config dict, None = off): every row gets its sampled system prompt (sysprompt.sample: deterministic per row_key);
    force_state ('NONE' / 'A' / 'B' / 'BOTH', eval variants) overrides the sampled state (the clips stay the row's own).
    S3_PERSONA=1 (env, default 0; not used by training yet): each item also carries out['persona'] = json [R] of {"A": ..., "B": ...}, the GCS
    persona system-prompt fields of the data role on model channel A / B (swap-consistent; persona_s3q.py, <shard>.persona.arrow sidecars).
    ptc (S3_PERSONA_TEXT config dict, persona_text_s3q.default_cfg; None = off, the default: items bit-identical to before): the speakers'
    persona TEXT blocks join the system prompt (needs spc); pt_mode 'rand' (sampled; training) | 'none' | 'all' (eval variants)."""

    def __init__(self, node, rows, row_ids, cfg, rank, R, light_frac, check=True, spc=None, force_state=None, swap_mode="none", eval_swap=False, flip=False,
                 ptc=None, pt_mode="rand", clip_src=None, scc=None, script_mode="rand"):
        assert len(rows) == len(row_ids) and len(rows) % R == 0, (len(rows), len(row_ids), R)
        self.clip_src = clip_src or ((spc or {}).get("src") or "own")     # S3_SP_SRC (sysprompt leak fix): own (default) | sibling | foreign (eval control)
        assert self.clip_src in ("own", "sibling", "foreign", "bank"), self.clip_src
        self.sibx = None
        self.bankx = None                                                   # 10-11 S3_SP_SRC=bank (sysprompt.bank_sources)
        self.node, self.rows, self.row_ids, self.cfg, self.rank, self.R, self.lf, self.check = node, list(rows), list(row_ids), dict(cfg), int(rank), int(R), float(light_frac), check
        self.spc, self.force = spc, force_state
        self.swap_mode, self.eval_swap, self.flip = swap_mode, bool(eval_swap), bool(flip)   # eval: fixed balanced (slot parity); flip: inverted (symmetry check)
        self.st = None
        self.persona = os.environ.get("S3_PERSONA", "0") == "1"           # persona sidecar (persona_s3q; off by default, not used by training yet)
        self.ps = None
        self.ptc, self.pt_mode = ptc, pt_mode                               # persona text (persona_text_s3q; None = off)
        assert ptc is None or spc is not None, "S3_PERSONA_TEXT needs the system prompt (S3_SYSPROMPT=1)"
        self.scc, self.script_mode, self.sc = scc, script_mode, None          # 10-09 S3_SCRIPT (script_s3q; None = off): rand | on | off
        assert scc is None or (ptc is None and script_mode in ("rand", "on", "off")), ("S3_SCRIPT is exclusive with S3_PERSONA_TEXT", script_mode)

    def __len__(self):
        return len(self.rows) // self.R

    def script(self, row, j, sw):
        """S3_SCRIPT: -> (script token ids or None, provenance fields) for row j of this rank (script_s3q doc)."""
        import script_s3q as SC
        if self.sc is None:
            self.sc = SC.ScriptSidecar(self.node)
        rec = self.sc.get(int(self.rows[j]), row["row_key"])
        want = self.script_mode == "on" or (self.script_mode == "rand" and SC.choose(row["row_key"], self.scc["seed"], self.scc["p"]))
        has = bool(want and rec["status"] == "ok" and rec["turns"])
        ids = SC.encode(SC.script_text(rec["turns"], sw)) if has else None
        return ids, {"script": int(has), "script_wanted": int(want), "script_status": rec["status"], "script_tokens": len(ids) if has else 0,
                     "script_turns": int(rec["n_turns"]), "script_dropped": int(rec["n_dropped"])}

    def resolve(self, smp, row):
        """S3_SP_SRC: replace the sampled clips' sources (sysprompt.resolve_sources; 'own' = unchanged, the default)."""
        if self.clip_src in ("own", "bank"):                                 # bank: already applied before the state (__getitem__)
            return smp
        import sysprompt as SPR
        if self.sibx is None:
            self.sibx = SPR.SibIndex(self.node, self.st if self.st is not None else R3.Rows(self.node))
        return SPR.resolve_sources(smp, row, self.spc, self.sibx, self.clip_src)

    def __getitem__(self, i):
        if self.st is None:
            self.st = R3.Rows(self.node)
        its, prov, pers = [], [], []
        for k in range(self.R):
            j = i * self.R + k
            row = self.st[int(self.rows[j])]
            if self.check:
                R3.check_row(row)
            pv = {k_: row[k_] for k_ in PROV}
            import text2_s3q as T2
            sw = (j % 2) if self.eval_swap else T2.swap_of(self.swap_mode, int(self.row_ids[j]), f"speech/{row['row_key']}")
            sw = int(sw) ^ int(self.flip)
            pv.update(swap=sw, chan_A_role=SIDES[1 if sw else 0], chan_B_role=SIDES[0 if sw else 1])
            if self.persona:                                                    # per-row persona fields, channel A/B consistent with the swap
                import persona_s3q as PS
                if self.ps is None:
                    self.ps = PS.PersonaSidecar(self.node)
                pch = PS.by_channel(self.ps.get(int(self.rows[j]), row["row_key"]), sw)
                pv.update(persona_A_status=pch["A"]["status"], persona_B_status=pch["B"]["status"])
                pers.append(pch)
            spx = None
            sc_ids = None
            if self.scc is not None:                                            # 10-09 S3_SCRIPT: the turn-taking script (script_s3q)
                sc_ids, scp = self.script(row, j, sw)
                pv.update(scp)
            if self.spc is not None:
                import sysprompt as SPR
                smp = SPR.sample(row, self.spc)
                if self.clip_src == "bank":                                     # 10-11: the call's bank reference clips (audio only), drawn state kept
                    if self.bankx is None:
                        self.bankx = SPR.BankIndex(self.spc["bank_map"], self.spc["bank_index"])
                    smp = SPR.bank_sources(smp, row, self.bankx)
                state = self.force or SPR.model_state(smp, sw)
                if state != "NONE":                                         # S3_SP_SRC: the shown clips' sources (states / feasibility unchanged)
                    smp = self.resolve(smp, row)
                segs = SPR.segments(state, smp, sw)
                if self.ptc is not None:                                    # persona text blocks (own rng stream; clip draws unchanged)
                    import persona_s3q as PS
                    import persona_text_s3q as PT
                    if self.ps is None:
                        self.ps = PS.PersonaSidecar(self.node)
                    pts = PT.sample(row["row_key"], self.ps.get(int(self.rows[j]), row["row_key"]), self.ptc)
                    segs, pti = PT.segments(segs, pts, sw, self.pt_mode, self.ptc, state=(state if self.force is None else None))
                    pv.update(pt_mode=self.pt_mode, **PT.prov(pti))
                if sc_ids is not None:                                          # S3_SYSPROMPT + S3_SCRIPT: the script is the system block's
                    import script_s3q as SC                                     # only text, after the voice clips
                    segs = list(segs) + [SC.segment(sc_ids)]
                spx = SPR.build_prefix(row, segs, self.spc)
                pv.update(sp_state=state, sp_sampled=SPR.model_state(smp, sw), sp_P=int(spx["P"]), sp_clips=SPR.clip_prov(state, smp, row, sw))
                if spx["P"] == 0:
                    spx = None
            if sc_ids is not None and self.spc is None:                         # S3_SCRIPT alone: <SOS_SCRIPT> .. <EOS_SCRIPT> is the whole prefix
                import script_s3q as SC
                spx = SC.build_prefix(sc_ids, pad_id=PAD_ID)
            if self.scc is not None:
                pv["script_P"] = int(spx["P"]) if spx is not None else 0
            a = row_arrays(row, self.cfg, self.rank, int(self.row_ids[j]), self.node["node"], self.lf, sp=spx, swap=sw)
            if a.get("clip_mask"):
                pv["sp_mask"] = a["clip_mask"]
            check_positions(a["ids"][a["roles"].index(0)], a["ids"][a["roles"].index(1)], a["pid"], a["wpid"], self.cfg["wpid"], a["P"])
            its.append(a); prov.append(pv)
        out = merge(its)
        out["row_ids"] = [int(self.row_ids[i * self.R + k]) for k in range(self.R)]
        out["node_rows"] = [int(self.rows[i * self.R + k]) for k in range(self.R)]
        out["prov"] = json.dumps(prov)
        if self.persona:
            out["persona"] = json.dumps(pers)                               # [R] of {"A": {role, status, system_prompt, ...}, "B": ...}
        return out
