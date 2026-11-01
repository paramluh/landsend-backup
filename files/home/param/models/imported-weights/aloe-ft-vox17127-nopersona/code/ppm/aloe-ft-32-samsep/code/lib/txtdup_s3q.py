"""STAGE-3 TEXT-DUPLEX CE rows (S3_TXT_MODE=duplex_ce): Demeter's Gemma text-duplex conversations rendered in OUR speech-row lane format
with NO audio, trained with the speech rows' text CE on both channels. Replaces the text-KL micro (one micro per GPU per step). CPU only
(DataLoader workers, CPU tests); the store is built by data/txtdup_build.py (see its doc for the source, selection and packing rules).

ONE CONVERSATION (data sides a = speaker 0 -> data role 0 'user', b = speaker 1 -> data role 1 'assistant'; n = min(n_frames, CAP)):
  lane of each side = data_s3q.lane_words (= the speech rows' speech_q.lane rule: a word's first piece on frame floor(start x 12.5 + 1e-6),
  later pieces on the following frames, never before the previous piece + 1, pieces at >= n dropped and counted) + speech_q.insert_new_word
  (NEW_WORD on the PAD frame right before a word's first piece, when NEW_WORD=1) = data_s3q.channel_stream with off 0 / no codes, identical
  code. No SOB / EOB (S3_EOB_GAP_S 0 as the speech rows; asserted none), no system prompt (Demeter has none; the speech rows' prompt is an
  audio clip, impossible here). No codes: the trunk input is sum_c s_c emb_c(lane_c) (S3In with depth None == AudioEmb of all-absent codes,
  which is exactly 0: heads_s2q.code_feedback masks frames with code -1).
SWAP: model channel c holds data role roles[c]; roles = (1, 0) if swap else (0, 1); swap = text2_s3q.swap_of(S3_SWAP, slot, 'txtdup/<conv_id>')
  (train), eval: fixed balanced = parity of the conversation's position in the eval list (xor flip for the symmetry check).
PACK -> MICRO (one pack = one micro, K conversations, each its OWN attention segment; segs [n_1 .. n_K], N = sum n_k):
  ids [2, N] (model channels), pid = 0..n_k-1 per segment, wpid = data_s3q.word_positions(user lane, assistant lane, S3_WPID) per segment
  (P 0), scored positions channel-major: for c in (0, 1), for k: segment positions 0..n_k-2 (target = that channel's token at the next frame);
  classes word (< 151643) / PAD / NEW_WORD, asserted exhaustive. n_keep [1, 2] = scored positions per CHANNEL over the whole pack: the
  trainer normalises each channel's loss by it (one mean per channel per pack = Demeter's one weighted mean per packed row, model.py:338)."""
_ALOE_ROOT = __import__("os").environ["ALOE_ROOT"]        # release: every path comes from aloe.env (ALOE_ROOT)
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import data_s3q as DS  # noqa: E402

SP = DS.SP
PAD_ID, NW_ID, EOB_ID, CTRL_LO = DS.PAD_ID, DS.NW_ID, DS.EOB_ID, DS.CTRL_LO
DEFAULT_ROOT = f"{_ALOE_ROOT}/data/gemma-txtdup-v1"
STATS = ("text_dropped", "n_new_word", "n_words", "n_pieces")


def manifest(root):
    return json.load(open(f"{root}/MANIFEST.json"))


class Store:
    """the txtdup store (memory-mapped Arrow shards + packs.npz). Opened lazily per DataLoader worker."""

    def __init__(self, root):
        self.root = root
        self.man = manifest(root)
        z = np.load(f"{root}/packs.npz")
        self.train_ptr, self.train_idx, self.eval_ptr, self.eval_idx = (z[k] for k in ("train_ptr", "train_idx", "eval_ptr", "eval_idx"))
        self.soff = np.concatenate([[0], np.cumsum(np.asarray(self.man["shard_rows"], np.int64))])
        self.cap = int(self.man["cap"])
        self._tabs = {}

    def n_packs(self, split):
        return len(self.train_ptr if split == "train" else self.eval_ptr) - 1

    def pack(self, split, p):
        ptr, idx = (self.train_ptr, self.train_idx) if split == "train" else (self.eval_ptr, self.eval_idx)
        assert 0 <= p < len(ptr) - 1, (split, p, len(ptr) - 1)
        return idx[ptr[p]:ptr[p + 1]].tolist()

    def _tab(self, k):
        if k not in self._tabs:
            import pyarrow as pa
            self._tabs[k] = pa.ipc.open_file(pa.memory_map(f"{self.root}/shard-{k:05d}.arrow")).read_all()
        return self._tabs[k]

    def get(self, i):
        k = int(np.searchsorted(self.soff, i, side="right") - 1)
        r = self._tab(k).slice(i - int(self.soff[k]), 1).to_pylist()[0]
        r["store_idx"] = int(i)
        return r


def conv_lanes(rec, cap, new_word=True):
    """one stored conversation -> data-role lanes ids [2, n] (0 = side a = user, 1 = side b = assistant) + per-side counts."""
    n = min(int(rec["n_frames"]), int(cap))
    assert n >= 2, ("conversation too short", rec.get("conv_id"), n)
    ids = np.full((2, n), PAD_ID, np.int64)
    cnt = {k: [0, 0] for k in STATS}
    for d, side in enumerate(("a", "b")):
        st = rec[f"start_{side}"]
        pid = np.asarray(rec[f"pids_{side}"], np.int64)
        off = np.asarray(rec[f"poff_{side}"], np.int64)
        assert len(off) == len(st) + 1 and int(off[-1]) == len(pid), ("pieces / offsets mismatch", rec.get("conv_id"), side)
        wif = np.zeros(n, np.int64)
        fr, _ = DS.lane_words([{"start": s} for s in st], pid, off, n, wif)
        nnw = SP.insert_new_word(fr, wif) if new_word else 0
        assert not np.isin(fr, (SP.SOB_ID, EOB_ID, SP.SOA_ID, SP.EOA_ID)).any()
        ids[d] = fr
        cnt["text_dropped"][d] = int(len(pid) - int((fr < CTRL_LO).sum()))
        cnt["n_new_word"][d], cnt["n_words"][d], cnt["n_pieces"][d] = int(nnw), len(st), int(len(pid))
    return ids, n, cnt


def micro(recs, swaps, cfg):
    """K stored conversations (one pack) + their swaps -> ONE merged micro (module doc). cfg: dict(cap, new_word, wpid)."""
    segs, ids_l, pid_l, wp_l, roles_l, cnts = [], [], [], [], [], []
    for rec, sw in zip(recs, swaps):
        lanes, n, cnt = conv_lanes(rec, cfg["cap"], cfg["new_word"])
        roles = (1, 0) if sw else (0, 1)
        ids = np.stack([lanes[roles[0]], lanes[roles[1]]])            # model channel c <- data role roles[c]
        wp = DS.word_positions(lanes[0], lanes[1], cfg["wpid"], 0)
        DS.check_positions(lanes[0], lanes[1], np.arange(n), wp, cfg["wpid"], 0)
        segs.append(n); ids_l.append(ids); pid_l.append(np.arange(n, dtype=np.int64)); wp_l.append(wp); roles_l.append(roles); cnts.append(cnt)
    N = int(sum(segs)); K = len(segs)
    assert N <= int(cfg["cap"]) or K == 1, ("pack longer than the cap", segs, cfg["cap"])
    ids = np.concatenate(ids_l, 1)
    starts = np.concatenate([[0], np.cumsum(segs)[:-1]]).astype(np.int64)
    sel, ch, rid = [], [], []
    for c in (0, 1):
        for k in range(K):
            s_ = starts[k] + np.arange(segs[k] - 1, dtype=np.int64)
            sel.append(s_); ch.append(np.full(len(s_), c, np.int64)); rid.append(np.full(len(s_), k, np.int64))
    sel, ch, rid = np.concatenate(sel), np.concatenate(ch), np.concatenate(rid)
    y = ids[ch, sel + 1]
    word, pad, nw, eob = y < CTRL_LO, y == PAD_ID, y == NW_ID, y == EOB_ID
    assert bool((word | pad | nw).all()) and not eob.any(), ("a scored text-duplex target is not word / PAD / NEW_WORD", np.unique(y[~(word | pad | nw)])[:8])
    n0 = int((ch == 0).sum())
    out = {"ids": ids, "pid": np.concatenate(pid_l), "wpid": np.concatenate(wp_l), "segs": segs, "T": N, "n_conv": K,
           "sel": sel, "ch": ch, "rid": rid, "y": y, "word": word, "pad": pad, "nw": nw, "eob": eob, "n0_sel": n0,
           "n_keep": np.array([[n0, len(sel) - n0]], np.float32), "roles": np.array(roles_l, np.int64), "swap": np.array(swaps, np.int64),
           "frames": np.array(segs, np.int64)}
    for k_ in STATS:
        out[k_] = np.array([[c[k_][0], c[k_][1]] for c in cnts], np.int64)    # [K, data side]
    return out


class TxdMicros:
    """map-style dataset: item i = the micro of pack packs[i] (split 'train' | 'eval'); steps[i] travels along for the trainer's order
    assertion. Swaps: S3_SWAP per conversation (train) or the fixed balanced eval rule (eval_pos = conversation position in the eval list)."""

    def __init__(self, root, split, packs, cfg, swap_mode, steps=None, eval_swap=False, flip=False):
        self.root, self.split, self.packs, self.cfg, self.sm = root, split, [int(p) for p in packs], dict(cfg), swap_mode
        self.steps = list(steps) if steps is not None else [-1] * len(self.packs)
        assert len(self.steps) == len(self.packs)
        self.eval_swap, self.flip = bool(eval_swap), bool(flip)
        self.st = None

    def __len__(self):
        return len(self.packs)

    def __getitem__(self, i):
        import text2_s3q as T2
        if self.st is None:
            self.st = Store(self.root)
        p = self.packs[i]
        ids = self.st.pack(self.split, p)
        recs = [self.st.get(j) for j in ids]
        if self.eval_swap:
            base = int(self.st.eval_ptr[p])
            sw = [((base + k) % 2) ^ int(self.flip) for k in range(len(recs))]
        else:
            sw = [T2.swap_of(self.sm, j, f"txtdup/{r['conv_id']}") ^ int(self.flip) for j, r in zip(ids, recs)]
        out = micro(recs, sw, self.cfg)
        out["pack"], out["step"], out["split"] = p, int(self.steps[i]), self.split
        out["prov"] = json.dumps([{"conv_id": r["conv_id"], "src": int(r["src"]), "store_idx": int(r["store_idx"]), "n_frames": int(r["n_frames"]),
                                   "swap": int(s), "chan_A_role": DS.SIDES[1 if s else 0], "chan_B_role": DS.SIDES[0 if s else 1]} for r, s in zip(recs, sw)])
        return out


def train_pack(step, rank, world, start, offset=0):
    """the train pack of (step, rank): (step - TXTDUP_START) x WORLD + rank + TXTDUP_OFFSET (every pack read at most once in a run)."""
    assert step >= start, (step, start)
    return (int(step) - int(start)) * int(world) + int(rank) + int(offset)


def eval_packs(rank, per_rank):
    return list(range(rank * per_rank, (rank + 1) * per_rank))
