"""PPM CHAT TEXT ROWS (user decision 10-11): half of the text micros are corpus15m chat conversations (the text-KL corpus: split-2 render
table qwen-1ch-data/split2 = corpus15m klpre-split2; split 3 via PPM_CHAT_SPLIT=3), rendered EXACTLY as stage 3's two-track text rows
(stage3/text2_s3q.two_track) but trained with pure cross-entropy on channel B (no teacher cache). CPU only (DataLoader workers).

RENDER (one conversation j): the stage-1/2 renderer, imported read-only from qwen-1ch (data_q.render_raw = render_q.build_row_q with
  ROW_SEED + j: the same frames as every earlier use of the table) -> sys_ids [P], a_ids / b_ids [F] (user / assistant lanes on a common
  frame timeline). Two-track row of length P + 1 + F (text2_s3q's layout: system text at 0..P-1, the 1ch PREPEND frame at P, frame k of
  the conversation at P + 1 + k):
    ids[user ch,  P + 1 + k] = a_ids[k], ids[asst ch, P + 1 + k] = b_ids[k];  every turn token (SOA / EOA / SOB / <|im_start|> /
    <|im_end|>) -> PAD (PPM_CHAT_EOB=1 keeps the assistant's <|im_end|>, = stage 3's S3_TXT_EOB_KL=1 variant);
    system text (corpus sys_ids, ~3% of rows) on channel B only at 0..P-1 (channel A present 0 there), no loss there.
  No NEW_WORD (the render has none: table key nw 0), no audio.
CHANNELS: the ASSISTANT is model channel B by default (PPM_CHAT_SIDE=assistant); PPM_CHAT_SIDE=random draws the side per conversation
  (data_ppm.side_of, as the Gemma duplex rows).
TARGETS (channel B only, D18): every conversation position p in [P, L - 2] predicts ids[B, p + 1] (word / PAD / <|im_end|> if kept).
PACKS: conversations of a seeded permutation of the split's train pool (held-out pool for eval) packed greedily into rows of <= CAP
  (7,500) frames, each conversation its own attention segment (pid restarts); a conversation longer than CAP is cropped to CAP (its
  head). The plan uses the table's lengths only (F = col 2, P = col 3): no rendering at plan time."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Q1 = "/data/home/amartyavarma/qwen-1ch"
PAD, EOB, CTRL_LO = 151670, 151645, 151643
TURN_IDS = (151673, 151674, 151675, 151645, 151644)          # SOA, EOA, SOB, EOB = <|im_end|>, <|im_start|> (text2_s3q.TURN_IDS)
PREPEND = 1


def split_paths(n):
    return {"arrow": f"/data/home/amartyavarma/data/stage2/corpus15m/arrow/klpre-split{n}", "root": f"/data/home/amartyavarma/qwen-1ch-data/split{n}"}


def table(n):
    sp = split_paths(n)
    s = json.load(open(f"{sp['root']}/summary.json"))
    tb = np.load(s["table"], mmap_mode="r")
    return tb, np.load(f"{sp['root']}/train_pool.npy"), np.load(f"{sp['root']}/heldout.npy")


def pack_plan(rows, tb, cap, seed, shuffle=True):
    """-> list of packs (lists of corpus row ids); lengths from the table (P + 1 + F, cropped to cap)."""
    rows = np.asarray(rows, np.int64)
    if shuffle:
        rows = rows[np.random.default_rng(np.random.SeedSequence([int(seed), 0xC4A7])).permutation(len(rows))]
    L = np.minimum(tb[rows, 3].astype(np.int64) + PREPEND + tb[rows, 2].astype(np.int64), int(cap))
    packs, cur, used = [], [], 0
    for j, l in zip(rows.tolist(), L.tolist()):
        if cur and used + l > cap:
            packs.append(cur); cur, used = [], 0
        cur.append(j); used += l
    if cur:
        packs.append(cur)
    return packs


class Renderer:
    """lazy (per DataLoader worker): corpus arrow + tokenizer + render cfg of the qwen-1ch stage-1/2 renderer."""

    def __init__(self, n):
        for p in (Q1, f"{Q1}/stage2"):
            if p not in sys.path:
                sys.path.append(p)
        from datasets import load_from_disk
        import data_q as Dq
        import render_q as RM
        self.Dq, self.RM = Dq, RM
        self.ds = load_from_disk(split_paths(n)["arrow"])
        self.tok, self.cfg = RM.load_tok(), RM.cfg_from({})

    def __call__(self, j):
        return self.Dq.render_raw(self.ds, int(j), self.cfg, self.tok)


def conv_lanes(row, keep_eob=False):
    """rendered conversation -> (data-role lanes [2, L] (0 user, 1 assistant), P, present [2, L])."""
    sysi = np.asarray(row["sys_ids"], np.int64)
    a, b = np.asarray(row["a_ids"], np.int64), np.asarray(row["b_ids"], np.int64)
    P, F = len(sysi), len(a)
    assert len(b) == F
    L = P + PREPEND + F
    lanes = np.full((2, L), PAD, np.int64)
    lanes[0, P + PREPEND:], lanes[1, P + PREPEND:] = a, b
    rm = tuple(t for t in TURN_IDS if not (keep_eob and t == EOB))
    lanes[np.isin(lanes, rm)] = PAD
    assert not (lanes[0] == EOB).any(), "EOT on the user lane"
    assert ((lanes < CTRL_LO) | (lanes == PAD) | ((lanes == EOB) if keep_eob else False)).all(), "a non-word, non-PAD token in a chat row"
    assert not ((lanes[0] != PAD) & (lanes[1] != PAD)).any(), "both roles on one frame"
    assert not np.isin(sysi, TURN_IDS).any()
    lanes[1, :P] = sysi                                          # system text: channel B's lane (data role 1 maps to B only if swap 0;
    pres = np.ones((2, L), np.float32)                           #   chat_micro re-places it on MODEL channel B regardless of the side)
    pres[0, :P] = 0.0
    return lanes, P, pres


def chat_micro(rows, conv_ids, cap, side_mode="assistant", seed=0, keep_eob=False):
    """rendered conversations of one pack -> data_ppm's common micro (kind 'text', src 'chat')."""
    ids_l, pres_l, pid_l, segs, sel_l, prov = [], [], [], [], [], []
    off = 0
    for row, cid in zip(rows, conv_ids):
        lanes, P, pres = conv_lanes(row, keep_eob)
        if side_mode == "random":
            import data_ppm as D
            sw = D.side_of(f"chat/{cid}", seed)
        else:
            assert side_mode == "assistant", side_mode
            sw = 0
        L = min(lanes.shape[1], int(cap))
        roles = (1, 0) if sw else (0, 1)                         # model channel c holds data role roles[c] (txtdup / text2 rule)
        ids = np.stack([lanes[roles[0]], lanes[roles[1]]])[:, :L].copy()
        if P:                                                    # system text always on MODEL channel B, channel A absent
            sysi = lanes[1, :P].copy()
            ids[0, :min(P, L)] = PAD
            ids[1, :min(P, L)] = sysi[:L]
        assert P < L - 1, ("conversation shorter than its system text after the crop", cid, P, L)
        ids_l.append(ids); pres_l.append(pres[:, :L]); pid_l.append(np.arange(L, dtype=np.int64)); segs.append(L)
        sel_l.append(off + np.arange(P, L - 1, dtype=np.int64))
        prov.append({"row": cid if isinstance(cid, str) else int(cid), "P": int(P), "L": int(L), "cropped": int(lanes.shape[1] > L), "swap": int(sw)})
        off += L
    ids = np.concatenate(ids_l, 1)
    sel = np.concatenate(sel_l)
    y = ids[1, sel + 1]
    word, pad, eob = y < CTRL_LO, y == PAD, y == EOB
    nw = np.zeros(len(y), bool)
    assert (word | pad | eob).all() and (keep_eob or not eob.any())
    assert sum(segs) <= int(cap) or len(segs) == 1
    return {"kind": "text", "src": "chat", "ids": ids, "present": np.concatenate(pres_l, 1), "pid": np.concatenate(pid_l), "segs": segs,
            "sel": sel, "y": y, "word": word, "pad": pad, "eob": eob, "nw": nw, "n_text": float(len(sel)), "prov": json.dumps(prov)}


class ChatDS:
    """map-style: item i = chat_micro of packs[i]; a pack = (split, [corpus row ids]) (one renderer per split, lazily per worker)."""

    def __init__(self, packs, cfg):
        self.packs, self.cfg, self.r = [(int(s), list(p)) for s, p in packs], dict(cfg), {}

    def __len__(self):
        return len(self.packs)

    def __getitem__(self, i):
        sp, ids = self.packs[i]
        if sp not in self.r:
            self.r[sp] = Renderer(sp)
        import data_ppm as D
        m = D.packed(chat_micro([self.r[sp](j) for j in ids], [f"{sp}:{j}" if sp != 2 else j for j in ids], self.cfg["txt_cap"],
                                self.cfg.get("chat_side", "assistant"), self.cfg["seed"], bool(self.cfg.get("chat_eob", True))), self.cfg)
        m["chat_split"] = sp
        return m


def multi_split_packs(splits, cap, seed):
    """train packs of every split in order (each split's pool permuted by its own seeded rng), as (split, rows) pairs."""
    out = []
    for sp in splits:
        tb, pool, _ = table(sp)
        out += [(sp, p) for p in pack_plan(pool, tb, cap, seed + 1000 * sp if sp != 2 else seed)]
    return out
