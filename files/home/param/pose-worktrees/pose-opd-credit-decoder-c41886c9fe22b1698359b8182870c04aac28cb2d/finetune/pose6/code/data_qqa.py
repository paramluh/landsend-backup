"""qqa-ft DATA: run C's quick-QA conversations (same store, same conversations per step, same order; qqa-ft/make_plan.py) in PPM's speech-row
format, PACKED with one attention segment per conversation. CPU only (DataLoader workers, CPU tests).

CONVERSATION k = data_ppm.speech_micro(row, node_row, sib, pers, cfg, rank, swap=0) on the run-C store row (s3rows format), i.e. PPM's own
  speech-row maths unchanged: model channel A = the data 'user' (its word lane + Mimi codes at PPM's code delay CD = 2, conditioning only,
  no loss), channel B = the data 'assistant' (word lane with NEW_WORD + Mimi codes at CD 2; B text CE, cb0 labels, 0.6B light frames =
  every coded conversation frame, PPM_LIGHT_FRAC 1.0), PPM's system-prompt rule (data_ppm.podcast_prefix: voice clips of both speakers
  from later chunks in the recording's second half on 1/3 of rows, persona-sidecar text blocks; quick-qa has neither -> the EMPTY
  system block, P = 5). swap = 0 for every conversation (B = the assistant, as run C's S3_SWAP none; PPM's random side not used).
  cfg crop_conv False: the conversation is never padded to 7,500 frames: segment length T_k = P_k + n_k + CD.
PACKED MICRO = the conversations of one plan step concatenated (offsets o_k): segs [T_1..T_K] (model_ppm.Ctx -> FA3 varlen block-diagonal
  attention; the towers' one-frame shifts restart at every segment), pid = 0..T_k - 1 per conversation (PPM's plain RoPE on the frame index,
  restarting per conversation), sel / light offset by o_k, cb0 concatenated. Nothing crosses a conversation boundary (the CD tail of
  conversation k stays inside its segment). n_text = the micro's scored B positions (the trainer replaces the normalisers by the STEP's
  token totals / WORLD = run C's S4_PACK_NORM tokens_step). own [K, 3] = per conversation (text targets, cb0 labels, light frames).
Item 0 of a rank's dataset also carries 'singles' = the K standalone conversation micros (the step-0 packed-loss check)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import data_ppm as D  # noqa: E402

BUDGET = 7600                                    # guard only. Run C packed to 7,503 positions with its layout (P = 0 on 2/3 of conversations,
#                                                 CD 3); PPM's layout (empty system block P = 5 everywhere, CD 2) is +4 / conversation on those, so
#                                                 the SAME packs reach 7,560 at most (2 of 5,352 micros > 7,503; tests/pack_lengths.py)


def qqa_cfg(**kw):
    """PPM prod's data cfg (prod_ppm.env / train_ppm CFG['data']) with crop_conv False (no 7,500-frame padding of short conversations)."""
    return D.default_cfg(**dict({"seed": 20261011, "sp_cap": 2000, "persona_sides": "both", "light_frac": 1.0, "pose_light_frac": 1.0,
                                 "txt_cap": 7500, "crop_conv": False, "chat_side": "assistant", "chat_eob": True, "pack_attn": "segments"}, **kw))


class Store:
    """the run-C node-local store, opened lazily (per DataLoader worker)."""

    def __init__(self, root):
        self.root, self.node = root, None

    def open(self):
        if self.node is None:
            self.node = D.R3.load_node(self.root)
            self.st = D.R3.Rows(self.node)
            self.sib = D.SPR.SibIndex(self.node, self.st)
            try:
                self.ps = D.PER.PersonaSidecar(self.node)
            except (FileNotFoundError, OSError):
                self.ps = None
        return self

    def conv(self, r, cfg, rank):
        """node row r -> PPM speech micro (one conversation, swap 0, own segment)."""
        self.open()
        row = self.st[int(r)]
        D.R3.check_row(row)
        pers = None
        if self.ps is not None:
            try:
                pers = self.ps.get(int(r), row["row_key"])
            except (FileNotFoundError, OSError):
                pers = None
        m = D.speech_micro(row, int(r), self.sib, pers, cfg, rank, swap=0)
        T = len(m["pid"])
        assert m["segs"] == [T] and np.array_equal(m["pid"], np.arange(T)) and m["swap"] == 0
        assert T == int(m["P"]) + int(row["n_frames"]) + D.CD, ("segment length", T, m["P"], row["n_frames"])
        m["row_key"] = row["row_key"]
        return m


def own_counts(m):
    return [float(len(m["sel"])), float((np.asarray(m["cb0"]) >= 0).sum()), float(len(m["light"]))]


def pack(ms, budget=BUDGET):
    """standalone conversation micros -> one packed micro (module doc)."""
    T = [len(m["pid"]) for m in ms]
    o = np.concatenate([[0], np.cumsum(T)[:-1]]).astype(np.int64)
    N = int(sum(T))
    assert N <= budget, ("packed micro over the budget", N, budget, T)
    out = {"kind": "speech", "ids": np.concatenate([m["ids"] for m in ms], 1), "depth": np.concatenate([m["depth"] for m in ms], 1),
           "present": np.concatenate([m["present"] for m in ms], 1), "pid": np.concatenate([np.arange(t, dtype=np.int64) for t in T]),
           "segs": [int(t) for t in T]}
    out["sel"] = np.concatenate([m["sel"] + ok for m, ok in zip(ms, o)])
    for k in ("y", "word", "pad", "eob", "nw"):
        out[k] = np.concatenate([m[k] for m in ms])
    out["cb0"] = np.concatenate([m["cb0"] for m in ms])
    out["light"] = np.concatenate([m["light"] + ok for m, ok in zip(ms, o)])
    out["n_text"] = float(len(out["sel"]))
    out["own"] = np.asarray([own_counts(m) for m in ms], np.float64)
    out["P"] = np.asarray([int(m["P"]) for m in ms], np.int64)
    out["prov"] = json.dumps([m["row_key"] for m in ms])
    # every target / light frame inside its own conversation's segment; no code of conversation k at another segment's positions
    for k, (m, ok, t) in enumerate(zip(ms, o, T)):
        assert (m["sel"] < t - 1).all() and (m["light"] < t).all() and (m["light"] >= int(m["P"]) + D.CD).all()
    assert out["ids"].shape == (2, N) and out["depth"].shape == (2, N, D.DEPTH) and len(out["cb0"]) == N
    return out


class PackDS:
    """one rank's training stream: item s = the packed micro of plan step start + s (conversations = node rows, run C's order)."""

    def __init__(self, root, steps_rows, cfg, rank, start=0, singles_at=0):
        self.store, self.rows, self.cfg, self.rank, self.start, self.singles_at = Store(root), steps_rows, dict(cfg), int(rank), int(start), singles_at

    def __len__(self):
        return len(self.rows) - self.start

    def __getitem__(self, i):
        s = self.start + int(i)
        ms = [self.store.conv(r, self.cfg, self.rank) for r in self.rows[s]]
        out = pack(ms)
        out["step"] = s
        if s == self.singles_at:
            out["singles"] = ms
        return out


def eval_micros(root, rows, cfg, rank):
    """run C's eval rows of this rank: each conversation its own (unpacked) micro."""
    st = Store(root)
    return [st.conv(r, cfg, rank) for r in rows]
