"""aloe-v2-ft DATA (10-16 eve, user spec) = aloe-ft's data_aloe.py (job 21980) with ONLY these changes:
  ROWS      ONLY the FIRST chunk of every call (rows.parquet chunk_index == 0); every later chunk (the rows that carry memory blocks) dropped.
            Both speakers as B (each kept row twice, swap 0 / 1), loss on B only, as 21980. Held-out = 21980's eval rows (split == 'eval')
            that are first chunks, x both orientations.
  SYSPROMPT NO TEXT AT ALL (no persona / starting context / memory for either speaker: persona sidecar not read, persona_sides 'none').
            B's own VOICE CLIP on a fixed, seeded 30 % of (row, orientation) items: h64('aloe-v2-voice|<seed>|<row_key>|<swap>') % 10 < 3
            (seed 20261011). The clip = PPM's voice_clips() rule unchanged (data_ppm.voice_clips: 38..125 frames, from a LATER chunk of the
            same recording in its SECOND half, nearest first, same speaker, sysprompt.window search, lane blanked) for the data speaker on
            model channel B (data track 1 ^ swap). A never gets a clip. Every other item (not drawn, or drawn but no valid clip) gets the
            EMPTY system block (header + footer only, P = 5), exactly data_ppm.podcast_prefix's empty case.
  LIGHT     training cfg light_frac 0.1 (21980: 1.0): PPM's own per-frame rule (data_s3q.row_arrays frame_light: a seeded uniform u per
            conversation frame, seed (light_seed, h64(row_key), data track); frame kept iff u < light_frac) -> a seeded random ~10 % of B's
            coded frames per micro. EVAL micros keep light_frac 1.0 (= 21980's eval definition: <= 2,048 seeded frames of all coded frames).
Everything else (plan seed / order rule, one row per rank per step, dummies on the last step, PPM speech-row maths, CD 2, crop_conv False,
segment / budget asserts) = data_aloe.py."""
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
STORES = [f"/scratch/amartyavarma/aloe-ft/stores/m1n{k}" for k in range(4)]
SRC_STORES = [f"/home/param/websites/pose-duplex-engine/deps/qwen-1ch-data/s2h-m1/stores/m1n{k}" for k in range(4)]
DIGESTS = ["c1c3b2517152741137cc4bb72dd6bdbb", "09bdd43fef84574175caacf4abdddd1e", "ecb1c731e8989b3f2e1b5266b51ea0fd",
           "3b16fb8dd2660a2e8c1fe32c0d361d7b"]
PLAN_SEED = 20261011
VOICE_SEED = 20261011
VOICE_NUM, VOICE_DEN = 3, 10                      # 30 % of (row, orientation) items draw B's own voice clip
BUDGET = 2000 + 7500 + D.CD                       # sp_cap + the longest row + CD (= data_aloe)


def aloe_cfg():
    """training data cfg = 21980's (data_qqa.qqa_cfg) with persona_sides none, light_frac 0.1 and the v2 voice keys (voice_mod unused)."""
    return Q.qqa_cfg(persona_sides="none", light_frac=0.1, sysprompt="v2_voiceB_only", voice_frac=VOICE_NUM / VOICE_DEN, voice_seed=VOICE_SEED)


def eval_cfg():
    """eval micros: = aloe_cfg() but light_frac 1.0 (21980's eval light definition)."""
    return dict(aloe_cfg(), light_frac=1.0)


def splits(roots=STORES):
    """-> {'train': [(k, r)], 'eval': [(k, r)], counts} over FIRST-CHUNK rows only (row == node row asserted)."""
    import pyarrow.parquet as pq
    out = {"train": [], "eval": [], "n_all": {"train": 0, "eval": 0}, "n_first": {"train": 0, "eval": 0}}
    for k, root in enumerate(roots):
        t = pq.read_table(f"{root}/rows.parquet", columns=["row", "split", "n_frames", "chunk_index"]).to_pydict()
        assert t["row"] == list(range(len(t["row"]))), root
        for r, s, ci in zip(t["row"], t["split"], t["chunk_index"]):
            out["n_all"][s] += 1
            if int(ci) == 0:
                out[s].append((k, int(r))); out["n_first"][s] += 1
    return out


def make_plan(world, roots=STORES):
    sp = splits(roots)
    items = [(k, r, s) for (k, r) in sp["train"] for s in (0, 1)]
    perm = np.random.default_rng(PLAN_SEED).permutation(len(items))
    items = [items[i] for i in perm]
    steps = -(-len(items) // world)
    ev = [(k, r, s) for (k, r) in sp["eval"] for s in (0, 1)]
    return {"items": items, "steps": steps, "eval": ev, "n_train_rows": len(sp["train"]), "n_eval_rows": len(sp["eval"]),
            "n_all": sp["n_all"], "n_first": sp["n_first"]}


def rank_items(plan, rank, world):
    """-> (train: [item or None (dummy)] per step, eval: [item] of this rank)."""
    it = plan["items"]
    tr = [it[t * world + rank] if t * world + rank < len(it) else None for t in range(plan["steps"])]
    return tr, plan["eval"][rank::world]


def want_voice(row_key, swap, seed=VOICE_SEED):
    return D.h64(f"aloe-v2-voice|{int(seed)}|{row_key}|{int(swap)}") % VOICE_DEN < VOICE_NUM


def prefix_v2(row, swap, sib, cfg):
    """-> (sysprompt.build_prefix dict, provenance): B's own clip (PPM voice_clips rule) on the drawn 30 %, else the EMPTY block."""
    sw = int(swap)
    wv = want_voice(row["row_key"], sw, cfg["voice_seed"])
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
    if not segs:                                                     # EMPTY system block: header + footer only (podcast_prefix)
        segs = [{"kind": "text", "track": 1, "ids": []}]
    spx = D.SPR.build_prefix(row, segs, {"new_word": cfg["new_word"]})
    assert 5 <= spx["P"] <= int(cfg["sp_cap"]), ("system prompt length", spx["P"])
    assert pv["voice_clip"] or spx["P"] == 5, ("empty system block must be 5 positions", spx["P"])
    assert (spx["ids"][0] == D.PAD_ID).all() and not (np.asarray(spx["depth"])[0] >= 0).any(), "channel A must carry nothing in the prefix"
    pv["P"] = int(spx["P"])
    return spx, pv


def speech_micro_v2(row, node_row, sib, cfg, rank, swap):
    """= data_ppm.speech_micro with prefix_v2 in place of podcast_prefix (no persona); everything else unchanged."""
    sw = int(swap)
    spx, pv = prefix_v2(row, sw, sib, cfg)
    P, n = int(spx["P"]), int(row["n_frames"])
    rf = P + n if not cfg["crop_conv"] else 7500
    dc = {"new_word": cfg["new_word"], "eob_gap_s": cfg["eob_gap_s"], "wpid": cfg["wpid"], "light_seed": cfg["light_seed"], "row_frames": rf,
          "frame_light": True, "no_crop": not cfg["crop_conv"]}
    a = D.DS.row_arrays(row, dc, rank, int(node_row), "ppm", float(cfg["light_frac"]), sp=spx, swap=sw)
    sel, y, word, pad, eob, nw, cb0, light = D._b_only(a)
    T = len(a["pid"])
    out = {"kind": "speech", "ids": a["ids"], "depth": a["depth"], "present": a["present"], "pid": a["pid"], "segs": [T],
           "sel": sel, "y": y, "word": word, "pad": pad, "eob": eob, "nw": nw, "n_text": float(len(sel)), "cb0": cb0, "light": light,
           "P": P, "swap": sw}
    assert (cb0[:P + D.CD] < 0).all() and (light >= P + D.CD).all() and (sel >= P).all()
    assert T == (7500 if cfg["crop_conv"] else P + n) + D.CD, ("row length", T, P, n)
    pv.update(row_key=row["row_key"], swap=sw, chan_B_data_side=("user" if sw else "assistant"), n_frames=n, T=T, n_light=int(len(light)))
    out["prov"] = json.dumps(pv)
    out["voice_clip"] = pv["voice_clip"]
    out["voice_drawn"] = pv["voice_drawn"]
    return out


class Stores:
    """the 4 node-local stores, opened lazily (per DataLoader worker). No persona sidecar (no text in the system prompt)."""

    def __init__(self, roots=STORES):
        self.roots, self.s = list(roots), {}

    def open(self, k):
        if k not in self.s:
            node = D.R3.load_node(self.roots[k])
            assert node["digest"] == DIGESTS[k], ("staged store differs from Model 1's", self.roots[k], node["digest"])
            st = D.R3.Rows(node)
            self.s[k] = (node, st, D.SPR.SibIndex(node, st))
        return self.s[k]

    def conv(self, item, cfg, rank):
        k, r, sw = item
        node, st, sib = self.open(int(k))
        row = st[int(r)]
        D.R3.check_row(row)
        assert int(row["chunk_index"]) == 0, ("v2: first chunks only", row["row_key"], row["chunk_index"])
        m = speech_micro_v2(row, int(r), sib, cfg, rank, int(sw))
        T = len(m["pid"])
        assert m["segs"] == [T] and np.array_equal(m["pid"], np.arange(T)) and m["swap"] == int(sw)
        assert T == int(m["P"]) + int(row["n_frames"]) + D.CD <= BUDGET, ("segment length", T, m["P"], row["n_frames"])
        m["row_key"] = row["row_key"]
        m["item"] = [int(k), int(r), int(sw)]
        return m


def as_micro(m, dummy=False):
    """one conversation -> the trainer's micro (data_qqa.pack of one = its own segment); dummy: zero counts (trainer zeroes its weights)."""
    out = Q.pack([m], budget=BUDGET)
    out["swap"] = int(m["swap"])
    out["item"] = json.dumps(m["item"])
    out["voice_clip"] = int(m["voice_clip"]) if not dummy else 0
    out["voice_drawn"] = int(m["voice_drawn"]) if not dummy else 0
    if dummy:
        out["dummy"] = 1
        out["own"] = np.zeros_like(out["own"])
    return out


class AloeDS:
    """one rank's training stream: item t = the micro of step t (dummy where the plan is exhausted)."""

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
            out["singles"] = [m]
        return out


def eval_micros(items, cfg, rank, roots=STORES):
    st = Stores(roots)
    return [as_micro(st.conv(it, cfg, rank)) for it in items]
