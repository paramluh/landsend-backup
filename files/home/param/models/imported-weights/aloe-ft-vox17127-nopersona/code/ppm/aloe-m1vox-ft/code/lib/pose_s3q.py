"""pose-s3 (S3_POSE=1, 10-09; default off: nothing imports this): CPU-only pieces of the multi-task stage-3 run. No torch.distributed.

NODE MIXER (S3_MIX_CYCLE, default 'kl,conv,pose,pose' = the user's 2:1:1 split, node-granular because the text-KL cache's CP groups span up
  to 8 GPUs of one node; KL_PLAN.json): node k at step t runs task cycle[(t - k) % 4] on ALL 8 local ranks. Every step: 4 KL nodes, 4
  conversation nodes, 8 pose nodes, all in parallel (hera's mix=ranks at node granularity); model_s3q.mix_forward walks the same FSDP units in
  the same order on every rank (dummy streams x 0 for the units a task does not use).
POSE ROWS (qwen-1ch-data/pose-s3/POSE_LAYOUT.md, format s3-pose-rows-v1, node-local $ALOE_SCRATCH/pose-s3/rows): 7,500-frame rows,
  channel B = the TikTok speaker (Mimi + words + pose), channel A ABSENT. A row packs clips as [SLOT 1 | clip n | GAP 4] ... ; every clip's
  span [slot, slot + 1 + n + 4) is its OWN attention segment here (clips of different videos never attend to each other; pid / wpid restart
  at 0 per segment, like separate rows). Per segment (L = n + 5 positions, q = 0..L-1 <-> row frame s0 + q):
    ids_B[q]   = lane_assistant[s0 + q] (PAD / word pieces / NEW_WORD, the store's placed lane)          ids_A = PAD, present_A = 0
    depth_B[q] = Mimi of row frame s0 + q - CD (CD = 3; -1 outside the clip)                              depth_A = -1
    pdepth[q]  = pose of row frame s0 + q - POSE_DELAY (default 3 = the user's final decision: pose cd = speech CD = 3; -1 outside the clip)
  targets: text B at q = 0..L-2 (predicts ids_B[q + 1]: word / PAD / NEW_WORD); speech cb0 where depth_B >= 0, speech light = a seeded
  LIGHT_FRAC subset of those; pose cb0 where pdepth >= 0 (tower position T predicts pose frame T - POSE_DELAY from trunk state T - 1, i.e.
  the trunk has read text <= frame T - 1 and pose <= frame T - 1 - POSE_DELAY; with POSE_DELAY = CD pose frame f and Mimi
  frame f share every position: the same slots, start PAD and tail as the speech code delay), pose light = a seeded POSE_LIGHT_FRAC subset. Channel A: none.
  Normalisers = per ROW (all its segments): text n_keep, cb0 n_cb0, light 31 x n_light, pose cb0 n_pcb0, pose light 31 x n_plight.
DUMMY STREAMS: a tiny fixed tower / pose-tower stream (T = 8 frames, all weights 0) for the units a rank's task does not use."""
import hashlib
import json
import math
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S2_DIR = os.path.join(os.path.dirname(HERE), "stage2")
if S2_DIR not in sys.path:
    sys.path.insert(0, S2_DIR)
import speech_q as SP  # noqa: E402

PAD_ID, EOB_ID, NW_ID, CTRL_LO = SP.PAD_ID, SP.EOB_ID, SP.NW_ID, SP.CTRL_LO
DEPTH, CBS, FPS, CD = SP.DEPTH, SP.CBS, SP.FPS, SP.CD
FORMAT = "s3-pose-rows-v1"
KIND_TAIL, KIND_CLIP, KIND_SLOT, KIND_GAP = 0, 1, 2, 3
SLOT, GAP = 1, 4
TASKS = ("pose", "kl", "conv")
DUMMY_T = 8


# ------------------------------------------------------------------------------------------------ node mixer (rotating, node-granular)
def parse_cycle(spec):
    """S3_MIX_CYCLE 'kl,conv,pose,pose' -> ('kl', 'conv', 'pose', 'pose'). Exactly one 'kl', at index 0 (the text-KL cache rule: node k runs KL
    on all 8 ranks iff k % P == t % P, reading key base + t // P; every (key, slice) unit read once)."""
    c = tuple(x.strip() for x in (spec if isinstance(spec, (list, tuple)) else str(spec).split(",")) if x.strip())
    assert c and c[0] == "kl" and c.count("kl") == 1 and set(c) <= set(TASKS) and "pose" in c and "conv" in c, ("S3_MIX_CYCLE", spec)
    return c


def role_at(step, node_index, cycle):
    """the task of node node_index at optimizer step `step`: cycle[(step - node_index) % P]."""
    return cycle[(int(step) - int(node_index)) % len(cycle)]


def task_count(step, node_index, cycle, task):
    """number of steps s in [0, step) at which node node_index ran `task` (= this node's micro counter for that task at step `step`)."""
    P = len(cycle)
    per = sum(1 for x in cycle if x == task)
    q, r = divmod(int(step), P)
    return q * per + sum(1 for s in range(q * P, q * P + r) if role_at(s, node_index, cycle) == task)


def task_steps_needed(max_steps, node_index, cycle, task):
    return task_count(max_steps, node_index, cycle, task)


def mix_fractions(cycle):
    return {t: cycle.count(t) / len(cycle) for t in TASKS}


# ------------------------------------------------------------------------------------------------ store
def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def load_node(root):
    meta = json.load(open(f"{root}/NODE.json"))
    assert meta.get("format") == FORMAT, (meta.get("format"), root)
    assert int(meta["slot"]) == SLOT and int(meta["gap"]) == GAP and int(meta["depth"]) == DEPTH, ("pose store framing", meta["slot"], meta["gap"])
    shards = list(meta["shards"])
    L, S, R, SPL = [], [], [], []
    for k, s in enumerate(shards):
        d = json.load(open(f"{root}/{s}.DONE"))
        z = np.load(f"{root}/{s}.index.npz")
        ln = z["length"].astype(np.int32)
        assert len(ln) == d["rows"] == meta["rows"][k] and int(ln.sum()) == d["frames"], (s, len(ln), d["rows"])
        L.append(ln); S.append(np.full(len(ln), k, np.int32)); R.append(np.arange(len(ln), dtype=np.int32)); SPL.append(z["split"].astype(np.int8))
    h = hashlib.blake2b(digest_size=16)
    for s, dg in zip(shards, meta["ids_blake2b"]):
        h.update(s.encode()); h.update(dg.encode())
    z = lambda a, dt: np.concatenate(a) if a else np.zeros(0, dt)  # noqa: E731
    return {"shards": shards, "length": z(L, np.int32), "shard_of": z(S, np.int32), "row_of": z(R, np.int32), "split": z(SPL, np.int8),
            "digest": h.hexdigest(), "node": meta["node"], "root": root, "meta": meta, "row_frames": int(meta["row_frames"])}


class PoseRows:
    """lazy per-worker random access: rows[i] -> dict(codes [32, F] int16, pose [32, F] int16, lane [F] int32, kind [F] int8 (mmap views),
    n_frames, row_key, clips (list), provenance)."""

    def __init__(self, node):
        self.node, self.mm, self.tb = node, {}, {}

    def _open(self, k):
        if k not in self.mm:
            import pyarrow as pa
            import pyarrow.ipc as ipc
            s, root = self.node["shards"][k], self.node["root"]
            self.mm[k] = {x: np.load(f"{root}/{s}.{x}.npy", mmap_mode="r") for x in ("codes_assistant", "pose_assistant", "lane_assistant", "frame_kind")}
            self.tb[k] = ipc.open_file(pa.memory_map(f"{root}/{s}.words.arrow", "r")).read_all()

    def __len__(self):
        return len(self.node["length"])

    def __getitem__(self, i):
        k, r = int(self.node["shard_of"][i]), int(self.node["row_of"][i])
        self._open(k)
        rec = self.tb[k].slice(r, 1).to_pylist()[0]
        m = self.mm[k]
        out = {"codes": m["codes_assistant"][r], "pose": m["pose_assistant"][r], "lane": m["lane_assistant"][r], "kind": m["frame_kind"][r],
               "row": int(i), "n_frames": int(rec["n_frames"]), "row_key": rec["row_key"], "clips": rec["clips"], "split": rec["split"],
               "dataset": rec["dataset"], "n_words": len(rec["words_assistant"]), "n_pieces": len(rec["piece_ids_assistant"])}
        assert out["n_frames"] == int(self.node["length"][i])
        return out


def plan(node, seed, n_task, j, rows_per_micro, lw, local, n_eval_per_rank):
    """THIS rank's pose plan. TRAIN: the node's split-0 rows permuted (seed, crc32(node name)), striped over the node's n_task POSE ranks
    (perm[j::n_task]); micro t = rows [t R, (t + 1) R). EVAL (every rank evaluates pose rows, uniform eval walk): the node's split-1 rows
    sorted by hash(row_key order = row index), local rank l takes [l E, (l + 1) E) (wraps if the node has fewer: logged)."""
    tr = np.flatnonzero(node["split"] == 0).astype(np.int64)
    ev = np.flatnonzero(node["split"] == 1).astype(np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(str(node["node"]).encode()), 0x9053]))
    perm = tr[rng.permutation(len(tr))]
    stripe = perm[j::n_task] if n_task > 0 else perm[:0]
    E = int(n_eval_per_rank)
    assert len(ev) >= 1, ("no pose eval rows on this node", node["node"])
    ev_order = ev[np.argsort([h64(f"pose-eval|{node['node']}|{int(i)}") for i in ev], kind="stable")]
    ev_rows = [int(ev_order[(local * E + k) % len(ev_order)]) for k in range(E)]
    h = hashlib.blake2b(digest_size=16)
    h.update(b"pose-s3-v1"); h.update(node["digest"].encode()); h.update(stripe.tobytes()); h.update(np.asarray(ev_rows, np.int64).tobytes())
    return {"rows": stripe, "eval_rows": ev_rows, "digest": h.hexdigest(), "steps": int(len(stripe) // max(int(rows_per_micro), 1)),
            "n_train_node": int(len(tr)), "n_eval_node": int(len(ev)), "eval_wrapped": bool(len(ev_order) < lw * E)}


# ------------------------------------------------------------------------------------------------ frame-0 pose state (Vansh's G slot)
JOINTS, CONFIDENT = 55, 0.2
G_DIM = 2 * JOINTS + JOINTS + 4                                    # 169 = Vansh's global_state input


def global_features(root, width, joints55):
    """Vansh's orpheus/common.global_features for ONE clip (numpy): root [2] (y, x) image-normalised, width (shoulder width), joints55
    [55 x 3] (y, x, confidence) -> f [169] = [a (55 x 2), c (55), e (4)]: a_j = 1[c_j >= 0.2] (p_j - r) / w, e = (-r_y, 1 - r_y, -r_x, 1 - r_x) / w.
    Non-finite / non-positive width -> all zeros (counted by the caller)."""
    r = np.asarray(root, np.float32).reshape(2)
    w = np.float32(width)
    j = np.asarray(joints55, np.float32).reshape(JOINTS, 3)
    if not (np.isfinite(w) and w > 0 and np.isfinite(r).all() and np.isfinite(j).all()):
        return np.zeros(G_DIM, np.float32), False
    p, c = j[:, :2], j[:, 2]
    a = (c >= CONFIDENT)[:, None].astype(np.float32) * (p - r[None]) / w
    e = np.array([-r[0], 1 - r[0], -r[1], 1 - r[1]], np.float32) / w
    return np.concatenate([a.reshape(-1), c, e]).astype(np.float32), True


# ------------------------------------------------------------------------------------------------ row -> segments -> micro
def row_segments(row):
    """[(s0, L)] = each clip's [slot, slot + SLOT + n + GAP) span; asserted contiguous from 0 and covering [0, n_frames)."""
    segs, a = [], 0
    for c in row["clips"]:
        s0, n = int(c["slot"]), int(c["n"])
        assert s0 == a and int(c["start"]) == s0 + SLOT, ("pose row clip framing", row["row_key"], s0, a, c["start"])
        L = SLOT + n + GAP
        segs.append((s0, L)); a = s0 + L
    assert a == int(row["n_frames"]), ("clips do not cover the row", row["row_key"], a, row["n_frames"])
    return segs


def light_pick(pos, frac, seed_words):
    """a seeded subset of the candidate positions (>= 1 if any)."""
    if len(pos) == 0:
        return pos
    u = np.random.default_rng(np.random.SeedSequence([int(x) & 0xFFFFFFFF for x in seed_words])).random(len(pos))
    k = np.flatnonzero(u < frac)
    if not len(k):
        k = np.array([int(np.argmin(u))])
    return pos[k]


def segment_arrays(row, s0, L, pd, cfg):
    """one clip segment -> per-position arrays (module doc)."""
    codes, pose, lane, kind = (np.asarray(row[k]) for k in ("codes", "pose", "lane", "kind"))
    q = np.arange(L, dtype=np.int64)
    ids_b = lane[s0:s0 + L].astype(np.int64)
    assert not np.isin(ids_b, (SP.SOB_ID, EOB_ID)).any() and (kind[s0] == KIND_SLOT) and (kind[s0 + L - GAP:s0 + L] == KIND_GAP).all()
    if not cfg["new_word"]:
        ids_b = np.where(ids_b == NW_ID, PAD_ID, ids_b)
    off = np.arange(DEPTH, dtype=np.int64) * CBS
    depth = np.full((L, DEPTH), -1, np.int64)
    src = s0 + q - CD
    ok = q >= CD
    c = codes[:, src[ok]].T.astype(np.int64)
    depth[ok] = np.where(c >= 0, c + off, -1)
    pdepth = np.full((L, DEPTH), -1, np.int64)
    ok = q >= pd
    p = pose[:, (s0 + q - pd)[ok]].T.astype(np.int64)
    pdepth[ok] = np.where(p >= 0, p + off, -1)
    for d_, nm in ((depth, "mimi"), (pdepth, "pose")):              # a frame's codebooks are all present or all absent (store invariant)
        assert ((d_ >= 0).all(1) | (d_ < 0).all(1)).all(), (f"partial {nm} frame", row["row_key"])
    return ids_b, depth, pdepth


def word_positions_b(ids_b):
    lb = ((ids_b < CTRL_LO) | (ids_b == EOB_ID)).astype(np.int64)
    return np.cumsum(lb).astype(np.int64)


def pose_micro(rows, cfg, rank, light_frac, plight_frac):
    """R pose rows -> ONE merged micro (all their clip segments). cfg: new_word, pose_delay, light_seed."""
    pd = int(cfg["pose_delay"])
    assert 1 <= pd <= GAP, ("POSE_DELAY must be 1..GAP", pd)
    R = len(rows)
    segs, ids_b, dep, pdep, pid, wpid, srow = [], [], [], [], [], [], []
    gfeat, g_ok = [], 0
    for r, row in enumerate(rows):
        for (s0, L), cl in zip(row_segments(row), row["clips"]):
            fj = cl["frame0_joints55"]
            f_, ok_ = global_features(cl["frame0_root"], cl["frame0_width"], json.loads(fj) if isinstance(fj, str) else fj)
            gfeat.append(f_); g_ok += int(ok_)
            a, d, p = segment_arrays(row, s0, L, pd, cfg)
            segs.append(L); ids_b.append(a); dep.append(d); pdep.append(p); pid.append(np.arange(L, dtype=np.int64)); wpid.append(word_positions_b(a))
            srow.append(np.full(L, r, np.int64))
    ids_b, dep, pdep, pid, wpid, srow = (np.concatenate(x) for x in (ids_b, dep, pdep, pid, wpid, srow))
    N = len(ids_b)
    starts = np.cumsum([0] + segs[:-1])
    # text targets: channel B, every position but each segment's last
    last = np.zeros(N, bool); last[starts + np.asarray(segs) - 1] = True
    sel = np.flatnonzero(~last).astype(np.int64)
    y = ids_b[sel + 1]
    word, pad, eob, nw = y < CTRL_LO, y == PAD_ID, y == EOB_ID, y == NW_ID
    assert bool((word | pad | nw).all()) and not eob.any(), ("a pose-row text target is not word / PAD / NEW_WORD", np.unique(y[~(word | pad | nw)])[:8])
    rid = srow[sel]
    cb0 = np.where(dep[:, 0] >= 0, dep[:, 0], -100).astype(np.int64)
    pcb0 = np.where(pdep[:, 0] >= 0, pdep[:, 0] - 0, -100).astype(np.int64)
    light, plight = [], []
    for r, row in enumerate(rows):
        m = srow == r
        light.append(light_pick(np.flatnonzero(m & (dep[:, 0] >= 0)), light_frac, (cfg["light_seed"], h64(row["row_key"]), 1, 0x11)))
        plight.append(light_pick(np.flatnonzero(m & (pdep[:, 0] >= 0)), plight_frac, (cfg["light_seed"], h64(row["row_key"]), 1, 0x9053)))
    light, plight = np.concatenate(light).astype(np.int64), np.concatenate(plight).astype(np.int64)
    nk, n0, nl = np.zeros((R, 2), np.float32), np.zeros((R, 2), np.float32), np.zeros((R, 2), np.float32)
    npc, npl = np.zeros(R, np.float32), np.zeros(R, np.float32)
    for r in range(R):
        nk[r, 1] = (rid == r).sum(); n0[r, 1] = ((srow == r) & (cb0 >= 0)).sum(); nl[r, 1] = (srow[light] == r).sum()
        npc[r] = ((srow == r) & (pcb0 >= 0)).sum(); npl[r] = (srow[plight] == r).sum()
    assert (nk[:, 1] > 0).all() and (n0[:, 1] > 0).all() and (nl[:, 1] > 0).all() and (npc > 0).all() and (npl > 0).all(), "a pose row without targets"
    ids = np.stack([np.full(N, PAD_ID, np.int64), ids_b])
    depth = np.stack([np.full((N, DEPTH), -1, np.int64), dep])
    present = np.stack([np.zeros(N, np.float32), np.ones(N, np.float32)])
    prov = [{"row_key": row["row_key"], "dataset": row["dataset"], "split": row["split"], "n_frames": int(row["n_frames"]), "n_clips": len(row["clips"]),
             "swap": 0, "chan_A_role": "absent", "chan_B_role": "assistant", "videos": sorted({c["video_id"] for c in row["clips"]})} for row in rows]
    gpos = np.asarray(starts, np.int64)                              # each clip's SLOT position = its segment's first position
    out = {"T": -1, "n_rows": R, "segs": segs, "gpos": gpos, "gfeat": np.stack(gfeat).astype(np.float32), "g_ok": g_ok, "ids": ids, "depth": depth, "present": present, "pdepth": pdep, "pid": pid, "wpid": wpid,
           "sel": sel, "ch": np.ones(len(sel), np.int64), "rid": rid, "y": y, "word": word, "pad": pad, "eob": eob, "nw": nw,
           "n_keep": nk, "n_cb0": n0, "n_light": nl, "n0_sel": 0, "roles": np.tile(np.array([[0, 1]], np.int64), (R, 1)), "swap": np.zeros(R, np.int64),
           "cb0": cb0, "cb0_rid": srow, "cb0_ch": np.ones(N, np.int64), "light": light, "light_rid": srow[light], "light_ch": np.ones(len(light), np.int64),
           "pcb0": pcb0, "plight": plight, "plight_rid": srow[plight], "n_pcb0": npc, "n_plight": npl, "srow": srow,
           "used": int(N), "prov": json.dumps(prov)}
    for k, f in (("text_dropped", lambda row: 0), ("n_new_word", lambda row: int((np.asarray(row["lane"]) == NW_ID).sum())),
                 ("n_words", lambda row: row["n_words"]), ("n_pieces", lambda row: row["n_pieces"])):
        out[k] = np.array([[0, f(row)] for row in rows], np.int64)
    return out


class PoseMicros:
    """map-style dataset: item i = pose_micro of the rank's rows [i R, (i + 1) R)."""

    def __init__(self, node, rows, row_ids, cfg, rank, R, light_frac, plight_frac):
        assert len(rows) == len(row_ids) and len(rows) % R == 0, (len(rows), len(row_ids), R)
        self.node, self.rows, self.row_ids, self.cfg, self.rank, self.R = node, list(rows), list(row_ids), dict(cfg), int(rank), int(R)
        self.lf, self.plf, self.st = float(light_frac), float(plight_frac), None

    def __len__(self):
        return len(self.rows) // self.R

    def __getitem__(self, i):
        if self.st is None:
            self.st = PoseRows(self.node)
        rows = [self.st[int(self.rows[i * self.R + k])] for k in range(self.R)]
        out = pose_micro(rows, self.cfg, self.rank, self.lf, self.plf)
        out["row_ids"] = [int(self.row_ids[i * self.R + k]) for k in range(self.R)]
        out["node_rows"] = [int(self.rows[i * self.R + k]) for k in range(self.R)]
        return out


# ------------------------------------------------------------------------------------------------ dummy streams (FSDP symmetry, weight 0)
def dummy_streams():
    """numpy dicts: tw = a 2-channel speech tower stream (segs [8, 8], segs_trunk [8]), ptw = a pose tower stream (segs [8]); every label
    weight 0. Codes are code 0 of every codebook where present (valid light labels), PAD text."""
    T = DUMMY_T
    off = np.arange(DEPTH, dtype=np.int64) * CBS
    d = np.full((T, DEPTH), -1, np.int64); d[CD:] = off
    pdp = np.full((T, DEPTH), -1, np.int64); pdp[1:] = off
    tw = {"ids": np.full(2 * T, PAD_ID, np.int64), "depth": np.concatenate([d, d]), "pid": np.concatenate([np.arange(T), np.arange(T)])[None],
          "segs": [T, T], "segs_trunk": [T], "ch": np.repeat(np.arange(2), T), "cb0": np.full(2 * T, -100, np.int64),
          "cb0_w": np.zeros(2 * T, np.float32), "light": np.array([CD], np.int64), "light_w": np.zeros(1, np.float32)}
    ptw = {"ids": np.full(T, PAD_ID, np.int64), "pdepth": pdp, "pid": np.arange(T)[None], "segs": [T], "cb0": np.full(T, -100, np.int64),
           "cb0_w": np.zeros(T, np.float32), "light": np.array([1], np.int64), "light_w": np.zeros(1, np.float32)}
    return {"T": T, "tw": tw, "ptw": ptw}


def kl_consumer_rank(L, local, lw=8):
    """the text-KL cache consumer rank read by local rank `local` of a host whose logical cache node is L (KL_PLAN: 8 L + lr)."""
    return L * lw + local


def kl_phys_b0(b0_consumer, L, node_index, lw=8):
    """a cached CP group's first consumer rank -> the physical global rank (same node, same local offset)."""
    assert b0_consumer // lw == L, ("a cached CP group outside this host's logical slice", b0_consumer, L)
    return node_index * lw + b0_consumer % lw


def task_loss(role, out, w_task, w_kl, w_cb0, w_cb, w_pcb0, w_pcb):
    """the rank's loss for its task (train_pose_s3q doc): kl -> w_task[kl] x W_KL x KL; conv -> w_task[conv] x (text + W_CB0 cb0 + W_CB light);
    pose -> w_task[pose] x (that + W_POSE_CB0 pose_cb0 + W_POSE_CB pose_light); + the zero-weighted FSDP touch."""
    if role == "kl":
        return w_task["kl"] * w_kl * out["kl"] + out["touch"]
    loss = out["text"] + w_cb0 * out["cb0"] + w_cb * out["light"]
    if role == "pose":
        loss = loss + w_pcb0 * out["pose_cb0"] + w_pcb * out["pose_light"]
    return w_task[role] * loss + out["touch"]


def frac_ok(x):
    return 0.0 < float(x) <= 1.0 and math.isfinite(float(x))
