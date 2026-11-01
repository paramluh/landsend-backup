"""8,1,1 schedule helpers (code-811, 10-20 PT): pure functions (numpy only) used by train_ppm.py under PPM_S811=1; CPU unit tests in
test811/test_s811_cpu.py. Doc: train_ppm.py "8,1,1 SCHEDULE" and README_811.md."""
import numpy as np

KINDS = ("pose", "speech", "text")


def expand_seq(s):
    """'pose*8,speech,text' (also 'pose×8') -> ['pose'] * 8 + ['speech', 'text']; a plain comma list is returned as is."""
    out = []
    for tok in [x.strip() for x in str(s).split(",") if x.strip()]:
        tok = tok.replace("×", "*")
        name, k = (tok.split("*") + ["1"])[:2]
        assert name and k.strip().isdigit() and int(k) >= 1, ("bad schedule token", tok)
        out += [name.strip()] * int(k)
    return out


def coefs(seq, modes, bb_w, tw_w, norm="task", sp_rule="all"):
    """per micro i of seq -> dict(kind, mode, sp (speech tower + speech decoder run), c_bb, c_s, c_p).
    c_bb: weight of the micro's loss in the BACKBONE gradient (0 for frozen micros); c_s: in the speech-tower side (tower_s + the speech
    decoder's input gradient); c_p: in the pose-tower side (tower_p + decp_bb + the pose decoder's input gradient).
    norm 'task' (default): c = w[kind] / (number of micros of that kind feeding that part) / Z, Z = sum of the BACKBONE task weights over the
      kinds present in seq (one normaliser for every part): each task's weight is its share of the part's gradient, split evenly over the
      task's micros. With bb_w = (pose 1, speech 2, text 1), tw_w = (pose 1, speech 2) every part gets exactly PPM's per-task coefficients
      (PPM: each of pose, speech, speech, text / 4).
    norm 'micro': c = w[kind] / (sum of w[kind(m)] over the micros m feeding that part) (per-micro weights normalised per part).
    sp_rule 'all': the speech tower + speech decoder train on every pose micro (PPM's per-micro maths); 'full': only on backbone-full pose
      micros (frozen pose micros run the pose tower + pose decoder only)."""
    assert len(seq) == len(modes) and all(k in KINDS for k in seq) and all(m in ("full", "frozen") for m in modes), (seq, modes)
    assert norm in ("task", "micro") and sp_rule in ("all", "full"), (norm, sp_rule)
    assert all(m == "full" for k, m in zip(seq, modes) if k == "text"), "text micros train only the backbone: they must be full"
    assert any(m == "full" for m in modes), "no backbone-full micro"
    sp = [k == "speech" or (k == "pose" and (sp_rule == "all" or m == "full")) for k, m in zip(seq, modes)]
    feeds = {"bb": [m == "full" for m in modes], "s": [k in ("pose", "speech") and s_ for k, s_ in zip(seq, sp)], "p": [k == "pose" for k in seq]}
    W = {"bb": bb_w, "s": tw_w, "p": tw_w}
    for k in set(seq):
        if any(f and k_ == k for f, k_ in zip(feeds["bb"], seq)):
            assert bb_w[k] > 0, ("backbone weight 0 for a task with a full micro: make its micros frozen instead", k)
        if k in ("pose", "speech"):
            assert tw_w[k] > 0, ("tower weight must be > 0", k)
    Z = sum(bb_w[k] for k in KINDS if k in seq)
    out = []
    for i, (k, m) in enumerate(zip(seq, modes)):
        c = {}
        for part in ("bb", "s", "p"):
            f = feeds[part]
            if not f[i]:
                c[part] = 0.0
            elif norm == "task":
                c[part] = W[part][k] / sum(1 for j in range(len(seq)) if f[j] and seq[j] == k) / Z
            else:
                c[part] = W[part][k] / sum(W[part][seq[j]] for j in range(len(seq)) if f[j])
        out.append({"kind": k, "mode": m, "sp": bool(sp[i]), "c_bb": c["bb"], "c_s": c["s"], "c_p": c["p"]})
    return out


def pose_stream(r1, consumed, epoch_rows, n_need):
    """THIS rank's pose rows from the first 8,1,1 step on: the rest of the PPM epoch-1 order r1 (rows [consumed, len(r1)); none if PPM had
    already consumed the whole epoch), then epochs 2, 3, ... = epoch_rows(e) (a fresh node permutation each; the caller seeds them), until
    n_need rows. -> (int64 rows [n_need], [(epoch, first index in the stream)])."""
    r1 = np.asarray(r1, np.int64)
    parts, marks, n, e = [], [], 0, 2
    if consumed < len(r1):
        parts.append(r1[consumed:]); marks.append((1, 0)); n = len(r1) - consumed
    while n < n_need:
        r = np.asarray(epoch_rows(e), np.int64)
        assert len(r) > 0, ("empty pose epoch", e)
        marks.append((e, n)); parts.append(r); n += len(r); e += 1
    return (np.concatenate(parts)[:n_need] if parts else np.zeros(0, np.int64)), marks


def speech_stream(rows, consumed, epoch_set, seed, rank, n_need):
    """THIS rank's speech rows from the first 8,1,1 step on: the PPM stream rows[consumed:] (the rest of the D2 epoch, in PPM's order),
    then epochs 2, 3, ... = epoch_set (the epoch's training rows) permuted with SeedSequence([seed + e - 2, rank, 0x5E2]) until n_need rows.
    -> (int64 rows [n_need], [(epoch, first index in the stream)])."""
    rows, epoch_set = np.asarray(rows, np.int64), np.asarray(epoch_set, np.int64)
    parts, marks, n, e = [], [], 0, 2
    if consumed < len(rows):
        parts.append(rows[consumed:]); marks.append((1, 0)); n = len(rows) - consumed
    while n < n_need:
        assert len(epoch_set) > 0, "empty speech epoch set"
        rng = np.random.default_rng(np.random.SeedSequence([int(seed) + e - 2, int(rank), 0x5E2]))
        marks.append((e, n)); parts.append(epoch_set[rng.permutation(len(epoch_set))]); n += len(epoch_set); e += 1
    return (np.concatenate(parts)[:n_need] if parts else np.zeros(0, np.int64)), marks


def seq_at(step, from_step, prev_seq, seq):
    """the GA list optimizer step `step` used: PPM's before the switch, the 8,1,1 list from it on."""
    return list(prev_seq) if from_step is not None and step < from_step else list(seq)


def consumed_rows(kind, steps, from_step, prev_seq, seq):
    """rows of `kind` consumed by optimizer steps [0, steps) (one row per micro)."""
    if from_step is None or steps <= from_step:
        return steps * list(prev_seq if from_step is not None else seq).count(kind)
    return from_step * list(prev_seq).count(kind) + (steps - from_step) * list(seq).count(kind)
