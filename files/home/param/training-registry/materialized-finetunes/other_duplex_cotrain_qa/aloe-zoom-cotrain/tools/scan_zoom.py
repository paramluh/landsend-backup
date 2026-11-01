"""scan zoom-strangers-5k with the PPM chat renderer (compute node only): every train / eval conversation rendered once (data_zoom.Renderer =
qwen-1ch render_q.build_row_q + check_row_q), its chat lanes checked (chat_ppm.conv_lanes asserts), and the table written:
plan/zoom_table.npz {train [4900, 8], eval [100, 8], meta json}: columns 0 ok, 1 reason, 2 F frames, 3 P prefix, 4 n_b_word (B word frames),
5 n_b_eob, 6 n_b_pad, 7 n_a_word. Then epoch-0 / eval pack plans + every epoch-0 pack built as a trainer micro (asserts) -> plan/zoom_stats.json.
usage: python -I scan_zoom.py <code dir> <plan dir> [workers]"""
import json
import os
import sys
import time
from multiprocessing import Pool

CODE, PLAN = sys.argv[1], sys.argv[2]
NW = int(sys.argv[3]) if len(sys.argv) > 3 else 8
sys.path.insert(0, CODE)
import numpy as np  # noqa: E402
import data_zoom as Z  # noqa: E402

_R = {}


def one(args):
    split, j = args
    if split not in _R:
        _R[split] = Z.Renderer(split)
    r = _R[split]
    out = np.zeros(8, np.int64)
    try:
        row = r(j)
    except r.RM.Unrenderable as e:
        out[1] = 1
        return out, str(e)
    r.RM.check_row_q(row)
    lanes, P, _ = Z.CH.conv_lanes(row, keep_eob=True)
    b, a = np.asarray(row["b_ids"]), np.asarray(row["a_ids"])
    out[:] = [1, 0, len(a), P, int((b < Z.CTRL_LO).sum()), int((b == Z.EOB).sum()), int((b == Z.PAD).sum()), int((a < Z.CTRL_LO).sum())]
    return out, ""


def main():
    os.makedirs(PLAN, exist_ok=True)
    t0 = time.time()
    tb, bad = {}, {}
    with Pool(NW) as pool:
        for split in ("train", "eval"):
            res = pool.map(one, [(split, j) for j in range(Z.N_CONV[split])], chunksize=16)
            tb[split] = np.stack([x for x, _ in res])
            bad[split] = [(j, e) for j, (_, e) in enumerate(res) if e]
    k, kd = Z.cfg_key()
    meta = {"cfg_key": k, "cfg": kd, "src": Z.SRC, "hf_repo": Z.HF_REPO, "time_s": time.time() - t0, "unrenderable": bad,
            "columns": ["ok", "reason", "F", "P", "n_b_word", "n_b_eob", "n_b_pad", "n_a_word"]}
    np.savez(f"{PLAN}/zoom_table.npz", train=tb["train"], eval=tb["eval"], meta=json.dumps(meta, default=str))
    print(f"[scan] rendered {sum(Z.N_CONV.values())} conversations in {time.time() - t0:.0f}s with {NW} workers; unrenderable "
          f"{ {s: len(v) for s, v in bad.items()} }", flush=True)
    st = {"cfg_key": k}
    for s in ("train", "eval"):
        t = tb[s]
        L = t[:, 3] + 1 + t[:, 2]
        st[s] = {"conversations": int(len(t)), "ok": int(t[:, 0].sum()), "frames_total": int(L.sum()), "frames_mean": float(L.mean()),
                 "frames_p50": float(np.median(L)), "frames_max": int(L.max()), "frames_min": int(L.min()), "over_cap": int((L > Z.CAP).sum()),
                 "b_word_frames": int(t[:, 4].sum()), "b_eob": int(t[:, 5].sum()), "b_pad_frames": int(t[:, 6].sum()), "a_word_frames": int(t[:, 7].sum()),
                 "equiv_hours_at_12.5Hz": float(L.sum() / 12.5 / 3600)}
    tbl, _ = Z.load_table(f"{PLAN}/zoom_table.npz")
    p0 = Z.epoch_packs(tbl["train"], 0)
    p1 = Z.epoch_packs(tbl["train"], 1)
    pe = Z.eval_packs(tbl["eval"])
    st["packs"] = {"epoch0": len(p0), "epoch1": len(p1), "eval": len(pe), "convs_per_pack_mean": float(np.mean([len(p) for p in p0])),
                   "pack_frames_mean": float(np.mean([min(sum(int(tbl['train'][j, 2] + tbl['train'][j, 3] + 1) for j in p), Z.CAP) for p in p0])),
                   "epoch0_covers_all": sorted(j for p in p0 for j in p) == list(range(Z.N_CONV["train"]))}
    # build every epoch-0 pack + every eval pack as a trainer micro (all asserts), count B text targets
    rt, re_ = Z.Renderer("train"), Z.Renderer("eval")
    t1 = time.time()
    tg, nw, ne, npd = [], 0, 0, 0
    for p in p0:
        m = Z.as_noop_speech(Z.chat_pack_micro(rt, p))
        tg.append(m["own_zoom"]); nw += int(m["word"].sum()); ne += int(m["eob"].sum()); npd += int(m["pad"].sum())
        assert m["ids"].shape[1] <= Z.CAP
    te = [Z.as_noop_speech(Z.chat_pack_micro(re_, p))["own_zoom"] for p in pe]
    st["epoch0_micros"] = {"text_targets_total": float(sum(tg)), "text_targets_per_micro_mean": float(np.mean(tg)), "word_targets": nw,
                           "eob_targets": ne, "pad_targets": npd, "build_s_per_micro": (time.time() - t1) / len(p0)}
    st["eval_micros"] = {"n": len(pe), "text_targets_total": float(sum(te))}
    json.dump(st, open(f"{PLAN}/zoom_stats.json", "w"), indent=1)
    print(json.dumps(st, indent=1), flush=True)


if __name__ == "__main__":
    main()
