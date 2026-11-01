"""aloe-ft-32-samsep DATA: decode spot-check (no ASR). 3 first-chunk rows (2 train, 1 eval; full 600 s rows) of the NEW stores: both lanes'
new codes and our original codes decoded with transformers MimiModel /data/home/amartyavarma/models/mimi (fp32, as build_rawgain_mimidec.py /
stage_v2vox.py decode), full row in one call. Writes the first 120 s of each decode as 24 kHz int16 wav to <data>/spot/ and per lane:
  env_corr_same / env_corr_other  Pearson corr of the 80 ms log-RMS envelope of the NEW lane decode vs our ORIGINAL same-lane / other-lane decode
  own_minus_other_db              mean frame RMS (dB) in frames where only THIS lane's words are active minus frames where only the OTHER lane's
                                  words are active (store word timings), for the new decode and for our original decode (bleed indicator)
-> <data>/spot/SPOT.json
usage (1 GPU): python -I decode_spot.py"""
import json
import os
import sys
import wave

import numpy as np
import torch

sys.path.insert(0, "/data/home/amartyavarma/qwen-1ch/stage3/data/rows_build/scripts")
import s3rows as F  # noqa: E402

SRC = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1/stores"
OUT = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores"
DATA = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/data"
SPOT = f"{DATA}/spot"
NODES = [f"m1n{k}" for k in range(4)]
SR, HOP, SAVE_S = 24000, 1920, 120


def wav_write(path, x):
    y = np.clip(x, -1.0, 1.0 - 2 ** -15)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((y * 32767).astype("<i2").tobytes())


def env_db(x, n):
    f = x[:n * HOP].reshape(n, HOP).astype(np.float64)
    return 10 * np.log10((f ** 2).mean(1) + 1e-10)


def word_mask(words, n):
    m = np.zeros(n, bool)
    for w in words:
        m[int(w["start_frame"]):max(int(w["end_frame"]), int(w["start_frame"]) + 1)] = True
    return m


def main():
    from transformers import MimiModel
    os.makedirs(SPOT, exist_ok=True)
    z = np.load(f"{DATA}/row_stats.npz")
    rm = json.load(open(f"{DATA}/row_map.json"))["map"]
    first = (z["chunk_index"] == 0) & (z["n"] == 7500)
    picks = []
    for k, split in ((0, "train"), (2, "train"), (1, "eval")):
        idx = np.flatnonzero(first & (z["store"] == k) & (z["split"] == split))
        d = z["diff_assistant"][idx, 0] / z["n"][idx]
        picks.append(int(idx[np.argsort(np.abs(d - np.median(d)))[0]]))   # median CB0 difference among that store's rows
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mimi = MimiModel.from_pretrained("/data/home/amartyavarma/models/mimi").to(dev).eval()
    res = {"decoder": "transformers MimiModel /data/home/amartyavarma/models/mimi fp32", "device": str(dev), "rows": []}
    for i in picks:
        k, nr, orr, n = int(z["store"][i]), int(z["new_row"][i]), int(z["old_row"][i]), int(z["n"][i])
        assert rm[NODES[k]]["new_of_old"][str(orr)] == nr
        new_rows = F.Rows(F.load_node(f"{OUT}/{NODES[k]}")); old_rows = F.Rows(F.load_node(f"{SRC}/{NODES[k]}"))
        rn, ro = new_rows[nr], old_rows[orr]
        assert rn["row_key"] == ro["row_key"] == str(z["row_key"][i])
        tag = rn["row_key"].split("/")[2][:8] + "_" + rn["row_key"].split("/")[-1]
        dec = {}
        for which, r in (("new", rn), ("orig", ro)):
            for s in F.SIDES:
                c = np.asarray(r[f"codes_{s}"])[:, :n].astype(np.int64)
                with torch.no_grad():
                    w = mimi.decode(torch.as_tensor(c, device=dev)[None]).audio_values[0, 0].float().cpu().numpy()[:n * HOP]
                dec[(which, s)] = w
                lane = "lane0_SPEAKER_00" if s == "assistant" else "lane1_SPEAKER_01"
                wav_write(f"{SPOT}/{tag}_{which}_{s}_{lane}.wav", w[:SAVE_S * SR])
        e = {key: env_db(v, n) for key, v in dec.items()}
        wm = {s: word_mask(rn[f"words_{s}"], n) for s in F.SIDES}
        out = {"row_key": rn["row_key"], "store": NODES[k], "new_row": nr, "old_row": orr, "split": str(z["split"][i]), "n_frames": n,
               "cb0_diff_frac": {s: round(float(z[f"diff_{s}"][i, 0] / n), 4) for s in F.SIDES},
               "all_cb_diff_frac": {s: round(float(z[f"diff_{s}"][i].sum() / (32 * n)), 4) for s in F.SIDES}}
        for s in F.SIDES:
            o = "user" if s == "assistant" else "assistant"
            own, oth = wm[s] & ~wm[o], wm[o] & ~wm[s]
            out[s] = {"env_corr_same": round(float(np.corrcoef(e[("new", s)], e[("orig", s)])[0, 1]), 4),
                      "env_corr_other": round(float(np.corrcoef(e[("new", s)], e[("orig", o)])[0, 1]), 4),
                      "own_minus_other_db_new": round(float(e[("new", s)][own].mean() - e[("new", s)][oth].mean()), 2),
                      "own_minus_other_db_orig": round(float(e[("orig", s)][own].mean() - e[("orig", s)][oth].mean()), 2),
                      "frames_own_only": int(own.sum()), "frames_other_only": int(oth.sum()),
                      "rms_db_new": round(float(10 * np.log10((dec[("new", s)] ** 2).mean() + 1e-12)), 2),
                      "peak_new": round(float(np.abs(dec[("new", s)]).max()), 3)}
        res["rows"].append(out)
        print(json.dumps(out), flush=True)
    ok = all(r[s]["env_corr_same"] > r[s]["env_corr_other"] + 0.1 and r[s]["own_minus_other_db_new"] > 6 for r in res["rows"] for s in F.SIDES)
    res["ok"] = bool(ok)
    res["wavs"] = sorted(os.listdir(SPOT))
    json.dump(res, open(f"{SPOT}/SPOT.json", "w"), indent=1)
    print("[spot] ok", ok, flush=True)


if __name__ == "__main__":
    main()
