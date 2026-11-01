"""spec item 2: realised voice-clip rate on the v3 items (B's own clip on the seeded 30 % draw), per clip-source rule:
  v2      PPM voice_clips() unchanged on the v3 store (later chunk of the same call in its 2nd half = only the excluded tail beyond 18 min)
  v2_notail  the same rule if the store held only the whole-call rows (no tail rows): 0 by construction (no later chunk exists) - counted
  actor   the alternative (ALOE_V3_CLIP_SRC=actor): same actor's other TRAIN calls (data_aloe_v3.clip_actor)
over every train item (8,067 calls x 2 orientations) and every eval item; also availability over ALL items (drawn or not).
Run on any node: python -I voice_clip_rates.py [workers]"""
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

os.environ.setdefault("S3_MIN_FRAMES", "125")
os.environ.setdefault("PPM_CD", "2")
os.environ["ALOE_STORE_ROOT"] = "/data/home/amartyavarma/ppm/aloe-v3-ft/stores"
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-v3-ft/code")
import data_aloe_v3 as A3  # noqa: E402

_S = {}


def work(job):
    k, rows = job
    if "s" not in _S:
        _S["s"] = A3.Stores(A3.SRC_STORES)
    st = _S["s"]
    cfg = dict(A3.aloe_cfg())
    node, rr, sib = st.open(k)
    sp = st.split_of(k)
    c = Counter()
    for r in rows:
        row = rr[r]
        for sw in (0, 1):
            d = 1 ^ sw
            tag = sp[r]
            drawn = A3.want_voice(row["row_key"], sw, cfg["voice_seed"])
            v2 = A3.clip_v2(row, d, sib, cfg) is not None
            act = A3.clip_actor(row, d, k, r, st, cfg) is not None
            c[f"{tag}/items"] += 1; c[f"{tag}/drawn"] += drawn
            c[f"{tag}/avail_v2"] += v2; c[f"{tag}/avail_actor"] += act
            c[f"{tag}/real_v2"] += drawn and v2; c[f"{tag}/real_actor"] += drawn and act
            c[f"{tag}/long_call"] += int(row["n_frames"]) >= A3.ROW_FRAMES
    return c


def main():
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    import pyarrow.parquet as pq
    jobs = []
    for k in range(4):
        t = pq.read_table(f"{A3.SRC_STORES[k]}/rows.parquet", columns=["row", "chunk_index"]).to_pydict()
        rows = [r for r, ci in zip(t["row"], t["chunk_index"]) if ci == 0]
        jobs += [(k, rows[i:i + 64]) for i in range(0, len(rows), 64)]
    tot = Counter()
    with Pool(W) as p:
        for c in p.imap_unordered(work, jobs):
            tot.update(c)
    out = {}
    for tag in ("train", "eval"):
        n = tot[f"{tag}/items"]
        out[tag] = {"items": n, "drawn": tot[f"{tag}/drawn"], "drawn_frac": tot[f"{tag}/drawn"] / n,
                    "realised_v2": tot[f"{tag}/real_v2"], "realised_v2_frac": tot[f"{tag}/real_v2"] / n,
                    "realised_v2_notail_frac": 0.0,
                    "realised_actor": tot[f"{tag}/real_actor"], "realised_actor_frac": tot[f"{tag}/real_actor"] / n,
                    "avail_v2_frac": tot[f"{tag}/avail_v2"] / n, "avail_actor_frac": tot[f"{tag}/avail_actor"] / n}
    v2m = [json.loads(x) for x in open("/data/home/amartyavarma/ppm/aloe-v2-ft/runs/ppm-aloe-v2-ft-j22383/metrics.jsonl")]
    last = [x for x in v2m if "train/voice_clip_frac_cum" in x][-1]
    out["v2_run_22383"] = {"train_realised_cum": last["train/voice_clip_frac_cum"], "train_drawn_cum": last["train/voice_drawn_frac_cum"]}
    print("JSON " + json.dumps(out))
    tr = out["train"]
    print(f"voice clip realised (train items, of all items; drawn {tr['drawn_frac']:.4f}): v2 rule on v3 rows {tr['realised_v2_frac']:.4f} "
          f"(without the tail rows: 0); actor alternative {tr['realised_actor_frac']:.4f}; v2 run 22383 realised {out['v2_run_22383']['train_realised_cum']:.4f}. "
          f"eval: v2 rule {out['eval']['realised_v2_frac']:.4f}, actor {out['eval']['realised_actor_frac']:.4f}")


if __name__ == "__main__":
    main()
