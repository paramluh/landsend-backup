"""si20k podcast A-side raw FLAC: row -> source mapping (10-19, user: "get the raw flacs for those rows"; keep ONLY the side that goes
through Voxtral = channel A, the partner of the row's B side).

For every (host, local rank j) of the 16 pretraining hosts: the first K unseen codes of that rank's stream (podcast_rows.Source: step t ->
unseen_codes[t]) + its 2 eval codes. code -> (slot, node row) -> rows.parquet of the slice (meta copy on /data; checked against the live
node copy by fetch_host.py before any download) -> row_key, episode, chunk_index, speaker dirs. swap = data_ppm.side_of(row_key, seed):
swap 1 = data 'user' on model channel B -> channel A = data 'assistant'; swap 0 -> channel A = data 'user'.
Source audio: the SAM source-separated per-speaker 10-min chunk FLACs the Mimi codes were encoded from:
  gs://podcast-raw-audio-data-mass/elias-final-test/pretraining-data-subsets/conversational-model/<dataset_folder>/<pod>/<ep>/<speaker_dir>/chunk_KK.flac
Streaming Mimi (mimi-stream-v5) chunk KK = episode samples [KK x 14,400,000, (KK+1) x 14,400,000) at 24 kHz = frames [7,500 KK, 7,500 (KK+1));
the store row KK = frames [7,500 KK, +7,500) (build_rows.py: f0 = k x ROW_FRAMES) -> the row window IS chunk KK (fetch_host.py asserts
14,400,000 samples, 24 kHz, mono, and the chunk index == frame_start / 7,500).
usage: map_rows.py K OUT_DIR"""
import hashlib
import json
import os
import sys

import pyarrow.parquet as pq

PLAN = "/data/home/amartyavarma/ppm/si20k-vox-ppm/plan/PODCAST_PLAN_ppm.json"
NPZ = "/data/home/amartyavarma/ppm/si20k-vox-ppm/plan/PODCAST_PLAN_ppm.npz"
META = "/data/home/amartyavarma/ppm/si20k-vox-ppm/plan/_ppmv1_meta/meta"
BUCKET = "podcast-raw-audio-data-mass"
PREFIX = "elias-final-test/pretraining-data-subsets/conversational-model"
DEST = "/scratch/amartyavarma/si20k-podcast-flac"
SLOT_SHIFT = 40
COLS = ["row", "row_key", "dataset", "dataset_folder", "podcast_id", "episode_id", "chunk_index", "frame_start", "frame_end", "n_frames",
        "user_speaker_dir", "user_speaker", "assistant_speaker_dir", "assistant_speaker"]
# licence (user rule: assert in code): the SAME internal podcast pretraining corpus (sam-700k / sam-intelligent-conversations SAM
# source-separated sets) the model already trained on as Mimi codes; no new external source is introduced. Commercial licence status = that
# of the pretraining podcast rows (docs: "internal"), flagged to the user.
LICENCE = {"source": "internal podcast pretraining corpus (SAM source-separated, gs://podcast-raw-audio-data-mass)", "same_as_pretraining": True,
           "commercial_licence_verified": False}
ALLOWED_FOLDERS = {"sam-700k-24khz-conversational-user-assistant-source-sep-8step-10min",
                   "sam-intelligent-conversations-200khr-24khz-user-assistant-source-sep-8step-10min"}


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def side_of(key, seed):                       # = data_ppm.side_of
    return int(h64(f"ppm-side|{int(seed)}|{key}") & 1)


def main():
    K, out = int(sys.argv[1]), sys.argv[2]
    os.makedirs(out, exist_ok=True)
    import numpy as np
    P = json.load(open(PLAN))
    seed = int(P["cfg"]["seed"])
    assert seed == int(P["speech_seed"]) == 20261011
    Z = np.load(NPZ, allow_pickle=True)
    tabs = {}

    def tab(node, slot_root_kind):
        k = (node, slot_root_kind)
        if k not in tabs:
            p = f"{META}/{node}/s3-rows/rows.parquet" if slot_root_kind == "native" else f"{META}/extra/{node}/rows.parquet"
            d = pq.read_table(p, columns=COLS).to_pydict()
            assert d["row"] == list(range(len(d["row"]))), p
            tabs[k] = (p, d)
        return tabs[k]

    summ = {"K": K, "hosts": {}, "licence": LICENCE}
    allrows = []
    for host in P["prod_hosts"]:
        hp = P["hosts"][host]
        slots = {0: (host, "native", hp["native_root"])}
        for sl in hp["extra_slots"]:
            slots[int(sl["slot"])] = (sl["node"], "extra", sl["root"])
        recs = []
        for j in range(8):
            rk = hp["ranks"][str(j)]
            codes_json = [int(c) for c in rk["unseen_codes"]]
            assert len(codes_json) >= K, (host, j, len(codes_json))
            want = [("train", t, c) for t, c in enumerate(codes_json[:K])] + [("eval", e, int(c)) for e, c in enumerate(rk["eval_codes"])]
            for kind, t, code in want:
                slot, r = code >> SLOT_SHIFT, code & ((1 << SLOT_SHIFT) - 1)
                node, sk, live_root = slots[slot]
                p, d = tab(node, sk)
                row = {c: d[c][r] for c in COLS}
                assert row["n_frames"] == 7500 and row["frame_start"] == 7500 * row["chunk_index"], (host, j, code, row)
                assert row["dataset_folder"] in ALLOWED_FOLDERS, row["dataset_folder"]
                sw = side_of(row["row_key"], seed)
                a_role = "assistant" if sw else "user"
                b_role = "user" if sw else "assistant"
                spk_dir = row[f"{a_role}_speaker_dir"]
                obj = f"{PREFIX}/{row['dataset_folder']}/{row['podcast_id']}/{row['episode_id']}/{spk_dir}/chunk_{int(row['chunk_index']):02d}.flac"
                rel = f"{row['row_key']}.A.flac"
                recs.append({"row_key": row["row_key"], "host": host, "local_rank": j, "prod_rank": int(rk["prod_rank"]), "kind": kind,
                             "stream_pos": t, "code": code, "slot": slot, "slice_node": node, "node_row": r, "live_root": live_root,
                             "meta_rows_parquet": p, "swap": sw, "chan_B_data_side": b_role, "chan_A_data_side": a_role,
                             "A_speaker_dir": spk_dir, "A_speaker": row[f"{a_role}_speaker"],
                             "B_speaker_dir": row[f"{b_role}_speaker_dir"], "B_speaker": row[f"{b_role}_speaker"],
                             "dataset": row["dataset"], "dataset_folder": row["dataset_folder"], "podcast_id": row["podcast_id"],
                             "episode_id": row["episode_id"], "chunk_index": int(row["chunk_index"]), "frame_start": int(row["frame_start"]),
                             "n_frames": int(row["n_frames"]), "gcs": f"gs://{BUCKET}/{obj}", "obj": obj, "path": f"{DEST}/{rel}"})
        # the npz full lists must agree with the json prefix (the trainer reads the json)
        with open(f"{out}/{host}.jsonl", "w") as f:
            for r_ in recs:
                f.write(json.dumps(r_) + "\n")
        uk = {r_["row_key"] for r_ in recs}
        summ["hosts"][host] = {"rows": len(recs), "unique_row_keys": len(uk), "swap1": sum(r_["swap"] for r_ in recs),
                               "slots": {str(s): sum(1 for r_ in recs if r_["slot"] == s) for s in slots}}
        allrows += recs
        print(host, summ["hosts"][host], flush=True)
    keys = [r_["row_key"] for r_ in allrows]
    summ["total_rows"] = len(allrows)
    summ["unique_row_keys"] = len(set(keys))
    summ["dup_row_keys_across_hosts"] = len(keys) - len(set(keys))
    summ["npz_files"] = list(Z.files)[:5]
    json.dump(summ, open(f"{out}/SUMMARY.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "hosts"}))


if __name__ == "__main__":
    main()
