"""10-27 user ("full finetune [the monoplex pose model] on rosie, 2 epochs"): Rosie's monoplex items in 24272's own stores (aloe-pose-ft
stores p0..p7, read only). A row's lane is Rosie's iff its aloe-handover-full pose.json source_video is actor_<her account>.mp4; the item is
(store k, row r, swap sw, n_frames, row_key) exactly as data_aloe_pose.splits_v2's MONO branch builds it (sw 0 -> B = assistant = b_ok col 1;
sw 1 -> B = user = b_ok col 0), kept only when that B track qualifies (b_ok) and has >= MIN_CLIP frames. Eval = her rows of the 4 calls her
voice FT held out, plus any of her rows in the stores' own eval split. -> items.json"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq

ACC = "xyvR1rjHTZbOvkfV1yDCEFr1dgx2"
HELD = {"09bb73ac-e659-4dce-8866-e4c625ef898e/seg00", "4cc4c9fc-c882-4cef-88aa-3f507c154fc5/seg00",
        "88e8e0f4-1c68-420f-ada8-6c8fef9ec0fb/seg00", "d65a446f-c635-4248-a8d2-451333b7857f/seg00"}
SD = "/data/home/amartyavarma/ppm/aloe-pose-ft/stores"
MIN_CLIP = 2
cache = {}


def is_rosie(src_dir, lane):
    k = (src_dir, lane)
    if k not in cache:
        try:
            cache[k] = ACC in json.load(open(f"{src_dir}/w00.{lane}.pose.json")).get("source_video", "")
        except (OSError, ValueError):
            cache[k] = None
    return cache[k]


out = {"train": [], "eval": [], "convs": {}, "unreadable": 0}
for k in range(8):
    root = f"{SD}/p{k}"
    t = pq.read_table(f"{root}/rows.parquet", columns=["row", "split", "n_frames", "row_key", "src_dir", "src_conversation",
                                                       "user_speaker_dir", "assistant_speaker_dir"]).to_pydict()
    b_ok = np.load(f"{root}/pose_index.npz")["b_ok"]
    jobs = [(t["src_dir"][i], lane) for i in range(len(t["row"])) for lane in (t["user_speaker_dir"][i], t["assistant_speaker_dir"][i])]
    with ThreadPoolExecutor(32) as ex:
        list(ex.map(lambda a: is_rosie(*a), jobs))
    for i, r in enumerate(t["row"]):
        ra, ru = is_rosie(t["src_dir"][i], t["assistant_speaker_dir"][i]), is_rosie(t["src_dir"][i], t["user_speaker_dir"][i])
        if ra is None or ru is None:
            out["unreadable"] += 1
        if not (ra or ru):
            continue
        assert not (ra and ru), ("both lanes Rosie", t["row_key"][i])
        sw = 0 if ra else 1
        conv = t["src_conversation"][i]
        d = out["convs"].setdefault(conv, {"side": "assistant" if ra else "user", "rows": 0, "ok_rows": 0, "frames": 0})
        d["rows"] += 1
        if not b_ok[r, 1 - sw] or int(t["n_frames"][i]) < MIN_CLIP:
            continue
        d["ok_rows"] += 1
        d["frames"] += int(t["n_frames"][i])
        split = "eval" if (conv in HELD or t["split"][i] == "eval") else "train"
        out[split].append([k, int(r), sw, int(t["n_frames"][i]), t["row_key"][i]])
    print(f"p{k}: train {len(out['train'])} eval {len(out['eval'])}", flush=True)
fr = lambda s: sum(x[3] for x in out[s])
out["summary"] = {"account": ACC, "convs": len(out["convs"]), "train_items": len(out["train"]), "eval_items": len(out["eval"]),
                  "train_hours": round(fr("train") / 12.5 / 3600, 2), "eval_hours": round(fr("eval") / 12.5 / 3600, 2),
                  "heldout_convs_found": sorted(c for c in out["convs"] if c in HELD), "unreadable_lanes": out["unreadable"]}
json.dump(out, open("/data/home/amartyavarma/ppm/aloe-pose-rosie/items.json", "w"), indent=1)
print(json.dumps(out["summary"]))
