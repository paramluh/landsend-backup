import json, numpy as np, pyarrow.parquet as pq
d = json.load(open("/data/home/amartyavarma/ppm/aloe-pose-rosie/items.json"))
bad = {k: v for k, v in d["convs"].items() if v["ok_rows"] == 0}
for k in range(8):
    root = f"/data/home/amartyavarma/ppm/aloe-pose-ft/stores/p{k}"
    t = pq.read_table(f"{root}/rows.parquet", columns=["row", "src_conversation", "xbleed", "b_ok", "user_speaker_dir", "assistant_speaker_dir", "src_dir"]).to_pydict()
    for i, c in enumerate(t["src_conversation"]):
        if c in bad:
            side = bad[c]["side"]
            lane = t["assistant_speaker_dir"][i] if side == "assistant" else t["user_speaker_dir"][i]
            st = None
            try:
                st = json.load(open(f"{t['src_dir'][i]}/w00.{lane}.pose.json")).get("status")
            except Exception as e:
                st = repr(e)[:40]
            print(c[:8], "rosie side", side, "lane", lane, "| row b_ok", t["b_ok"][i], "| xbleed bucket", t["xbleed"][i], "| her pose status", st)
