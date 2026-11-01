"""Audio samples for the top actors: each sample = 10 s of the actor's OWN lane in a train row, decoded from the row's SAM-isolated Mimi
codes (the sam-sep store codes the FT trains on) with transformers MimiModel /data/home/amartyavarma/models/mimi (fp32, CPU). The window
(actor_stats.json samples: most own speech, least partner speech) is decoded with 2 s of left context that is trimmed off.
usage: python -I decode_samples.py N_ACTORS N_PER -> actors/audio/a<rank>_<j>.mp3 (wav fallback) + actors/audio/samples.json"""
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/data/home/amartyavarma/qwen-1ch/stage3/data/rows_build/scripts")
import s3rows as F  # noqa: E402

STORES = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores"
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors"
SR, HOP, WIN, CTX = 24000, 1920, 125, 25


def main():
    na, nper = int(sys.argv[1]), int(sys.argv[2])
    torch.set_num_threads(4)
    os.makedirs(f"{D}/audio", exist_ok=True)
    from transformers import MimiModel
    mimi = MimiModel.from_pretrained("/data/home/amartyavarma/models/mimi").eval()
    st = json.load(open(f"{D}/actor_stats.json"))
    stores = {}
    out = []
    for rank, a in enumerate(st["actors"][:na], 1):
        for j, s in enumerate(a["samples"][:nper]):
            k = s["store"]
            if k not in stores:
                stores[k] = F.Rows(F.load_node(f"{STORES}/m1n{k}"))
            r = stores[k][s["row"]]
            assert r["row_key"] == s["row_key"]
            c = np.asarray(r[f"codes_{s['side']}"]).astype(np.int64)
            lo = max(0, s["f0"] - CTX)
            with torch.no_grad():
                w = mimi.decode(torch.as_tensor(c[:, lo:s["f0"] + WIN])[None]).audio_values[0, 0].float().numpy()
            w = np.clip(w[(s["f0"] - lo) * HOP:(s["f0"] - lo + WIN) * HOP], -1, 1)
            name = f"a{rank:02d}_{j}"
            try:
                sf.write(f"{D}/audio/{name}.mp3", w, SR, format="MP3")
                fn = f"{name}.mp3"
            except Exception:
                sf.write(f"{D}/audio/{name}.wav", w, SR, subtype="PCM_16")
                fn = f"{name}.wav"
            out.append({"rank": rank, "actor_id": a["actor_id"], "j": j, "file": fn, "row_key": s["row_key"], "side": s["side"],
                        "lane": s["lane"], "start_s": round(s["f0"] / 12.5, 1), "own_frac": round(s["own"] / WIN, 2),
                        "partner_frac": round(s["oth"] / WIN, 2)})
            print(name, fn, s["row_key"], s["side"], flush=True)
            json.dump(out, open(f"{D}/audio/samples.json", "w"), indent=0)


if __name__ == "__main__":
    main()
