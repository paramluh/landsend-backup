"""CPU check of the final A path: si20k train micros (corruption + gauss:6 gain in the collator) and, if the podcast FLAC manifest exists,
podcast micros (Voxtral A from the mono FLAC + corruption): shapes, finiteness, vin, A empty, corruption mode, gain, peak.
usage: python -I micro_check.py --stores DIR [--n 4] [--pod-host H]"""
import argparse
import json
import os
import sys
import time

import numpy as np

C = "/data/home/amartyavarma/ppm/si20k-vox-ppm/code"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stores", required=True); ap.add_argument("--n", type=int, default=4); ap.add_argument("--pod-host", default="")
    a = ap.parse_args()
    os.environ.update(ALOE_SI20K_STORE_ROOT=a.stores, ALOE_SI20K_SRC=a.stores, ALOE_SI20K_STORES_JSON=f"{a.stores}/STORES.json", PPM_CD="2")
    sys.path.insert(0, C); sys.path.insert(0, f"{C}/lib")
    import data_si20k_vox as A
    import podcast_rows as PR
    cfg = A.aloe_cfg(); print("train cfg a_gain", cfg["a_gain"], "a_corrupt", cfg["a_corrupt"], "| eval", A.eval_cfg()["a_gain"], A.eval_cfg()["a_corrupt"], "| low", (A.eval_low_cfg() or {}).get("a_gain"))
    PL = A.make_plan(64); st = A.Stores()
    for it in PL["items"][:a.n]:
        t = time.time(); m = st.conv(it, cfg, 0); mic = A.as_micro(m)
        w = mic["wav"]; P, n = (int(x) for x in mic["vin"])
        print("si20k", m["row_key"], "T", len(m["pid"]), "P", P, "n", n, "wav", w.shape, w.shape[0] == n * 1920, "finite", bool(np.isfinite(w).all()),
              "peak dBFS", round(float(20 * np.log10(np.abs(w).max() + 1e-12)), 1), "gain", m["a_gain"]["gain_db"], "limited", m["a_gain"]["limited"],
              "corrupt", m["a_gain"]["corrupt"].get("mode"), m["a_gain"]["corrupt"].get("sev"), "A empty", bool((m["ids"][0][P:] == A.D.PAD_ID).all() and not m["present"][0][P:].any()),
              f"{time.time() - t:.1f}s", flush=True)
    if a.pod_host and os.path.exists(PR.FLAC_MANIFEST):
        pl = PR.load_plan("/data/home/amartyavarma/ppm/si20k-vox-ppm/plan/PODCAST_PLAN_ppm.json"); tr, ev = PR.cfgs(pl)
        for j in range(2):
            src = PR.Source(pl, a.pod_host, j, tr)
            t = time.time(); m = src.micro(src.codes[0]); mic = PR.as_micro(m, src.codes[0]); w = mic["wav"]; P, n = (int(x) for x in mic["vin"])
            print("podcast", json.loads(m["prov"])["row_key"], "T", len(m["pid"]), "P", P, "n", n, "wav", w.shape, w.shape[0] == n * 1920, "finite", bool(np.isfinite(w).all()),
                  "A lane PAD everywhere", bool((m["ids"][0] == A.D.PAD_ID).all()), "A present in conv", float(m["present"][0][P:].sum()),
                  "corrupt", m["aref"].get("corrupt"), "light", round(len(m["light"]) / max(int((m["cb0"] >= 0).sum()), 1), 3), f"{time.time() - t:.1f}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
