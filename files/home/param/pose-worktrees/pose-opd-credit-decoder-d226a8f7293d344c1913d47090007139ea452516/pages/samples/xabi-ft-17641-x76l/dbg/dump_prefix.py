import json, pickle, numpy as np, sys
from safetensors.numpy import load_file
R = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20/rows"
PR = "/data/home/amartyavarma/video/demo-data/prompts"
req = json.load(open("/data/home/amartyavarma/ppm/ppm-pose-server/requests/9157be8ce90e/request.json"))
print("request", {k: req[k] for k in ("clip", "voice_clip", "persona", "prefill_s", "gen_s", "n_samples", "seed")})
OFF = np.arange(32) * 2048
for idn in ["10_27286bdd", "06_74807fe0", "19_47473a46", "03_b437950e", "05_e25c6859", "09_d9ef3c94", "12_72cb45d3", "13_956c09af", "15_d1e1322c", "18_1ffa9083"]:
    x = pickle.load(open(f"{R}/{idn}.pkl", "rb")); j = json.load(open(f"{R}/{idn}.json")); P = int(x["P"])
    pv = json.loads(x["prov"])
    cA = np.flatnonzero(x["depth"][0, :P, 0] >= 0); cB = np.flatnonzero(x["depth"][1, :P, 0] >= 0)
    # contiguous runs of B clip positions
    runs = np.split(cB, np.flatnonzero(np.diff(cB) != 1) + 1) if len(cB) else []
    clipc = (x["depth"][1, cB] - OFF)                     # [L, 32]
    gm = load_file(f"{PR}/{idn}/gt/mimi_local.safetensors")["codes"].astype(np.int64)   # [32, F] package GT (12 s)
    pf = gm[:, :25].T
    # any window of the clip equal (cb0, >= 80%) to any window of the package GT audio
    best = 0.0
    W = 10
    for a in range(0, len(clipc) - W + 1):
        for b in range(0, gm.shape[1] - W + 1):
            best = max(best, float((clipc[a:a + W, 0] == gm[0, b:b + W]).mean()))
    if idn == "10_27286bdd":
        print("PREFIX P", P, "ids A uniq", np.unique(x["ids"][0, :P]).tolist()[:8], "present A", float(x["present"][0, :P].sum()), "present B", float(x["present"][1, :P].sum()))
        print("ids B[:P]", x["ids"][1, :P].tolist())
        print("B code positions", [(int(r[0]), int(r[-1]) + 1) for r in runs], "A code positions", len(cA))
        print("prov", {k: v for k, v in pv.items() if k != "clips"}, "clips", [{k: v for k, v in c.items() if k in ("src", "src_key", "src_chunk", "w0", "L", "side", "n_chunks", "blank_lane")} for c in pv.get("clips", [])])
    c = pv["clips"][0]
    print(idn, "row", x["row_key"].split("/")[-1], "| clip src", c.get("src"), c.get("src_key", "").split("/")[-1], "w0", c.get("w0"), "L", c.get("L"),
          "| B clip runs", [(int(r[0]), int(r[-1]) + 1) for r in runs], "A codes", len(cA), "| clip vs package GT audio best 10-frame cb0 match", round(best, 2))
