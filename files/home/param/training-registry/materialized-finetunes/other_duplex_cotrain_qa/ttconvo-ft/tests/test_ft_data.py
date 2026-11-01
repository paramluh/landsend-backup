"""CPU check of the FT data path on a host holding slot slinky-11 + slinky-18: plan rows -> PPM speech micros (swap 0), prompt rule, eval rows."""
import json, sys, collections
import numpy as np
sys.path.insert(0, "/data/home/amartyavarma/ppm/ttconvo-ft/code")
import data_ppm as D
P = json.load(open("/data/home/amartyavarma/ppm/ttconvo-ft/plan/PLAN.json"))
cfg = dict(D.default_cfg(seed=20261011, sp_cap=2000, persona_sides="both", light_frac=1.0, pose_light_frac=1.0, txt_cap=7500, crop_conv=True,
                         chat_side="assistant", chat_eob=True, pack_attn="causal"), voice_p=0.8, persona_p=0.2)
R = int(sys.argv[1])
ds = D.FTSpeechDS("/scratch/amartyavarma/ttconvo-ft/stores", "/data/home/amartyavarma/ppm/ttconvo-ft/persona", P["slots"], P["ranks"][R][:60], cfg, R, 0)
c = collections.Counter()
for i in range(len(ds)):
    m = ds[i]; pv = json.loads(m["prov"])
    assert m["swap"] == 0 and pv["chan_B_data_side"] == "assistant"
    assert m["ids"].shape == (2, 7500 + D.CD) and m["depth"].shape == (2, 7500 + D.CD, 32), (m["ids"].shape, D.CD)
    nw = int((m["ids"][1] == D.NW_ID).sum()); c["nw_rows"] += int(nw > 0)
    c["voice_drawn"] += pv["voice_drawn"]; c["persona_drawn"] += pv.get("persona_drawn", 0)
    c["clip"] += int(len(pv["clips"]) > 0); c["persona"] += int(len(pv["persona"]) > 0); c["short"] += int(pv["n_frames"] < 7500)
    if i < 2:
        k, r = ds.codes[i]; node, st, sib, ps = ds._open(k); row = st[r]
        P_ = m["P"]; lane = m["ids"][1, P_:]; words = [x for x in lane if x < D.CTRL_LO][:8]
        print("row", pv["row_key"], "P", P_, "clips", pv["clips"], "persona", pv["persona"], "nB_words_ids", len(words), "first B ids", words[:5],
              "row asst pieces", list(row["piece_ids_assistant"][:5]), "sel", len(m["sel"]), "light", len(m["light"]), "NEW_WORD", nw)
        assert list(row["piece_ids_assistant"][:5]) == [int(x) for x in lane[lane < D.CTRL_LO][:5]], "B lane != assistant words"
        la = m["ids"][0, P_:]; assert list(row["piece_ids_user"][:5]) == [int(x) for x in la[la < D.CTRL_LO][:5]], "A lane != user words"
print("rank", R, "rows", len(ds), dict(c))
e = D.FTSpeechDS("/data/home/amartyavarma/qwen-1ch-data/tt-v3-rows/evalstore", "/data/home/amartyavarma/ppm/ttconvo-ft/persona", ["rows"],
                 [(0, r) for r in P["eval_rows"][R]], cfg, R, 0, persona_slot="evalv3")
ks = [json.loads(e[i]["prov"])["row_key"] for i in range(len(e))]
assert ks == P["eval_keys"][R * 4:(R + 1) * 4], ks
print("eval ok", ks)
