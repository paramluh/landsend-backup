"""posefix SAMSEP: staged server.prefill_inputs switch vs the live one (CPU): default 'real' must equal the live output exactly; 'none' drops
only the codes (lane + pose unchanged); 'sam' without openings_sam falls back to 'none'."""
import importlib.util, json, os, sys
LIVE = "/data/home/amartyavarma/ppm/aloe-posedup-live"
sys.path.insert(0, LIVE)
sys.argv = ["server.py"]
import numpy as np

def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
S0 = load(f"{LIVE}/server.py", "srv_live")
S1 = load(f"{LIVE}/stage-samsep/server.py", "srv_stage")
tok = S0.LP.IP.tok()
res = {}
for person in ("01_ab63d745", "03_b437950e", "10_27286bdd"):
    a = S0.prefill_inputs(person, 2.0, 0, True, True, True, tok)
    for mode in ("real", "none", "sam"):
        os.environ["PPMLIVE_DUPLEX_PREFILL_CODES"] = mode
        b = S1.prefill_inputs(person, 2.0, 0, True, True, True, tok)
        same_pose = np.array_equal(a[0], b[0])
        same_lane = np.array_equal(a[1]["lane"], b[1]["lane"])
        codes_a, codes_b = a[1].get("codes"), (b[1] or {}).get("codes")
        same_codes = (codes_b is not None) and np.array_equal(codes_a, codes_b)
        res[f"{person}/{mode}"] = {"pose_same": same_pose, "lane_same": same_lane, "codes_same": same_codes,
                                   "codes_present": codes_b is not None, "info_equal": a[2] == b[2], "notes": b[2]["notes"]}
        print(person, mode, json.dumps(res[f"{person}/{mode}"]), flush=True)
json.dump(res, open(f"{LIVE}/posefix/samsep/test_switch.json", "w"), indent=1)
