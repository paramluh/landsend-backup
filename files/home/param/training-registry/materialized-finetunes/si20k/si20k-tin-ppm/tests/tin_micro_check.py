"""si20k-tin CPU check: a few si20k + podcast text-in micros: A lane words present, A Mimi absent everywhere, present_A as built, prefix A
empty, B as si20k-vox (B codes / clip), and the no-A blank."""
import json, os, socket, sys
os.environ.setdefault("PPM_CD", "2"); os.environ["SI20K_A_GAIN"] = "raw"; os.environ["SI20K_A_CORRUPT"] = "none"
os.environ["ALOE_SI20K_STORE_ROOT"] = "/data/home/amartyavarma/qwen-1ch-data/si20k-vox/stores"
C = sys.argv[1]
sys.path.insert(0, C); sys.path.insert(0, C + "/lib")
import numpy as np
import data_si20k_vox as A
import podcast_rows as PR
D = A.D
PL = A.make_plan(64); st = A.Stores(); cfg = A.aloe_cfg()
for it in PL["items"][:4] + PL["eval"][:2]:
    m = A.as_micro(st.conv(it, cfg, 0, audio=False))
    P = int(m["P"][0]) if hasattr(m["P"], "__len__") else int(m["P"])
    ids, dep, pr = np.asarray(m["ids"]), np.asarray(m["depth"]), np.asarray(m["present"])
    print("si20k", it, "P", P, "T", ids.shape[1], "A words", int((ids[0, P:] != D.PAD_ID).sum()), "A codes", int((dep[0] >= 0).sum()),
          "present_A conv", float(pr[0, P:].mean()), "prefix A ids", int((ids[0, :P] != D.PAD_ID).sum()), "B codes", int((dep[1] >= 0).sum()),
          "clip", m["voice_clip"], "persona", m["persona_used"], "has wav/vin", "wav" in m, "vin" in m, "text targets", int(np.asarray(m["own"]).sum(0)[0]))
host = socket.gethostname().split(".")[0]
pl = PR.load_plan("/data/home/amartyavarma/ppm/si20k-vox-ppm/plan/PODCAST_PLAN_ppm.json"); tr, ev = PR.cfgs(pl)
src = PR.Source(pl, host, 0, tr)
for c in src.codes[:3] + src.eval_codes[:1]:
    mm = src.micro(c); m = PR.as_micro(mm, c)
    P = int(mm["P"]); ids, dep, pr = np.asarray(m["ids"]), np.asarray(m["depth"]), np.asarray(m["present"])
    pv = json.loads(mm["prov"])
    print("podcast", c, "P", P, "T", ids.shape[1], "A words", int((ids[0, P:] != D.PAD_ID).sum()), "A codes", int((dep[0] >= 0).sum()),
          "present_A conv", round(float(pr[0, P:].mean()), 3), "prefix A ids", int((ids[0, :P] != D.PAD_ID).sum()), "B codes", int((dep[1] >= 0).sum()),
          "clips", pv.get("clips"), "a_clip_dropped", pv.get("a_clip_dropped"), "persona", pv.get("persona"), "has wav/vin", "wav" in m, "vin" in m)
