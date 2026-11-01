"""time Stores.conv (decode + corruption + gain) and the podcast micro for a few items, stores from ROOT (argv[1]); CPU, 1 thread."""
import os, sys, time
os.environ["ALOE_SI20K_STORE_ROOT"] = sys.argv[1]
os.environ.setdefault("PPM_CD", "2"); os.environ["SI20K_A_CORRUPT"] = "qqa-corrupt-v2"; os.environ["SI20K_A_GAIN"] = "gauss:6"
sys.path.insert(0, "/data/home/amartyavarma/ppm/si20k-txt-ppm/code"); sys.path.insert(0, "/data/home/amartyavarma/ppm/si20k-txt-ppm/code/lib")
import data_si20k_vox as A
import podcast_rows as PR
PL = A.make_plan(64); st = A.Stores(); cfg = A.aloe_cfg()
its = PL["items"][int(sys.argv[2]):int(sys.argv[2]) + 4]
for it in its:
    t0 = time.time(); m = st.conv(it, cfg, 0, audio=False); t1 = time.time(); m = st.conv(it, cfg, 0, audio=True); t2 = time.time()
    print(f"si20k {it}: no-audio {t1 - t0:.2f}s, with audio+corrupt {t2 - t1:.2f}s ({m['a_gain']['corrupt'].get('mode')})", flush=True)
import socket
host = socket.gethostname().split(".")[0]
pl = PR.load_plan("/data/home/amartyavarma/ppm/si20k-vox-ppm/plan/PODCAST_PLAN_ppm.json"); tr, ev = PR.cfgs(pl)
src = PR.Source(pl, host, 0, tr)
for c in src.codes[:3]:
    t0 = time.time(); m = src.micro(c, audio=False); t1 = time.time(); m = src.micro(c, audio=True); t2 = time.time()
    print(f"podcast {c}: no-audio {t1 - t0:.2f}s, with audio+corrupt {t2 - t1:.2f}s ({m['aref'].get('corrupt')})", flush=True)
