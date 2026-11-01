import os, sys
os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", PPM_CD="2")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_ppm as T
for kind in ("speech", "pose", "gemma"):
    n = T.NV["text" if kind == "gemma" else kind]
    v = [float(i + 1) for i in range(n)]
    d = T.summarize(kind, v, "eval")
    print(kind, n, sorted(k for k in d if "cb" in k))
d = T.summarize("speech_s3_A", [1.0] * T.NV["speech"], "eval", "speech"); print(sorted(d)[:3], len(d))
