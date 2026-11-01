import sys, os
sys.argv = [sys.argv[0], "/dev/null", "1"] + sys.argv[1:]
src = open("/data/home/amartyavarma/ppm/aloe-posedup-live/fastmg/dev/t_tp.py").read()
pre = src[:src.index("# ---- 1. gemv vs F.linear")]
funcs = src[src.index("# ---- 4. the TP trunk on a bare engine object"):src.index("# 4a. correctness")]
exec(pre + funcs)
G = int(os.environ.get("TPG", 4)); nlay = int(os.environ.get("NL", 2)); spl = int(os.environ.get("SPL", 1))
e = bare(G, 3, nlay, 9600, splits=spl)
gs = capture(e)
print("captured", flush=True)
for T in (100, 2000):
    e.set_T(T)
    for r, d in enumerate(e.tpd):
        with torch.cuda.device(d):
            gs[r].replay()
    for d in e.tpd:
        torch.cuda.synchronize(d)
        print("sync ok", d, T, flush=True)
e.check_errors()
print("OK", G, nlay, spl)
