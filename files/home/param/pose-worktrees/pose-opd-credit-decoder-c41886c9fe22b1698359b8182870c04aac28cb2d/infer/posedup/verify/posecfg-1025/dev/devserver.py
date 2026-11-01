"""run the STAGED server.py on a dev node with RANDOM weights (export-498 architecture): full load + smoke (incl. the pose CFG pass) + sessions.
state dir / sessions / token / feed = dev-only (FEED_URL.txt 'off'). usage: python devserver.py <ci.pkl> <port> <token file> <sessions dir>"""
import math, pickle, sys
DEMO = "/data/home/amartyavarma/ppm/aloe-posedup-live"
STAGE = DEMO + "/stage-posecfg"
sys.path.insert(0, DEMO)
sys.path.insert(0, STAGE)
import torch  # noqa: E402
import live_ppm_pose as LP  # noqa: E402
assert LP.__file__.startswith(STAGE), LP.__file__
ci = pickle.load(open(sys.argv[1], "rb"))


@torch.no_grad()
def rand_fill(m, ci_, adapter):
    g = {}
    for n, p in m.named_parameters():
        d = p.device
        if d not in g:
            g[d] = torch.Generator(device=d); g[d].manual_seed(1234 + (d.index or 0))
        if n.startswith("vgain."):
            p.fill_(math.log(0.0038))
        elif "norm" in n:
            p.fill_(1.0)
        elif n.endswith(".bias"):
            p.zero_()
        else:
            p.copy_(torch.empty(p.shape, dtype=torch.float32, device=d).normal_(0.0, 0.02, generator=g[d]).to(p.dtype))


LP.ckpt_info = lambda p: dict(ci)
LP.load_weights = rand_fill
sys.argv = ["server.py", "--ck", ci["path"], "--port", sys.argv[2], "--token-file", sys.argv[3], "--sessions", sys.argv[4]]
import server  # noqa: E402
assert server.__file__.startswith(STAGE), server.__file__
server.main()
