import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import train2 as T2
torch.set_num_threads(1)
TR = T2.load_split("train")[:12]
mu, P = T2.fit_pca(TR, n=5000); T2.prepare(TR, mu, P)
print("tracks", len(TR), "valid starts", [len(t["valid"]) for t in TR][:6])
m = T2.Judge(); print("params M", T2.nparams(m) / 1e6)
rng = np.random.default_rng(0); opt = torch.optim.AdamW(m.parameters(), 1e-3)
t0 = time.time()
for s in range(10):
    sl = [(t, T2.pick_group(t["valid"], rng)) for t in TR[:8]]
    sl = [x for x in sl if x[1] is not None]
    p, w, mm = T2.batch(TR, sl)
    pe = m.enc_pose(p); se = m.enc_speech(w, mm)
    lg = m.logit_scale.exp() * pe @ se.T
    loss = torch.nn.functional.cross_entropy(lg, torch.arange(len(lg)))
    opt.zero_grad(); loss.backward(); opt.step()
print("10 steps", round(time.time() - t0, 1), "s, loss", loss.item(), "batch", p.shape, w.shape, mm.shape)
print(T2.evaluate(m, TR[:6], n_groups=2))
