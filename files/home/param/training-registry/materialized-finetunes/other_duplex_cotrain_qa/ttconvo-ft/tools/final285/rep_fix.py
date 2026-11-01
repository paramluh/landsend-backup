"""export-285 rep.pt: re-write so its torch.save archive name is 'rep' like the trainer's export_local (which saves straight to rep.pt;
export_from_dcp.py had saved to rep.pt.tmp -> archive 'rep.pt', same tensors, 35 KB larger zip). Source = dcp-285/rep.pt (digest checked
again), written to <out>/.repfix/rep.pt (archive name from that file name) and os.replace'd onto <out>/rep.pt; re-read equal asserted."""
import os

import sys
import zipfile

import torch

sys.path.insert(0, "/data/home/amartyavarma/ppm/ttconvo-ft/code")
import safety_ppm as SF  # noqa: E402

torch.set_num_threads(1)
src, out = "/scratch/amartyavarma/ttconvo-ft/ckpt/ttconvo-ft/dcp-285", "/scratch/amartyavarma/ttconvo-ft/exports/export-285"
sd = torch.load(f"{src}/rep.pt", map_location="cpu", mmap=True, weights_only=True)
assert SF.digest_state(sd) == open(f"{src}/digest-rep.txt").read().strip(), "rep digest"
m = sd["model"]
assert all(bool(torch.isfinite(t).all()) for t in m.values())
o = {n: t.to(torch.bfloat16).contiguous().clone() for n, t in m.items()}
os.makedirs(f"{out}/.repfix", exist_ok=True)
torch.save(o, f"{out}/.repfix/rep.pt")
assert zipfile.ZipFile(f"{out}/.repfix/rep.pt").namelist()[0] == "rep/data.pkl"
back = torch.load(f"{out}/.repfix/rep.pt", map_location="cpu", mmap=True, weights_only=True)
assert list(back) == list(m) and all(torch.equal(back[n], m[n].to(torch.bfloat16)) for n in m), "re-read"
del back
os.replace(f"{out}/.repfix/rep.pt", f"{out}/rep.pt")
os.rmdir(f"{out}/.repfix")                         # empty temp dir of this fix
print("rep.pt rewritten", os.path.getsize(f"{out}/rep.pt"), "B, archive", zipfile.ZipFile(f"{out}/rep.pt").namelist()[0])
