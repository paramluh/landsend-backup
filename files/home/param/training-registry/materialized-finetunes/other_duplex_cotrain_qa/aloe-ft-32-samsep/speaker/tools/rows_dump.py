"""init subset (10-24): this host's export-249 rank files -> {rank: {tensor: dim-0 rows}} (the input init_subset_hosts.py reads from one dir;
the world-64 export is spread over 8 hosts, so every host dumps its own files and subset_w16.py merges them). usage: python -I rows_dump.py <out dir>"""
import glob, json, os, socket, sys
import torch
X = "/scratch/amartyavarma/aloe-ft-32-samsep/exports-vox17127/export-249"
H = socket.gethostname().split(".")[0]
out = {}
for f in sorted(glob.glob(f"{X}/rank*.pt")):
    sd = torch.load(f, map_location="cpu", mmap=True, weights_only=True)
    out[int(os.path.basename(f)[4:-3])] = {k: (int(t.shape[0]) if t.dim() else 1) for k, t in sd.items()}
json.dump({"host": H, "ranks": out}, open(f"{sys.argv[1]}/rows_{H}.json", "w"))
print(H, sorted(out), len(next(iter(out.values()))))
