"""pose5 C: fetch the 11 small files (digest-*.txt, digest-rep.txt, train_state.json, COMPLETE) of every host dir of HF InternalCan/ppm-v1-c
dcp-17850/ into c/hf17850/dl (byte references for launch-day verification). Read only on HF."""
from huggingface_hub import hf_hub_download
D = "/data/home/amartyavarma/ppm/pose5/c/hf17850"
n = 0
for ln in open(f"{D}/hf_files.tsv").read().splitlines()[1:]:
    p, s, sha = ln.split("\t")
    if p.startswith("dcp-17850/") and not sha:
        hf_hub_download("InternalCan/ppm-v1-c", p, repo_type="model", local_dir=f"{D}/dl"); n += 1
print("small files", n)
