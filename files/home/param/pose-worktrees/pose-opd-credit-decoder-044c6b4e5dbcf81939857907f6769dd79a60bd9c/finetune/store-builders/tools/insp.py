import csv, json, glob, os, math, collections, sys
import numpy as np
D="/data/home/amartyavarma/data/aloe-handover-full/DATA"
man={r["relative_dir"]:r for r in csv.DictReader(open(f"{D}/manifest.csv"))}
pose=collections.defaultdict(dict)
for r in csv.DictReader(open(f"{D}/pose_manifest.csv")):
    pose[r["relative_dir"]][r["speaker"]]=r
sel=[]
for rd,m in man.items():
    side=m["combined_crossbleed_side"]
    q=[spk for spk in ("SPEAKER_00","SPEAKER_01") if pose.get(rd,{}).get(spk,{}).get("pose_status")=="ok" and side not in (spk,"both","unknown")]
    if q: sel.append((rd,q))
print("convs with >=1 qualifying B", len(sel), "both", sum(len(q)==2 for _,q in sel))
# tail lengths, chunk counts
tails=collections.Counter(); nchunk_mismatch=0; ex=0
T_all=[]
for rd,q in sel:
    T=int(man[rd]["mimi_T"]); T_all.append(T)
    n_last=T-7500*(math.ceil(T/7500)-1)
    for spk in q:
        tails["<125" if n_last<125 else "<1500" if n_last<1500 else ">=1500"]+=1
print("tail-chunk lengths per B track", tails)
print("T min/max", min(T_all), max(T_all), "sessions", len({rd.split('/')[2] for rd,_ in sel}))
# sample checks on 30 convs
import random; random.seed(0)
for rd,q in random.sample(sel,30):
    d=f"{D}/{rd}"; m=man[rd]; T=int(m["mimi_T"])
    W=json.load(open(f"{d}/w00.words.json"))
    if ex==0:
        print("words.json keys", list(W.keys())[:10], "word0", W["words"][0], "aloe", (W.get("aloe") or {}).get("clip"))
    spk=collections.Counter(w.get("_separated_speaker") for w in W["words"])
    nch={s: len(glob.glob(f"{d}/system_prompts/{s}/chunk_*.json")) for s in ("SPEAKER_00","SPEAKER_01")}
    roles={}
    for s in ("SPEAKER_00","SPEAKER_01"):
        f=sorted(glob.glob(f"{d}/system_prompts/{s}/chunk_*.json"))
        if f:
            j=json.load(open(f[0])); roles[s]=j.get("persona_speaker")
    c0=np.load(f"{d}/w00.SPEAKER_00.codes.npy", mmap_mode="r")
    pt={s: np.load(f"{d}/w00.{s}.pose_tokens.npy") for s in q}
    pinfo={s:(str(a.dtype),a.shape,int(a.min()),int(a.max())) for s,a in pt.items()}
    mj=json.load(open(f"{d}/w00.mimi.json"))
    print(rd.split('/')[1], T, c0.dtype, c0.shape, "spk", dict(spk), "chunks", nch, "ceil", math.ceil(T/7500), "roles", roles, "pose", pinfo, "mimi clip", mj["clip"])
    ex+=1
