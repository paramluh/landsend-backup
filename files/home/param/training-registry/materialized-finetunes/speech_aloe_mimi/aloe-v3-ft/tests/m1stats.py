import pyarrow.parquet as pq, numpy as np, json, csv
S="/data/home/amartyavarma/qwen-1ch-data/s2h-m1/stores"
calls={}
for k in range(4):
    t=pq.read_table(f"{S}/m1n{k}/rows.parquet",columns=["row","row_key","src_conversation","split","chunk_index","n_frames","src_frames"]).to_pydict()
    for r,rk,c,sp,ci,n,T in zip(*[t[x] for x in ["row","row_key","src_conversation","split","chunk_index","n_frames","src_frames"]]):
        d=calls.setdefault(c,{"k":k,"split":sp,"T":T,"chunks":[]}); assert d["k"]==k and d["split"]==sp and d["T"]==T
        d["chunks"].append((ci,n))
T=np.array([d["T"] for d in calls.values()]); sp=np.array([d["split"] for d in calls.values()])
print("calls",len(calls),"train",(sp=="train").sum(),"eval",(sp=="eval").sum())
print("T pct",np.percentile(T,[0,10,50,90,99,100]).astype(int))
print("calls >13500:",(T>13500).sum(),"tail>=125:",(T-13500>=125).sum(),"tail>=1500:",(T-13500>=1500).sum(), ">27000:",(T>27000).sum())
n0=np.minimum(T,13500)
print("v3 hours train", n0[sp=="train"].sum()/12.5/3600, "eval", n0[sp=="eval"].sum()/12.5/3600, "all", n0.sum()/12.5/3600)
v2=np.minimum(T,7500)
print("v2 first-chunk hours train", v2[sp=="train"].sum()/12.5/3600)
m1rows=sum(sum(n for _,n in d["chunks"]) for d in calls.values())
print("M1 total frames hours",m1rows/12.5/3600)
man=set()
for r in csv.DictReader(open("/data/home/amartyavarma/data/aloe-handover-full/DATA/manifest.csv")):
    man.add(f"{r['session_id']}/{r['seg']}")
miss=[c for c in calls if c not in man]
print("purged (not in manifest):",len(miss),"train",sum(calls[c]["split"]=="train" for c in miss))
print("n rows at 13500:",(T>=13500).sum())
