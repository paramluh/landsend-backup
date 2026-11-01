import pyarrow.parquet as pq, numpy as np, collections
C="/data/home/amartyavarma/aloe-pose/sam-cache-active-v2pure/eval"
ci = pq.read_table(f"{C}/items.parquet").to_pandas()
print(ci.columns.tolist()); print(len(ci), ci.rid.nunique())
print(ci.iloc[0].to_dict().keys())
runs=[]
for (rid, spk), g in ci.groupby(["rid","speaker"]):
    g=g.sort_values("start_frame_25fps")
    s=g.start_frame_25fps.values; n=g.n.values
    cur=[0]
    for i in range(1,len(g)):
        gap = s[i] - (s[i-1] + 2*n[i-1])
        if abs(gap) <= 1:
            cur.append(i)
        else:
            runs.append((rid,spk,[g.index[j] for j in cur], int(sum(n[j] for j in cur)))); cur=[i]
    runs.append((rid,spk,[g.index[j] for j in cur], int(sum(n[j] for j in cur))))
L=np.array([r[3] for r in runs])
print("runs", len(runs), "max frames", L.max(), "runs >=750:", (L>=750).sum(), ">=600:", (L>=600).sum(), ">=500", (L>=500).sum())
gaps=collections.Counter()
for (rid, spk), g in ci.groupby(["rid","speaker"]):
    g=g.sort_values("start_frame_25fps"); s=g.start_frame_25fps.values; n=g.n.values
    for i in range(1,len(g)): gaps[int(s[i]-(s[i-1]+2*n[i-1]))//25]+=1
print("gap histogram (s):", sorted(gaps.items())[:25])
big=sorted(runs, key=lambda r:-r[3])[:15]
for r in big: print(r[0][:10], r[1], len(r[2]), r[3], [int(ci.at[i,'n']) for i in r[2]])
