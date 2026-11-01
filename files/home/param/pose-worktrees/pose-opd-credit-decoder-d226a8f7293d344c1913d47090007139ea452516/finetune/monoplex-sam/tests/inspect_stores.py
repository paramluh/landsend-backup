import numpy as np, pyarrow.parquet as pq, json
R="/data/home/amartyavarma/ppm/aloe-pose-ft/stores"
ns_tr=[]; ns_ev=[]; nrow={"train":0,"eval":0}
for k in range(8):
    t=pq.read_table(f"{R}/p{k}/rows.parquet")
    if k==0: print(t.schema.names)
    d=t.select(["row","split","n_frames","row_key"]).to_pydict()
    z=np.load(f"{R}/p{k}/pose_index.npz"); b=z["b_ok"]
    if k==0: print(list(z.keys()), b.shape, b.dtype)
    for r,s,n in zip(d["row"],d["split"],d["n_frames"]):
        nrow[s]+=1
        for sw in (0,1):
            if b[r,1-sw]:
                (ns_tr if s=="train" else ns_ev).append(n)
ns=np.array(ns_tr); print("rows",nrow,"train items",len(ns),"eval items",len(ns_ev))
print("frames total",ns.sum(), "hours",ns.sum()/12.5/3600)
print("pcts",np.percentile(ns,[0,1,5,10,25,50,75,90,100]))
print("frac==7500",(ns==7500).mean(),"frac<2048",(ns<2048).mean(),"frac<3750",(ns<3750).mean(), "lt2",(ns<2).sum(), (np.array(ns_ev)<2).sum())
print("print row_key sample", d["row_key"][:2])
