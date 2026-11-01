import sys, numpy as np, json
G=sys.argv[1]; ids=sys.argv[2:]
T=[np.load(f"{G}/runs/{r}/tokens.npz") for r in ids]
W=[json.load(open(f"{G}/runs/{r}/status.json")).get("worker") for r in ids]
for i in range(1,len(ids)):
    for k in ("lane","mimi","pose"):
        a,b=T[0][k],T[i][k]; n=min(len(a),len(b)); eq=(a[:n]==b[:n]).all(axis=-1) if a.ndim>1 else (a[:n]==b[:n])
        first=int(np.argmin(eq)) if not eq.all() else None
        print(ids[i], W[0], "vs", W[i], k, "identical" if first is None else f"diverges at frame {first} of {n}")
