import json, hashlib, collections, sys
NODES = [0,1,2,3,4,6,7,8,9,10,11,12,13,14,16,17]
L = [json.loads(l) for l in open("/scratch/amartyavarma/tt-v3/manifest/node18.jsonl")]
eps = collections.Counter(l["ep"] for l in L)
k = collections.Counter(NODES[int(hashlib.sha1(('tt-s3-node:'+e).encode()).hexdigest(),16) % 16] for e in eps)
sp = collections.Counter(l["split"] for l in L)
print("node18 manifest chunks", len(L), "episodes", len(eps), "slots(by episode)", dict(k), "split", dict(sp))
print(L[0])
