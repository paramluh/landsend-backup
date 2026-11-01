import sys, time; sys.path.insert(0, "/data/home/amartyavarma/ppm/code-811")
import chat_ppm as CH
t=time.time(); a=CH.multi_split_packs([2,3], 7500, 20261011); b=CH.multi_split_packs([1], 7500, 20261011)
print("split2,3 packs", len(a), "split1 packs", len(b), "per rank +", len(b)//128, f"{time.time()-t:.0f}s")
tb, pool, ho = CH.table(1); print("split1 pool", len(pool), "heldout", len(ho), "first pack", b[0][0], b[0][1][:3])
s23 = set(int(r) for _, p in a for r in p); print("row-id overlap with split2/3 row ids (different tables, info only):", len(s23 & set(int(r) for _, p in b[:1000] for r in p)))
