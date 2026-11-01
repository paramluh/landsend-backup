import re, collections, pyarrow.parquet as pq, sys
F = "/data/home/amartyavarma/ppm/code-811/fresh/scan"
pat = re.compile(r"^(\d+)_(\d+)-(\d+(?:\.\d+)?)_to_(\d+)-(\d+(?:\.\d+)?)$")
c = collections.Counter(); ex = collections.defaultdict(list); mx = 0; segfmt = collections.Counter()
for h in sys.argv[1:]:
    t = pq.read_table(f"{F}/{h}.parquet", columns=["segment_id", "clip_id", "video_id", "parent_clip_id", "segment_index"]).to_pydict()
    for s, cl, v, pc, si in zip(t["segment_id"], t["clip_id"], t["video_id"], t["parent_clip_id"], t["segment_index"]):
        m = pat.match(cl)
        if not m: c["noparse"] += 1; ex["noparse"].append(cl) if len(ex["noparse"]) < 5 else None; continue
        if m.group(1) != v: c["vid_mismatch"] += 1
        a = int(m.group(2)) * 60 + float(m.group(3)); b = int(m.group(4)) * 60 + float(m.group(5))
        mx = max(mx, b); c["ok"] += 1
        if b <= a: c["nonpos"] += 1
        if pc != cl: c["parent_ne_clip"] += 1; ex["pc"].append((cl, pc)) if len(ex["pc"]) < 3 else None
        segfmt[s[len(cl):len(cl) + 5] if s.startswith(cl) else "NOPREFIX"] += 1
        if si > 0: c["si>0"] += 1
print(dict(c), "max end s", mx, dict(segfmt.most_common(5)), dict(ex))
