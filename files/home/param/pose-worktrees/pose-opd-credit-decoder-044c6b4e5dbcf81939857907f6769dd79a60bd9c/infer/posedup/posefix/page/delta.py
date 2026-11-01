#!/usr/bin/env python3
"""posefix page: print the files map (published path -> path under the page dir) of files changed since the last publish mark, capped at
~55 MB per publish (data.json always included), then update the mark. usage: python3 delta.py [--all]"""
import json
import os
import sys

P = os.path.dirname(os.path.abspath(__file__))
MARK = f"{P}/.published.json"
files = json.load(open(f"{P}/files.json"))
files.pop("index.html", None)
seen = json.load(open(MARK)) if os.path.isfile(MARK) and "--all" not in sys.argv else {}
out, tot = {"data.json": "data.json"}, os.path.getsize(f"{P}/data.json")
for k, v in files.items():
    st = os.stat(f"{P}/{v}")
    sig = f"{st.st_size}:{int(st.st_mtime)}"
    if k == "data.json" or seen.get(k) == sig:
        continue
    if tot + st.st_size > 55e6:
        break
    out[k] = v
    tot += st.st_size
    seen[k] = sig
json.dump(seen, open(MARK, "w"))
left = sum(1 for k, v in files.items() if k != "data.json" and seen.get(k) != "%d:%d" % (os.stat(f"{P}/{v}").st_size, int(os.stat(f"{P}/{v}").st_mtime)))
print(json.dumps(out))
print(f"# {len(out)} files {tot / 1e6:.1f} MB; {left} changed files left for the next publish", file=sys.stderr)
