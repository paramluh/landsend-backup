import json, re, sys
def cfg(p):
    for l in open(p):
        m = re.search(r"\]: world \d+ .*?; cfg (\{.*\})\s*$", l) or re.search(r"qqa-ft \S+: world .*cfg (\{.*\})\s*$", l)
        if m:
            return json.loads(m.group(1))
def flat(d, p=""):
    o = {}
    for k, v in d.items():
        o.update(flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v})
    return o
a, b = flat(cfg(sys.argv[1])), flat(cfg(sys.argv[2]))
for k in sorted(set(a) | set(b)):
    if a.get(k, "<absent>") != b.get(k, "<absent>"):
        print(f"{k}: {a.get(k, '<absent>')} -> {b.get(k, '<absent>')}")
