import json, re, sys, csv
csv.field_size_limit(1 << 30)
DATA = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
man = {m["relative_dir"]: m for m in csv.DictReader(open(f"{DATA}/manifest.csv"))}
pat = re.compile(r"\bdais(y|ie|ey|i)?\b", re.I)
for f in open(sys.argv[1]).read().split():
    rel = f.split("/DATA/")[1].split("/system_prompts/")[0]
    lane = f.split("/system_prompts/")[1].split("/")[0][-2:]
    m = man.get(rel, {})
    t = open(f).read()
    d = json.loads(t)
    prof = d.get("profile", {}) if "profile" in d else {}
    ctx = [t[max(0, x.start() - 90):x.end() + 50].replace("\\n", " ") for x in pat.finditer(t)][:2]
    print(f"{m.get('recording_key','?')[:13]} lane{lane} acc={m.get(f'speaker_{lane}_account_uid','?')[:8]} a_speaker={prof.get('a_speaker')} other={prof.get('other_speaker')} {f.split('/')[-1]} :: " + " || ".join(ctx))
