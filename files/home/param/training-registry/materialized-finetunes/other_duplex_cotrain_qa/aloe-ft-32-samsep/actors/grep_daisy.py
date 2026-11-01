"""Where is 'Daisy' said in the handover transcripts? For each conversation: who says it (lane), context, the lanes' accounts + identified names."""
import csv, json, re, sys
from concurrent.futures import ThreadPoolExecutor
DATA = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
csv.field_size_limit(1 << 30)
rows = list(csv.DictReader(open(f"{DATA}/manifest.csv")))
def one(m):
    p = f"{DATA}/{m['relative_dir']}/w00.words.json"
    try:
        t = open(p).read()
    except Exception:
        return None
    if not re.search(r"\bdaisy\b", t, re.I):
        return None
    d = json.loads(t)
    names = {}
    for L in ("00", "01"):
        try:
            names[L] = json.load(open(f"{DATA}/{m['relative_dir']}/system_prompts/SPEAKER_{L}/episode.json"))["profile"].get("a_speaker")
        except Exception:
            names[L] = None
    hits = []
    for seg in re.split(r"(?=\[SPEAKER_0[01]\])", d.get("text", "")):
        mm = re.match(r"\[SPEAKER_0([01])\]", seg)
        for h in re.finditer(r"\bdaisy\b", seg, re.I):
            hits.append((mm.group(1) if mm else "?", seg[max(0, h.start() - 70):h.end() + 50].replace("\n", " ")))
    return m["recording_key"], m["speaker_00_account_uid"][:8], names.get("00"), m["speaker_01_account_uid"][:8], names.get("01"), m["combined_crossbleed_side"], hits[:3]
with ThreadPoolExecutor(32) as ex:
    for r in ex.map(one, rows):
        if r:
            print(json.dumps(r))
