"""Scan every handover conversation (all 12,681, both lanes) for a name in profile.a_speaker / other_speaker -> prints account, lane, conv."""
import csv, json, sys
from concurrent.futures import ThreadPoolExecutor
DATA = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
pat = sys.argv[1].lower()
csv.field_size_limit(1 << 30)
rows = list(csv.DictReader(open(f"{DATA}/manifest.csv")))
def one(m):
    out = []
    for L in ("00", "01"):
        try:
            p = json.load(open(f"{DATA}/{m['relative_dir']}/system_prompts/SPEAKER_{L}/episode.json"))["profile"]
        except Exception:
            continue
        if pat in str(p.get("a_speaker", "")).lower():
            out.append((m["recording_key"], L, m[f"speaker_{L}_account_uid"], p.get("a_speaker"), m["combined_crossbleed_side"]))
    return out
with ThreadPoolExecutor(48) as ex:
    for r in ex.map(one, rows):
        for x in r:
            print(*x)
