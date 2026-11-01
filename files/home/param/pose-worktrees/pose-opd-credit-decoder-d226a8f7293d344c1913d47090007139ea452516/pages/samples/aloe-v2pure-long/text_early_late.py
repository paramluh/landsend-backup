"""generated text (model word lane) of the 120 s runs: words, distinct-2 and word=previous-word share in the first 30 s (2-32 s) vs the last
30 s (90-120 s) of the generated part, real lane alongside. python text_early_late.py <work root>"""
import json
import re
import sys

W = sys.argv[1]
rows = [r["row"] for r in json.load(open(f"{W}/rows/rows.json"))]


def stats(ws):
    ws = [re.sub(r"[^\w']", "", w.lower()) for w in ws]
    ws = [w for w in ws if w]
    bi = list(zip(ws, ws[1:]))
    return len(ws), (len(set(bi)) / len(bi) if bi else 0), (sum(a == b for a, b in bi) / len(bi) if bi else 0)


for src in ("s1", "gt"):
    E, L = [], []
    for r in rows:
        j = json.load(open(f"{W}/runs-gen-k140/{r}/words.json"))
        ws = j["s1"] if src == "s1" else j["gt"]["s1"]
        E += [w["text"] for w in ws if not w["prompt"] and 2.0 <= w["t"] < 32.0]
        L += [w["text"] for w in ws if not w["prompt"] and 90.0 <= w["t"] < 120.0]
    e, l = stats(E), stats(L)
    print(f"{'generated' if src == 's1' else 'real':9s} first 30 s: {e[0]} words, distinct-2 {e[1]:.2f}, word=prev {100 * e[2]:.1f}% | last 30 s: {l[0]} words, "
          f"distinct-2 {l[1]:.2f}, word=prev {100 * l[2]:.1f}%")
j = json.load(open(f"{W}/runs-gen-k140/M-003/words.json"))
print("M-003 last 30 s:", " ".join(w["text"] for w in j["s1"] if w["t"] >= 90.0)[:500])
