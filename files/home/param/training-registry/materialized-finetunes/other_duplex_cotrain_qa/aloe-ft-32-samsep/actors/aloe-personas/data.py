"""Personas of the Aloe FT TRAIN rows (sam-sep stores, first chunks, split train; 7,949 rows x 2 sides).
Per side: the store's persona sidecar (rows-*.persona.arrow: status, *_tokens_json, *_system_prompt = what a persona FT would train with) and
the chunk prompt it was rendered from (aloe-handover-full .../system_prompts/SPEAKER_NN/chunk_00.json: sections.persona_items with kind
real / variant / invented and field) -> data.json (stats + 15 examples with a Persona part + 1 context-only)."""
import collections, csv, hashlib, json, re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pyarrow as pa, pyarrow.ipc as ipc
OUT = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/aloe-personas/data.json"
STORES = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores"
DATA = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
NM = json.load(open("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/name_map.json"))
csv.field_size_limit(1 << 30)
man = {m["recording_key"]: m for m in csv.DictReader(open(f"{DATA}/manifest.csv"))}
LANE = {"assistant": "00", "user": "01"}
items = []
for k in range(4):
    meta = [json.loads(l) for l in open(f"{STORES}/m1n{k}/rows.jsonl")]
    t = ipc.open_file(pa.memory_map(f"{STORES}/m1n{k}/rows-00000-of-00001.persona.arrow", "r")).read_all().to_pylist()
    assert len(t) == len(meta)
    for r, p in zip(meta, t):
        assert p["row_key"] == r["row_key"]
        if int(r["chunk_index"]) != 0 or r["split"] != "train":
            continue
        m = man[r["src_conversation"]]
        for s in ("user", "assistant"):
            items.append({"row_key": r["row_key"], "conv": r["src_conversation"], "side": s, "status": p[f"{s}_status"], "note": p[f"{s}_note"],
                          "prompt": p[f"{s}_system_prompt"] or "", "tokens": json.loads(p[f"{s}_tokens_json"] or "{}"),
                          "flags": json.loads(p[f"{s}_flags_json"] or "{}"), "version": p[f"{s}_version"],
                          "account": m[f"speaker_{LANE[s]}_account_uid"], "chunk_json": f"{DATA}/{m['relative_dir']}/system_prompts/SPEAKER_{LANE[s]}/chunk_00.json"})
def load(it):
    try:
        c = json.load(open(it["chunk_json"]))
        return c["sections"], c.get("sampling", {}), c.get("system_prompt")
    except Exception as e:
        return None, None, None
with ThreadPoolExecutor(32) as ex:
    res = list(ex.map(load, items))
n = len(items) // 2
st = collections.Counter(); sec_tok = collections.defaultdict(list); tot_tok = collections.defaultdict(list)
field_conv = collections.Counter(); field_items = collections.Counter(); field_kind = collections.defaultdict(collections.Counter); field_words = collections.defaultdict(list)
kind_items = collections.Counter(); n_items = []; modes = collections.Counter(); same_prompt = collections.Counter(); sec_present = collections.Counter()
flag_c = collections.Counter()
ok_items = []
for it, (sec, samp, cprompt) in zip(items, res):
    st[(it["side"], it["status"])] += 1
    if it["status"] != "ok":
        continue
    tk = it["tokens"]
    tot_tok[it["side"]].append(tk.get("system_prompt", 0))
    for key in ("persona", "starting_context", "memory"):
        if tk.get(key):
            sec_present[key] += 1
            sec_tok[key].append(tk[key])
    flag_c[("channel", (it["flags"].get("channel_check") or {}).get("status"))] += 1
    flag_c[("mid_call_start", it["flags"].get("mid_call_start"))] += 1
    if sec is None:
        st[("chunk_json", "missing")] += 1
        continue
    same_prompt[cprompt == it["prompt"]] += 1
    pi = sec.get("persona_items") or []
    n_items.append(len(pi))
    modes[samp.get("mode")] += 1
    for f in {x.get("field") for x in pi}:
        field_conv[f] += 1
    for x in pi:
        field_items[x.get("field")] += 1
        field_kind[x.get("field")][x.get("kind")] += 1
        kind_items[x.get("kind")] += 1
        field_words[x.get("field")].append(len((x.get("text") or "").split()))
    ok_items.append((it, sec, samp))
q = lambda a, p: float(np.percentile(a, p)) if len(a) else 0  # noqa: E731
allt = tot_tok["user"] + tot_tok["assistant"]
nok = len(ok_items)
stats = {"rows": n, "status": {f"{a}|{b}": v for (a, b), v in st.items()},
         "tokens": {s: {"median": q(v, 50), "p90": q(v, 90), "mean": float(np.mean(v)) if v else 0, "n": len(v)} for s, v in [("user", tot_tok["user"]), ("assistant", tot_tok["assistant"]), ("all", allt)]},
         "sections": {k: {"share": sec_present[k] / max(1, len(allt)), "median": q(v, 50), "p90": q(v, 90)} for k, v in sec_tok.items()},
         "items_per_prompt": {"median": q(n_items, 50), "p90": q(n_items, 90), "mean": float(np.mean(n_items)) if n_items else 0},
         "kinds": dict(kind_items), "modes": dict(modes), "same_prompt": {str(k): v for k, v in same_prompt.items()},
         "fields": [{"field": f, "conv_share": field_conv[f] / max(1, nok), "items": field_items[f], "per_prompt": field_items[f] / max(1, nok),
                     "kinds": dict(field_kind[f]), "median_words": q(field_words[f], 50)} for f, _ in field_conv.most_common()],
         "flags": {f"{a}|{b}": v for (a, b), v in flag_c.items()}, "n_ok_with_items": nok}
# 15 examples: distinct accounts, alternate sides, spread over length quintiles, deterministic order
allc = sorted(ok_items, key=lambda x: hashlib.md5((x[0]["row_key"] + x[0]["side"]).encode()).hexdigest())
cand = [x for x in allc if x[1].get("persona_items")]           # 15 with a Persona part, spread over its length range, + 1 context-only
toks = np.array([x[0]["tokens"].get("system_prompt", 0) for x in cand])
edges = np.percentile(toks, [0, 20, 40, 60, 80, 100])
ex, seen = [], set()
for b in range(5):
    for side in ("user", "assistant", "user"):
        for (it, sec, samp), tkn in zip(cand, toks):
            if it["account"] in seen or it["side"] != side or not (edges[b] <= tkn <= edges[b + 1]):
                continue
            seen.add(it["account"])
            nm = (NM.get(it["account"]) or {}).get("name")
            ex.append({"row_key": it["row_key"], "conv": it["conv"], "side": it["side"], "account": it["account"], "name": nm, "tokens": it["tokens"],
                       "items": [{"kind": x.get("kind"), "field": x.get("field"), "text": x.get("text"), "related": x.get("related_to_topic")} for x in sec.get("persona_items") or []],
                       "persona": sec.get("persona", ""), "starting_context": sec.get("starting_context", ""), "memory": sec.get("memory", ""),
                       "prompt": it["prompt"], "mode": samp.get("mode"), "flags": it["flags"]})
            break
for it, sec, samp in allc:
    if not sec.get("persona_items") and it["account"] not in seen:
        ex.append({"row_key": it["row_key"], "conv": it["conv"], "side": it["side"], "account": it["account"], "name": (NM.get(it["account"]) or {}).get("name"),
                   "tokens": it["tokens"], "items": [], "persona": "", "starting_context": sec.get("starting_context", ""), "memory": "", "prompt": it["prompt"],
                   "mode": samp.get("mode"), "flags": it["flags"]})
        break
json.dump({"stats": stats, "examples": ex}, open(OUT, "w"))
print(json.dumps(stats)[:3000])
print("examples", len(ex))
