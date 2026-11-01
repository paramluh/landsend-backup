"""Persona stats + examples for the Training Personas page. Reads only; writes data.json next to this script.
PPM podcast: sample of processed/{r1,r2}/sXXXX/<node>.parquet (status ok rows); row/side counts from sidecars/REPORT.json.
Aloe: every manifest.csv segment, system_prompts/SPEAKER_0N/chunk_KK.json (SPEAKER_00 = assistant, SPEAKER_01 = user)."""
import collections, csv, glob, json, os, random, re, statistics, sys
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq, pyarrow.compute as pc

OUT = os.path.dirname(os.path.abspath(__file__))
P = "/data/home/amartyavarma/qwen-1ch-data/persona"
AL = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
NFILES = int(sys.argv[1]) if len(sys.argv) > 1 else 36
CAP = 10 ** 9
OCC = ["author", "professor", "founder", "ceo", "coach", "journalist", "reporter", "pastor", "comedian", "lawyer", "attorney", "therapist",
       "psychologist", "producer", "writer", "teacher", "entrepreneur", "researcher", "scientist", "physician", "doctor", "consultant",
       "editor", "historian", "economist", "artist", "musician", "actor", "athlete", "nurse", "student", "director", "president",
       "analyst", "priest", "rabbi", "chef", "designer", "engineer", "investor", "creator", "lecturer", "speaker", "columnist", "comic"]


def name_of(a):
    a = re.sub(r"\([^)]*\)", " ", a or "")
    a = re.split(r"[,;—–]| - ", a)[0].strip()
    toks = [t for t in a.split() if t.lower().strip(".") not in ("dr", "mr", "mrs", "ms", "prof", "rev", "pastor", "the")]
    if not toks or toks[0].lower().strip(".,") in ("host", "co-host", "cohost", "guest", "unnamed", "unknown", "unidentified", "speaker",
                                                    "interviewer", "narrator", "caller", "a", "an", "male", "female", "both", "first", "second"):
        return None
    return toks[0].strip(".'\"")


def conv_role(a, sc):
    a, sc = (a or "").lower(), (sc or "").lower()
    if "co-host" in a or "cohost" in a or "co host" in a:
        return "co-host"
    if re.search(r"\bhost", a):
        return "host"
    if "guest" in a or "interviewee" in a:
        return "guest"
    if "co-host" in sc:
        return "co-host"
    if "you are a guest" in sc or "being interviewed" in sc or "you are the guest" in sc:
        return "guest"
    if "hosting" in sc or "your guest" in sc or "you are the host" in sc:
        return "host"
    return "other"


def occs(a):
    a = re.sub(r"\bdr\.?\s", "doctor ", (a or "").lower())
    return {o for o in OCC if re.search(r"\b" + o + r"s?\b", a)}


def summ(xs):
    if not xs:
        return None
    xs = sorted(xs)
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 1), "median": xs[len(xs) // 2], "p10": xs[len(xs) // 10], "p90": xs[len(xs) * 9 // 10]}


def ppm():
    fs = sorted(glob.glob(f"{P}/processed/r1/s*/*.parquet") + glob.glob(f"{P}/processed/r2/s*/*.parquet"))
    random.seed(0)
    fs = random.sample(fs, NFILES)
    st = collections.Counter()
    tok = collections.defaultdict(list); words = collections.defaultdict(list); ptok = collections.defaultdict(list)
    names, roles, occ = collections.Counter(), collections.Counter(), collections.Counter()
    seen = set(); cands = collections.defaultdict(list)
    for f in fs:
        t = pq.read_table(f, columns=["ek", "side", "chunk_index", "status", "system_prompt", "sections_json", "tokens_json", "flags_json", "prompt_tokens"])
        for side in ("user", "assistant"):
            ts = t.filter(pc.equal(t["side"], side))
            st[f"{side}_rows"] += ts.num_rows
            st[f"{side}_ok"] += int(pc.sum(pc.equal(ts["status"], "ok")).as_py() or 0)
        t = t.filter(pc.equal(t["status"], "ok"))
        for r in t.to_pylist():
            ds, side, k = r["ek"].split("/")[0], r["side"], r["chunk_index"]
            sec = json.loads(r["sections_json"] or "{}"); tk = json.loads(r["tokens_json"] or "{}"); fl = json.loads(r["flags_json"] or "{}")
            n = r["prompt_tokens"]
            for key in (side, f"{ds}:{side}", "all", "chunk0" if k == 0 else "chunk>0"):
                tok[key].append(n); words[key].append(len(r["system_prompt"].split()))
            ptok["persona"].append(int(tk.get("persona", 0) or 0)); ptok["starting_context"].append(int(tk.get("starting_context", 0) or 0))
            if k > 0:
                ptok["memory(chunk>0)"].append(int(tk.get("memory", 0) or 0))
            st["mem_nonempty" if (sec.get("memory") or "").strip() else "mem_empty"] += 1
            a = (fl.get("channel_check") or {}).get("a_speaker")
            if (r["ek"], side) not in seen:
                seen.add((r["ek"], side))
                nm = name_of(a)
                names[nm or "(unnamed)"] += 1
                roles[conv_role(a, sec.get("starting_context"))] += 1
                for o in occs(a):
                    occ[o] += 1
            if n <= CAP:
                c = cands[(ds, side, "c0" if k == 0 else "cN")]
                if len(c) < 400:
                    c.append({"dataset": f"PPM podcast {ds}", "tree": ds, "side": side, "key": f"{r['ek']} · chunk {k}", "chunk": k,
                              "tokens": n, "words": len(r["system_prompt"].split()), "who": a, "text": r["system_prompt"]})
        print("[ppm]", f, st["user_ok"] + st["assistant_ok"], flush=True)
    random.seed(1)
    ex, used = [], set()
    plan = [("A", "user", "c0"), ("A", "assistant", "cN"), ("A", "user", "cN"), ("A", "assistant", "c0"), ("A", "assistant", "cN"),
            ("B", "user", "cN"), ("B", "assistant", "c0"), ("B", "user", "c0"), ("B", "assistant", "cN"), ("B", "user", "cN")]
    for p in plan:
        c = [x for x in cands[p] if x["key"].rsplit(" · ", 1)[0] not in used]
        x = random.choice(c); used.add(x["key"].rsplit(" · ", 1)[0]); ex.append(x)
    return {"files_sampled": len(fs), "counts": dict(st), "eps_sides": len(seen), "tokens": {k: summ(v) for k, v in tok.items()},
            "words": {k: summ(v) for k, v in words.items()}, "section_tokens": {k: summ(v) for k, v in ptok.items()},
            "names": names.most_common(15), "roles": roles.most_common(), "occupations": occ.most_common(15)}, ex


SIDE_OF = {"SPEAKER_00": "assistant", "SPEAKER_01": "user"}


def aloe_seg(rel):
    d = f"{AL}/{rel}"
    out = []
    for spk in ("SPEAKER_00", "SPEAKER_01"):
        fs = sorted(glob.glob(f"{d}/system_prompts/{spk}/chunk_*.json"))
        hasdir = os.path.isdir(f"{d}/system_prompts/{spk}")
        js = []
        for f in fs:
            try:
                js.append(json.load(open(f)))
            except Exception:
                js.append(None)
        out.append((rel, spk, hasdir, js))
    return out


def aloe():
    rels = [r["relative_dir"] for r in csv.DictReader(open(f"{AL}/manifest.csv"))]
    st = collections.Counter(); tok = collections.defaultdict(list); words = collections.defaultdict(list); ptok = collections.defaultdict(list)
    names, rel_ = collections.Counter(), collections.Counter(); cands = collections.defaultdict(list); vers = collections.Counter()
    with ThreadPoolExecutor(32) as ex:
        for i, res in enumerate(ex.map(aloe_seg, rels, chunksize=16)):
            if i % 2000 == 0:
                print("[aloe]", i, flush=True)
            for rel, spk, hasdir, js in res:
                side = SIDE_OF[spk]; bucket = rel.split("/")[1]
                st[f"{side}_segments"] += 1; st[f"{side}_segments_{bucket}"] += 1
                good = [j for j in js if j and (j.get("system_prompt") or "").strip()]
                if not good:
                    continue
                st[f"{side}_seg_with_prompt"] += 1; st[f"{side}_seg_with_prompt_{bucket}"] += 1
                first = True
                for j in good:
                    cc = ((j.get("flags") or {}).get("channel_check") or {})
                    ok = cc.get("status") != "major" and str(cc.get("identity_confidence", "")).lower() != "low"
                    st[f"{side}_chunks"] += 1; st[f"{side}_chunks_ok"] += ok
                    vers[j.get("version")] += 1
                    sp = j["system_prompt"]; n = int((j.get("tokens") or {}).get("system_prompt", -1)); k = int(j.get("chunk_index", 0))
                    sec = j.get("sections") or {}; tk = j.get("tokens") or {}
                    st["persona_nonempty" if (sec.get("persona") or "").strip() else "persona_empty"] += 1
                    for key in (side, "all", "chunk0" if k == 0 else "chunk>0"):
                        tok[key].append(n); words[key].append(len(sp.split()))
                    ptok["persona"].append(int(tk.get("persona", 0) or 0)); ptok["starting_context"].append(int(tk.get("starting_context", 0) or 0))
                    if k > 0:
                        ptok["memory(chunk>0)"].append(int(tk.get("memory", 0) or 0))
                    if first:
                        first = False
                        names[name_of(cc.get("a_speaker")) or "(unnamed)"] += 1
                        s = (sec.get("starting_context") or "").lower()
                        r = ("just met" if ("just met" in s or "had not met" in s or "first conversation" in s) else
                             "friend" if "friend" in s else "partner/spouse" if re.search(r"partner|wife|husband|girlfriend|boyfriend", s) else
                             "family" if re.search(r"sister|brother|mother|father|mum|mom|dad|cousin", s) else "other")
                        rel_[r] += 1
                    if n <= CAP and ok:
                        c = cands[(side, "c0" if k == 0 else "cN", bucket == "none")]
                        if len(c) < 400:
                            c.append({"dataset": "Aloe handover", "side": side, "key": f"{rel.split('/', 2)[2]} · {spk} · chunk {k}", "chunk": k,
                                      "bucket": bucket, "tokens": n, "words": len(sp.split()), "who": cc.get("a_speaker"), "text": sp})
    random.seed(2)
    ex, used = [], set()
    plan = [("user", "c0", True), ("assistant", "c0", True), ("user", "cN", True), ("assistant", "cN", True), ("user", "c0", True),
            ("assistant", "c0", True), ("user", "cN", True), ("assistant", "cN", True), ("user", "c0", False), ("assistant", "cN", False)]
    for p in plan:
        c = [x for x in cands[p] if x["key"].split(" · ")[0] not in used] or [x for x in cands[(p[0], p[1], not p[2])] if x["key"].split(" · ")[0] not in used]
        x = random.choice(c); used.add(x["key"].split(" · ")[0]); ex.append(x)
    return {"segments": len(rels), "counts": dict(st), "versions": vers.most_common(), "tokens": {k: summ(v) for k, v in tok.items()},
            "words": {k: summ(v) for k, v in words.items()}, "section_tokens": {k: summ(v) for k, v in ptok.items()},
            "names": names.most_common(15), "relationship": rel_.most_common()}, ex


if __name__ == "__main__":
    A, aex = aloe()
    print(json.dumps(A)[:3000], flush=True)
    S, pex = ppm()
    R = json.load(open(f"{P}/sidecars/REPORT.json"))
    json.dump({"ppm": S, "ppm_report": {"by_dataset": R["by_dataset"], "status": R["status"]}, "aloe": A, "examples": pex + aex},
              open(f"{OUT}/data.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(S)[:3000])
