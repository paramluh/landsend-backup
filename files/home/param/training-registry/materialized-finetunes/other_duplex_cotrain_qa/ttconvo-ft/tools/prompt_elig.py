"""prompt-rule eligibility over the FT plan rows of ONE store slot (run on a host holding it): voice clip (data_ppm.voice_clips, PPM's rule)
and persona (sidecar status ok + prose) availability, independent of the draws; + speaker-label consistency per episode."""
import json, sys, collections
sys.path.insert(0, "/data/home/amartyavarma/ppm/ttconvo-ft/code")
import data_ppm as D
slot = sys.argv[1]
P = json.load(open("/data/home/amartyavarma/ppm/ttconvo-ft/plan/PLAN.json"))
k = P["slots"].index(slot)
rows = sorted({r for rr in P["ranks"] for s, r in rr if s == k})
cfg = dict(D.default_cfg(seed=20261011), voice_p=0.8, persona_p=0.2)
node = D.R3.load_node(f"/scratch/amartyavarma/ttconvo-ft/stores/{slot}"); st = D.R3.Rows(node); sib = D.SPR.SibIndex(node, st)
ps = D.PER.PersonaSidecar(node, root=f"/data/home/amartyavarma/ppm/ttconvo-ft/persona/{slot}")
c = collections.Counter(); spk = collections.defaultdict(set)
for i, r in enumerate(rows):
    row = st[r]
    ch = D.rec_chunks(sib, row); ci = int(row["chunk_index"]); nch = max(ch) + 1
    later2 = any(x > ci and x >= nch / 2.0 for x in ch)
    c["n"] += 1; c["last_chunk"] += int(ci == max(ch)); c["has_later_2nd_half_chunk"] += int(later2)
    vc = D.voice_clips(row, sib, cfg)
    a, b = vc[0] is not None, vc[1] is not None
    c["clip_any"] += int(a or b); c["clip_both"] += int(a and b); c["clip_user"] += int(a); c["clip_asst"] += int(b)
    c["later2_but_no_clip"] += int(later2 and not (a or b))
    for side in ("user", "assistant"):
        _, lat = sib.siblings(row, side)
        c[f"sib_later_{side}"] += int(any(x > ci and x >= nch / 2.0 for x, _, _ in lat))
    pers = ps.get(r, row["row_key"])
    pa = [bool(D.PT.avail(pers.get(s_))) for s_ in ("user", "assistant")]
    c["persona_any"] += int(any(pa)); c["persona_both"] += int(all(pa))
    spk[D.SPR.rec_key(row["row_key"])].add((row["user_speaker"], row["assistant_speaker"]))
c["episodes"] = len(spk); c["episodes_label_inconsistent"] = sum(1 for v in spk.values() if len(v) > 1)
n = c["n"]
print(json.dumps({"slot": slot, **dict(c), **{f"frac_{x}": round(c[x] / n, 4) for x in ("last_chunk", "has_later_2nd_half_chunk", "clip_any", "clip_both", "persona_any", "persona_both", "later2_but_no_clip", "sib_later_user", "sib_later_assistant")}}))
