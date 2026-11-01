"""B's SYSTEM PROMPT for the aloe-m1vox live demo = the trainer's prefix builder (data_aloe_vox.podcast_prefix_vox: data_ppm's podcast
prefix with channel A's voice clip dropped), driven by the page's choices:
  * B's persona text (a held-out Aloe preset or free text): the persona sidecar format 'Persona\\n...\\n\\nStarting context\\n...' is parsed
    back into sections and rendered with persona_text_s3q.block_text (the training renderer) between <sop> / <eop> on channel B;
  * channel A's persona: none (the live user has no sidecar persona; training rows without one have no A block);
  * B's voice clip (optional): the preset speaker's own training clip (data_ppm.voice_clips: a later chunk in the recording's second half,
    audio codes only, blank lane, 3-10 s) or an upload (mono 24 kHz, -23 LUFS, Mimi-encoded, <= 10 s, >= 3 s);
  * cap 2,000 positions (crop_blocks), sysprompt.build_prefix.
PRESETS = held-out Aloe rows (split 'eval' of Model 1's handover stores m1n0..m1n3 on /data, never trained on), first chunks only (c00: no
'Earlier conversation memory'), one preset per (row, B side) with a named B persona.
SELFTEST: for held-out rows, this builder with the row's own choices (both personas, the drawn clip) == podcast_prefix_vox exactly.
10-17 (aloe-v2vox): + SELFTEST_V2: for held-out first-chunk rows, this builder with NO text and the v2-drawn B clip (or none) == the aloe-v2 /
v2vox trainer's prefix (data_aloe_v2.prefix_v2: voice clip only, no persona / context / memory) exactly; and the voice-clip-only prompt mode
('' persona + a clip) is what the server builds for a checkpoint whose CK_LABELS.json entry says prompt 'voice_clip_only'."""
import re
import sys

AC = "/home/param/websites/pose-duplex-engine/deps/aloe-m1vox-ft-code"
for _p in (f"{AC}/vendor", f"{AC}/lib", AC):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
import numpy as np  # noqa: E402

import data_aloe as AL  # noqa: E402
import data_aloe_vox as V  # noqa: E402

D = AL.D
PT, SPR = D.PT, D.SPR
CFG = AL.aloe_cfg()
HEAD2KEY = {v.lower(): k for k, v in PT.HEADER.items()}


def parse_sections(text):
    """'Persona\\n...\\n\\nStarting context\\n...' -> {section: prose}; text without headers = the persona section."""
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return {}
    out, cur, buf = {}, None, []
    for ln in text.split("\n"):
        k = HEAD2KEY.get(ln.strip().lower())
        if k is not None:
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            cur, buf = k, []
        else:
            buf.append(ln)
    if cur is None:
        return {"persona": text}
    out[cur] = "\n".join(buf).strip()
    return {k: v for k, v in out.items() if v}


def render_text(av):
    return PT.block_text(av, [k for k in PT.SECTIONS if k in av])


def build_prefix(b_text, clip=None, a_text=None, a_clip=None, prose=False):
    """-> (prefix dict of sysprompt.build_prefix, info). clip / a_clip: data_ppm.voice_clips dicts or upload dicts (w0, L, src_row, src_side,
    blank_lane). a_clip exists only for the selftest's comparison with podcast_prefix_vox (which drops it: always None).
    10-24 prose=True (prompt mode 'persona_prose' = the vox17127-persona FT, aloe-ft-32-samsep/persona/code/data_aloe_persona.prefix_persona):
    B's block = <sop> + b_text EXACTLY as written (plain prose, no section labels, no starting context) + <eop> after the clip on channel B;
    empty / blank text -> no block; NEVER cropped (too long -> AssertionError, shown to the user)."""
    D._tok_init()
    segs = []
    if clip is not None:
        segs.append({"kind": "speech", "track": 1, **clip})
    fixed = 5 + sum(int(s["L"]) + 5 for s in segs)
    assert fixed <= int(CFG["sp_cap"])
    if prose:
        txt = b_text if isinstance(b_text, str) and b_text.strip() else ""
        if txt:
            blk = [PT.SOP] + PT.encode(txt) + [PT.EOP]
            assert len(blk) <= int(CFG["sp_cap"]) - fixed, (f"persona too long: {len(blk)} tokens, at most {int(CFG['sp_cap']) - fixed} fit "
                                                            "with this voice clip (never truncated)")
            segs.append({"kind": "text", "track": 1, "ids": blk})
        spx = SPR.build_prefix({}, segs, {"new_word": CFG["new_word"]})
        assert 5 <= spx["P"] <= int(CFG["sp_cap"]), spx["P"]
        assert (spx["depth"][0] < 0).all(), "no A codes in the prefix"
        return spx, {"P": int(spx["P"]), "persona_tokens": [len(blk)] if txt else [], "persona_cut": 0,
                     "clip_frames": int(clip["L"]) if clip is not None else 0, "persona_format": "prose (no labels)"}
    blocks, chans = [], []
    for c, txt in ((0, a_text), (1, b_text)):
        av = parse_sections(txt) if isinstance(txt, str) else (txt or {})
        if av:
            blocks.append([PT.SOP] + PT.encode(render_text(av)) + [PT.EOP]); chans.append(c)
    blocks2, cut = D.crop_blocks(blocks, int(CFG["sp_cap"]) - fixed)
    segs += [{"kind": "text", "track": c, "ids": b} for c, b in zip(chans, blocks2) if b is not None]
    if not segs:
        segs = [{"kind": "text", "track": 1, "ids": []}]
    spx = SPR.build_prefix({}, segs, {"new_word": CFG["new_word"]})
    assert 5 <= spx["P"] <= int(CFG["sp_cap"]), spx["P"]
    assert (spx["depth"][0] < 0).all(), "no A codes in the prefix"
    info = {"P": int(spx["P"]), "persona_tokens": [len(b) for b in blocks2 if b is not None], "persona_cut": int(cut),
            "clip_frames": int(clip["L"]) if clip is not None else 0}
    return spx, info


# 10-24 (a16861w480 = the SAM-sep FT): a checkpoint's CK_LABELS.json entry may say "preset_codes": "samsep" -> the presets are built from the
# sam-sep stores (same rows / words / personas / windows as the handover stores, 122 purged conversations dropped and rows renumbered; both
# lanes' Mimi codes = the SAM-ISOLATED streams; aloe-ft-32-samsep/data/VERIFY.md), i.e. the B voice clip codes that FT trained on.
# Default 'handover' = AL.SRC_STORES, unchanged.
CODES = {"handover": (AL.SRC_STORES, AL.DIGESTS),
         "samsep": ([f"/home/param/websites/pose-duplex-engine/deps/qwen-1ch-data/s2h-m1-samsep/stores/m1n{k}" for k in range(4)],
                    ["7318ffb3ea75cbcc3ff48b63b6cf8b88", "31ecb2273542310096936c0baf49efa8", "1759200f7818c6017dfb6a46e37fa5df",
                     "611b2e7edb1503cc600a6489e49dcf40"])}


# 10-30 pdxhost (USER "host the 1-epoch version": job 33111 'pdxraw' = the pose-duplex FT on RAW codes, POSEDUP_RAW_CODES=1): "raw" = the sam-sep
# stores' rows (same preset ids / words / personas / voice-clip choice) whose codes_user / codes_assistant are the MAPPED raw s2h-m1 row's
# (ALOE_SAMSEP_ROW_MAP new -> old; raw-lane mimi-stream-v5 = what that run trained on as B, incl. the voice clip cut from the later-chunk
# sibling row). Same rule and asserts as poseduplex-raw/code/data_aloe_posedup.RawRows. "handover" / "samsep" unchanged.
CODES["raw"] = CODES["samsep"]
RAW_STORES = "/home/param/websites/pose-duplex-engine/deps/qwen-1ch-data/s2h-m1/stores"
RAW_ROW_MAP = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/data/row_map.json"
RAW_DIGESTS = ["c1c3b2517152741137cc4bb72dd6bdbb", "09bdd43fef84574175caacf4abdddd1e", "ecb1c731e8989b3f2e1b5266b51ea0fd",
               "3b16fb8dd2660a2e8c1fe32c0d361d7b"]
RAW_ROW_COLS = ("row_key", "n_frames", "chunk_index", "frame_start", "split", "user_speaker_dir", "assistant_speaker_dir", "src_conversation")


class _RawRows:
    """= data_aloe_posedup.RawRows: sam-sep store k's rows carrying the mapped raw row's code arrays (mmap, in place)."""

    def __init__(self, k, node, rows, new_digest):
        import json
        import pyarrow.parquet as pq
        j = json.load(open(RAW_ROW_MAP))
        assert j["format"] == "aloe-ft-32-samsep-row-map-v1", j.get("format")
        assert j["old_digest"][f"m1n{k}"] == RAW_DIGESTS[k] and j["new_digest"][f"m1n{k}"] == new_digest, ("row map digests", k)
        mp = j["map"][f"m1n{k}"]
        no = {int(o): int(n) for o, n in mp["new_of_old"].items()}
        R, n_old = len(no), int(mp["old_rows"])
        assert sorted(no.values()) == list(range(R)) and all(0 <= o < n_old for o in no), ("row map bijection", k)
        self.oon = np.full(R, -1, np.int64)
        for o, n in no.items():
            self.oon[n] = o
        self.k, self.rows = int(k), rows
        assert len(self.oon) == len(rows) == len(node["length"]), ("row map rows != sam-sep store rows", k)
        self.root = f"{RAW_STORES}/m1n{k}"
        self.raw = D.R3.load_node(self.root)
        assert self.raw["digest"] == RAW_DIGESTS[k], ("raw store digest", self.root, self.raw["digest"])
        assert len(self.raw["length"]) == n_old, ("raw store rows != row map old_rows", k)
        cols = ["row", *RAW_ROW_COLS]
        tn = pq.read_table(f"{node['flat_root']}/rows.parquet", columns=cols).to_pydict()
        to = pq.read_table(f"{self.root}/rows.parquet", columns=cols).to_pydict()
        assert tn["row"] == list(range(R)) and to["row"] == list(range(n_old)), "rows.parquet row ids"
        ol = self.oon.tolist()
        for c in RAW_ROW_COLS:
            assert [to[c][o] for o in ol] == tn[c], ("mapped raw row != sam-sep row", k, c)
        assert np.array_equal(self.raw["length"][self.oon], node["length"]), ("mapped raw n_frames != sam-sep", k)
        self.cu, self.ca = {}, {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        o = int(self.oon[int(i)])
        s, r = int(self.raw["shard_of"][o]), int(self.raw["row_of"][o])
        if s not in self.cu:
            sh = f"{self.root}/{self.raw['shards'][s]}"
            self.cu[s] = np.load(f"{sh}.codes_user.npy", mmap_mode="r")
            self.ca[s] = np.load(f"{sh}.codes_assistant.npy", mmap_mode="r")
        cu, ca = self.cu[s][r], self.ca[s][r]
        assert int(row["n_frames"]) == int(self.raw["length"][o]), ("raw n_frames", row["row_key"])
        for sd, c in (("user", cu), ("assistant", ca)):
            sm = row[f"codes_{sd}"]
            assert c.shape == sm.shape and c.dtype == sm.dtype, ("raw codes shape / dtype", row["row_key"], sd, c.shape)
            assert np.array_equal(np.asarray(c) >= 0, np.asarray(sm) >= 0), ("raw / sam-sep valid-frame mask differs", row["row_key"], sd)
        out = dict(row)
        out["codes_user"], out["codes_assistant"] = cu, ca
        return out


class _Stores(AL.Stores):
    """AL.Stores with the given stores' own digests (AL.Stores.open asserts Model 1's handover digests). raw=True (10-30): rows (and the
    sibling index's rows, for voice clips) carry the mapped raw s2h-m1 codes (_RawRows)."""

    def __init__(self, roots, digests, raw=False):
        super().__init__(roots)
        self.digests = list(digests)
        self.raw = bool(raw)

    def open(self, k):
        if k not in self.s:
            node = D.R3.load_node(self.roots[k])
            assert node["digest"] == self.digests[k], ("store digest differs", self.roots[k], node["digest"])
            st = D.R3.Rows(node)
            if self.raw:
                st = _RawRows(k, node, st, self.digests[k])
            self.s[k] = (node, st, D.SPR.SibIndex(node, st), D.PER.PersonaSidecar(node))
        return self.s[k]


class Presets:
    def __init__(self, n_max=12, codes="handover", first=None, extra=None):
        roots, digests = CODES[codes]
        self.codes, self.first, self.extra = codes, first, extra
        self.st = _Stores(roots, digests, raw=(codes == "raw"))
        sp = AL.splits(roots)
        self.eval_rows = sp["eval"]
        self.items = []
        seen_names = set()
        for (k, r) in self.eval_rows:
            node, rows, sib, ps = self.st.open(k)
            row = rows[r]
            if int(row["chunk_index"]) != 0:
                continue
            try:
                pers = ps.get(r, row["row_key"])
            except (FileNotFoundError, OSError):
                continue
            for sw in (0, 1):
                side = D.PER.SIDES[1 ^ sw]
                av = PT.avail((pers or {}).get(side))
                if "persona" not in av or "memory" in av:
                    continue
                mname = re.search(r"[Yy]our name is ([A-Z][a-z]+)", av["persona"])
                if not mname or mname.group(1) in seen_names:
                    continue
                clips = D.voice_clips(row, sib, CFG)
                clip = clips[1 ^ sw]
                if clip is None:
                    continue
                seen_names.add(mname.group(1))
                first = re.split(r"(?<=[.!?])\s", av["persona"].strip())
                blurb = next((s for s in first if "name is" not in s), first[0])[:70]
                self.items.append({"id": f"aloe-{k}-{r}-{sw}", "k": k, "r": r, "sw": sw, "name": mname.group(1), "row_key": row["row_key"],
                                   "side": side, "persona": render_text(av), "prose": av["persona"], "clip": clip,
                                   "label": f"{mname.group(1)} · {blurb}{'…' if len(blurb) >= 70 else ''}"})
                if len(self.items) >= n_max:
                    break
            if len(self.items) >= n_max:
                break
        assert self.items, "no held-out presets found"
        # 10-25 (Rosie voice FT): a checkpoint's CK_LABELS 'extra_presets' = [{k, r, sw, name, row_key}] add presets from given store rows (e.g. a
        # speaker FT's held-out calls, in the train split): B = data side SIDES[1 ^ sw], clip = data_ppm.voice_clips (a later chunk of the same
        # recording, the training clip rule), no persona text; listed first, in order; a row without a clip for that side is skipped (logged)
        ex = []
        for e in (extra or []):
            k, r, sw = int(e["k"]), int(e["r"]), int(e["sw"])
            node, rows, sib, ps = self.st.open(k)
            row = rows[r]
            assert row["row_key"] == e["row_key"], ("extra preset row moved", e, row["row_key"])
            clip = D.voice_clips(row, sib, CFG)[1 ^ sw]
            if clip is None:                                           # held-out call without a later 2nd-half chunk in the store: the same
                side = D.PER.SIDES[1 ^ sw]                             # window rule (SPR.window, voice_clips' own length / position draw) on
                rng = np.random.default_rng(np.random.SeedSequence([int(CFG["seed"]), D.h64(row["row_key"]) & 0x7FFFFFFFFFFF, 0x5643]))
                draws = [(int(rng.integers(int(CFG["clip_min"]), int(CFG["clip_max"]) + 1)), float(rng.random())) for _ in range(2)]
                L0, u = draws[1 ^ sw]                                  # this chunk's own lane of that speaker (audio codes only, blank lane)
                wcfg = {x: CFG[x] for x in ("clip_min", "stride", "min_cov", "top_frac")}
                w = SPR.window(row[f"words_{side}"], int(row["n_frames"]), L0, u, wcfg, lo=0)
                if w is not None:
                    clip = dict(w, src="heldout_same_chunk", src_row=row, src_side=side, src_key=row["row_key"], src_chunk=int(row["chunk_index"]),
                                blank_lane=True, side=side, n_chunks=1)
            if clip is None:
                print(f"[presets] extra {e['name']} {e['row_key']}: no voice clip for side {D.PER.SIDES[1 ^ sw]}, skipped", flush=True); continue
            ex.append({"id": f"aloe-{k}-{r}-{sw}", "k": k, "r": r, "sw": sw, "name": e["name"], "row_key": row["row_key"], "side": D.PER.SIDES[1 ^ sw],
                       "persona": "", "prose": "", "clip": clip, "label": f"{e['name']} · held-out call {row['row_key'].split('/')[2][:8]}"})
        self.items = ex + [p for p in self.items if p["id"] not in {x["id"] for x in ex}]
        # 10-24 (Brianna voice FT): a checkpoint's CK_LABELS 'default_preset' id goes first = the page's default preset + voice clip; the set of
        # presets / ids is unchanged (every backend serves the same ids whatever its checkpoint)
        if self.first is not None:                                  # (self.first: the loop above reuses the local name 'first')
            hit = [p for p in self.items if p["id"] == self.first]
            assert hit, ("default_preset not among the presets", self.first, [p["id"] for p in self.items])
            self.items = hit + [p for p in self.items if p["id"] != self.first]
        # 10-24: the sam-sep stores renumber rows, so most preset ids change; the handover ids of the same speakers (pages already open,
        # saved settings) resolve to the same speaker's preset here instead of 'unknown clip'
        self.alias = {}
        if codes != "handover":
            mine = {p["clip"]["src_key"]: p["id"] for p in self.items}
            self.alias = {q["id"]: mine[q["clip"]["src_key"]] for q in Presets(n_max).items if q["clip"]["src_key"] in mine}

    def get(self, pid):
        pid = self.alias.get(pid, pid)
        return next((p for p in self.items if p["id"] == pid), None)

    def public(self, prose=False):
        """prose (prompt mode 'persona_prose'): each preset's persona = its persona prose only, no labels, as that FT trained."""
        return [{"id": p["id"], "label": p["label"], "name": p["name"], "persona": p["prose"] if prose else p["persona"], "row": None,
                 "group": p.get("group", "Held-out Aloe speakers"), "default": i == 0} for i, p in enumerate(self.items)]

    def clips_public(self):
        out = [{"id": "none", "label": "no voice clip", "row": None}]
        for p in self.items:
            out.append({"id": "preset:" + p["id"], "label": f"{p['name']}'s own voice ({p['clip']['L'] / 12.5:.1f} s, " + (f"held-out call {p['row_key'].split('/')[2][:8]})"
                        if p["clip"].get("src") == "heldout_same_chunk" else "the audio of the avatar's own video)" if p["clip"].get("src") == "video_person"
                        else "a later chunk of the same recording)"),
                        "row": None})
        return out

    def default_clip(self):
        return {p["id"]: "preset:" + p["id"] for p in self.items}

    def selftest(self, n=6):
        """this builder == data_aloe_vox.podcast_prefix_vox on held-out rows with the row's own choices (personas of both sides, the drawn
        B clip): ids / depth / present identical."""
        res = []
        for (k, r) in self.eval_rows[:n]:
            node, rows, sib, ps = self.st.open(k)
            row = rows[r]
            try:
                pers = ps.get(r, row["row_key"])
            except (FileNotFoundError, OSError):
                pers = None
            for sw in (0, 1):
                ref, pv = V.podcast_prefix_vox(row, sw, sib, pers, CFG)
                want_voice = D.h64(f"ppm-voice|{int(CFG['seed'])}|{row['row_key']}") % int(CFG["voice_mod"]) == 0
                clips = D.voice_clips(row, sib, CFG) if want_voice else {0: None, 1: None}
                a_av = PT.avail((pers or {}).get(D.PER.SIDES[0 ^ sw])) if pers else {}
                b_av = PT.avail((pers or {}).get(D.PER.SIDES[1 ^ sw])) if pers else {}
                mine, _ = build_prefix(b_av, clips[1 ^ sw], a_av)
                ok = all(np.array_equal(np.asarray(mine[x]), np.asarray(ref[x])) for x in ("ids", "depth", "present")) and mine["P"] == ref["P"]
                res.append({"row": row["row_key"], "sw": sw, "P": int(ref["P"]), "equal": bool(ok)})
        return {"PASS": all(x["equal"] for x in res), "rows": res}


def selftest_v2(presets, n=4):
    """this builder with b_text '' and the v2 trainer's drawn B clip == data_aloe_v2.prefix_v2 (aloe-v2vox-ft/code copy, = aloe-v2-ft's) on
    held-out FIRST-CHUNK rows, both orientations: ids / depth / present / P identical (rows with and without a realised clip)."""
    import importlib.util
    sp0 = list(sys.path)                                          # data_aloe_v2 puts its own dir first on sys.path: restored right after
    try:
        spec = importlib.util.spec_from_file_location("data_aloe_v2_live", "/home/param/websites/pose-duplex-engine/deps/aloe-v2vox-ft-code/data_aloe_v2.py")
        A2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(A2)
    finally:
        sys.path[:] = sp0
    assert A2.D is D, "data_aloe_v2 must use the already-loaded data_ppm (byte-identical copies)"
    cfg2 = A2.aloe_cfg()
    res = []
    for (k, r) in presets.eval_rows:
        nc = sum(x["clip"] for x in res)
        if (nc >= n and len(res) - nc >= n) or len(res) >= 400:
            break
        node, rows, sib, ps = presets.st.open(k)
        row = rows[r]
        if int(row["chunk_index"]) != 0:
            continue
        for sw in (0, 1):
            ref, pv = A2.prefix_v2(row, sw, sib, cfg2)
            clip = D.voice_clips(row, sib, cfg2)[1 ^ sw] if pv["voice_drawn"] else None
            mine, _ = build_prefix("", clip)
            ok = all(np.array_equal(np.asarray(mine[x]), np.asarray(ref[x])) for x in ("ids", "depth", "present")) and mine["P"] == ref["P"]
            res.append({"row": row["row_key"], "sw": sw, "P": int(ref["P"]), "clip": int(pv["voice_clip"]), "equal": bool(ok)})
    nc = sum(x["clip"] for x in res)
    return {"PASS": nc >= 1 and len(res) > nc and all(x["equal"] for x in res), "items": len(res), "with_clip": nc,
            "rows": [x for x in res if x["clip"]][:n] + [x for x in res if not x["clip"]][:n]}


def upload_clip(codes):
    """Mimi codes [32, L] of an uploaded clip (-23 LUFS, <= 125 frames) -> a clip dict for build_prefix (audio codes only, blank lane)."""
    L = int(codes.shape[1])
    return {"w0": 0, "L": L, "src_row": {"codes_user": codes, "codes_assistant": codes}, "src_side": "user", "blank_lane": True, "side": "assistant"}
