"""generates code/train_aloe_v3.py = code/train_aloe_m1vox.py (job 22335's trainer, channel A Voxtral) + the v2 changes (train_aloe_v2.py, job
22383) + the v3 data; every edit is an exact, asserted string replacement (re-run to regenerate)."""
C = "/data/home/amartyavarma/ppm/aloe-v3-ft/code"
s = open(f"{C}/train_aloe_m1vox.py").read()
E = []


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:120])
    s = s.replace(old, new)
    E.append(old[:60])


DOC = '''"""aloe-v3-ft TRAINER (10-17, user spec: "for finetuning data our row limit is much longer - it should be up to 18 minutes, and we shouldn't
apply the mid-call personas"; coordinator: channel A = Voxtral as aloe-m1vox-ft / job 22335) = aloe-v2-ft (job 22383: train_aloe_v2.py + data_aloe_v2
+ aloe_v2.env, init PPM local-11488) EXACTLY, with channel A switched to Voxtral exactly as aloe-m1vox-ft and whole-call rows. Built from
aloe-m1vox-ft's train_aloe_m1vox.py (= 21980's maths + the VOX edits) with the v2 changes and the v3 data (tools/make_trainer.py, asserted edits):
  DATA      data_aloe_v3: ONE row per call = frames 0..min(T, 13,500) (18 min) of the whole call (v3 stores v3n0..v3n3, Model 1's 8,229 calls / split
            / store assignment, Model 1's row rules; v2 had Model 1's 10-min chunk 0); both speakers as B, held-out = v2's 162 calls x 2. Plan = v2's
            rule; the (call, orientation) sequence is ASSERTED equal to v2's plan at start (same calls, same order, same eval list).
            System prompt = v2: no text; B's own voice clip on v2's seeded 30 % draw, source ALOE_V3_CLIP_SRC (v2 rule default / actor alternative).
  CHANNEL A Voxtral latents of the partner (frozen Voxtral Realtime encoder, up to 18 min of audio per row) through the step-1972 ASR adapter
            (trained, group vadapter lr 5e-4) x ONE learnable global scalar s = exp(theta) (group vgain lr 5e-4, theta init log(0.0030930437));
            A has no text / Mimi / prefix clip. compile_decoder ON.
  v2 KNOBS  decoder K 1 on a seeded ~10 % of B's coded frames (light_frac 0.1, ratio band [0.09, 0.11] asserted every step); qqa LRs; no pose
            parts built (nopose); bf16; one epoch at 64 items / step, warmup round(10 x steps / 306), linear decay to exactly 0; 5 evenly spaced
            node-local exports + final (node-local + consolidated /data).
  CONFIG    resolved CFG diffed against v2's logged cfg (plan/ref_cfg_v2_22383.json) at start; a difference outside the allowed keys aborts.
  TEST      QQA_TEST_STEPS > 0 stops after that many steps without any export; ALOE_TEST_LONGEST=1 (test only) gives each node's ranks the node's
            LONGEST items first (memory check at 13,500 frames), from the node's own items so their A FLACs are staged.
Original aloe-m1vox-ft doc follows.
'''
rep('"""aloe-m1vox-ft TRAINER (10-16, user:', DOC + 'aloe-m1vox-ft TRAINER (10-16, user:')
rep("import data_aloe_m1vox as A  # noqa: E402", "import data_aloe_v3 as A  # noqa: E402")
rep('''    "local_export_root": E("ALOE_LOCAL_EXPORT_ROOT"), "stores": A.STORES, "flac_root": A.FLAC_ROOT, "m1vox_index": A.INDEX,''',
    '''    "local_export_root": E("ALOE_LOCAL_EXPORT_ROOT"), "stores": A.STORES, "flac_root": A.FLAC_ROOT, "row_frames": A.ROW_FRAMES, "budget": A.BUDGET,
    # v2 keys (train_aloe_v2.py) + v3
    "eval_data": A.eval_cfg(), "model_nopose": True, "whole_calls_max_frames": A.ROW_FRAMES, "ref_cfg": E("ALOE_REF_CFG"),
    "light_ratio_lo": float(E("ALOE_LIGHT_RATIO_LO", "0.09")), "light_ratio_hi": float(E("ALOE_LIGHT_RATIO_HI", "0.11")),
    "rowcache": os.environ.get("ALOE_ROWCACHE", ""), "test_longest": E("ALOE_TEST_LONGEST", "0") == "1",''')
rep('''CFG["vox_calib_rows"] = int(E("VOX_CALIB_ROWS", "1"))''', '''CFG["vox_calib_rows"] = int(E("VOX_CALIB_ROWS"))
assert not CFG["test_longest"] or CFG["test_steps"] > 0, "ALOE_TEST_LONGEST is a test-only reordering (needs QQA_TEST_STEPS > 0)"''')

# ---- resolved-config diff vs v2 (train_aloe_v2.cfg_diff, reference = v2's logged cfg)
CFGDIFF = '''
# ------------------------------------------------------------------------------------------------ resolved-config diff vs v2 (22383)
CFG_ALLOWED = (                                                   # run dirs + the v3 data + channel A Voxtral (+ test knobs); any other difference aborts
    "run_tag", "log_root", "plan_dir", "control_dir", "export", "local_export_root",
    "stores", "flac_root", "row_frames", "budget", "whole_calls_max_frames", "first_chunks_only", "ref_cfg", "rowcache", "data.clip_src",
    "vadapter.", "vox_a_norm", "vox_a_scale_init", "vox_a_scale_learn", "vox_model", "vox_attn", "vox_calib_rows", "lr.vadapter", "lr.vgain",
    "clip.vadapter", "clip.vgain", "test_steps", "test_no_main_opt", "test_longest")


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = json.loads(json.dumps(v, default=str))
    return out


def cfg_diff():
    ref = _flat(json.load(open(CFG["ref_cfg"])))
    cur = _flat(json.loads(json.dumps(CFG, default=str)))
    rows, bad = [], []
    for k in sorted(set(ref) | set(cur)):
        a, b = ref.get(k, "<absent>"), cur.get(k, "<absent>")
        if k.startswith("eval_data.") and b == cur.get("data." + k[len("eval_data."):], "<absent>") and (a == ref.get("data." + k[len("eval_data."):], "<absent>") or a == "<absent>"):
            continue                                              # eval cfg = the training data cfg except the keys shown (= v2's rule)
        if a != b:
            ok = any(k == x or (x.endswith(".") and k.startswith(x)) or k.startswith(x + ".") for x in CFG_ALLOWED)
            rows.append(f"  {'ok ' if ok else 'BAD'} {k}: v2 {a} -> v3 {b}")
            if not ok:
                bad.append(k)
    log(f"[cfg-diff] resolved config vs v2 22383 ({CFG['ref_cfg']}): {len(rows)} keys differ, every other key identical:\\n" + "\\n".join(rows))
    assert not bad, ("config differs from v2 outside the allowed keys", bad)


# ------------------------------------------------------------------------------------------------ main
STOP = {"flag": False}'''
rep('''
# ------------------------------------------------------------------------------------------------ main
STOP = {"flag": False}''', CFGDIFF)

# ---- main(): header log + cfg diff (replaces m1vox's informative diff vs 21980)
i0 = s.index('    log(f"aloe-m1vox-ft {CFG[\'run_tag\']}: world {WORLD}')
i1 = s.index('    # ---- data = 21980\'s (data_aloe plan')
s = s[:i0] + '''    log(f"aloe-v3-ft {CFG['run_tag']}: world {WORLD} (lw {LW}); host {HOST}; cfg {json.dumps(CFG, default=str)}")
    cfg_diff()
''' + s[i1:]
E.append("header+cfgdiff")

# ---- data section: v3 plan + = v2 checks (replaces m1vox's = 21980 checks)
i0 = s.index('    # ---- data = 21980\'s (data_aloe plan')
i1 = s.index('    # ---- model\n')
DATA = '''    # ---- data (v3 whole-call stores; plan = v2's rule, identical on every rank; one row per rank per step)
    import importlib.util
    PL = A.make_plan(WORLD)
    assert PL["steps"] == CFG["steps"], ("QQA_STEPS != the plan's steps (one epoch over the doubled calls)", CFG["steps"], PL["steps"])
    stx = A.Stores()
    for k in range(A.NST):
        stx.open(k)                                               # staged store digests == plan/STORES.json (asserted)
    rows, ev_items = A.rank_items(PL, RANK, WORLD)
    fallback = PL["items"][0]
    hosts_ = os.environ["PPM_HOSTS"].split(",")                   # node i (ranks 8i..8i+7) = the i-th host sorted -V = staging node i
    assert HOST == hosts_[RANK // LW], ("rank -> host order differs from the staging order", HOST, hosts_[RANK // LW])
    assert os.path.isfile(f"{A.FLAC_ROOT}/_ledger/DONE_node{RANK // LW}.json"), ("this node's A FLACs not staged + verified", RANK // LW)
    if CFG["test_longest"]:                                       # TEST ONLY: this node's longest items first (its own items: FLACs staged here)
        node_, j_ = RANK // LW, RANK % LW
        mine = [PL["items"][t * WORLD + node_ * LW + jj] for t in range(PL["steps"]) for jj in range(LW) if t * WORLD + node_ * LW + jj < len(PL["items"])]
        lens = {k: stx.open(k)[0]["length"] for k in range(A.NST)}
        mine.sort(key=lambda it: -int(lens[it[0]][it[1]]))
        rows = [mine[t * LW + j_] for t in range(CFG["test_steps"])] + rows[CFG["test_steps"]:]
        tl = allsum([float(sum(int(lens[it[0]][it[1]]) for it in rows[:CFG["test_steps"]])), float(min(int(lens[it[0]][it[1]]) for it in rows[:CFG["test_steps"]]))], dev)
        log(f"[test] ALOE_TEST_LONGEST: steps 0..{CFG['test_steps'] - 1} take each node's longest items (mean frames {tl[0] / (WORLD * CFG['test_steps']):.0f})")
    ndum = sum(x is None for x in rows)
    nconv = allsum([float(len(rows) - ndum), float(len(ev_items)), float(ndum)], dev)
    assert CFG["test_longest"] or (int(nconv[0]) == len(PL["items"]) and int(nconv[1]) == len(PL["eval"])), nconv
    dg = hashlib.sha256(json.dumps(PL["items"]).encode()).hexdigest()[:16]
    dga = [None] * WORLD; dist.all_gather_object(dga, dg)
    assert len(set(dga)) == 1, "plan differs across ranks"
    n_fr = [int(stx.open(k)[0]["length"][r]) for k, r, _ in PL["items"]]
    log(f"[data] v3 plan {dg}: {PL['n_train_rows']:,} train calls x 2 orientations = {len(PL['items']):,} micros in {CFG['steps']} steps x {WORLD} "
        f"ranks (1 whole-call row / rank / step; {int(nconv[2])} dummy micros on the last step); eval {PL['n_eval_rows']} calls x 2 = {len(PL['eval'])} "
        f"micros; tail rows (voice-clip source only) {PL['n_tail']}; frames / item mean {np.mean(n_fr):.0f} max {max(n_fr)} "
        f"({sum(n_fr) / 2 / 12.5 / 3600:,.1f} h per orientation); data cfg {CFG['data']}")
    # ---- = v2 checks: the (call, orientation) sequence of the plan and the eval list == v2's (v2's data module on Model 1's stores), and
    #      every rank's step-0 item: v3 micro == v2's micro of the same call on its first 7,500 frames (B targets / lanes / codes; prefix may
    #      differ only by the voice-clip source), tests/bitident_v3.py does the full check offline
    ref_dm = "/data/home/amartyavarma/ppm/aloe-v2-ft/code/data_aloe_v2.py"
    spec = importlib.util.spec_from_file_location("data_aloe_v2_ref", ref_dm)
    _sp = list(sys.path)
    R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
    sys.path[:] = _sp                                             # v2's code dir must not shadow any later (worker-side) import
    PR = R.make_plan(WORLD, roots=R.SRC_STORES)
    import pyarrow.parquet as pq
    conv_of = [pq.read_table(f"{R.SRC_STORES[k]}/rows.parquet", columns=["src_conversation"]).to_pydict()["src_conversation"] for k in range(4)]
    v2_calls = [(conv_of[k][r], s_) for k, r, s_ in PR["items"]]
    v2_ev = [(conv_of[k][r], s_) for k, r, s_ in PR["eval"]]
    v2dg = hashlib.sha256(json.dumps(PR["items"]).encode()).hexdigest()[:16]
    same_seq = [tuple(x) for x in PL["item_calls"]] == v2_calls and [tuple(x) for x in PL["eval_calls"]] == v2_ev and PR["steps"] == PL["steps"]
    log(f"[same-as-v2] v2 plan digest (fresh, v2 data module) {v2dg} (v2 logged ba2d70729ea437ab); (call, orientation) sequence of all "
        f"{len(PL['items']):,} items == v2's: {same_seq}; eval list ({len(PL['eval'])}) == v2's; steps {PL['steps']} == v2 {PR['steps']}")
    assert v2dg == "ba2d70729ea437ab" and same_seq, "v3 plan (calls, orientations, order) != v2's"
    it0 = rows[0] if rows[0] is not None else fallback
    k0, r0, s0 = it0
    call0 = stx.open(k0)[1][r0]["row_key"]                        # = Model 1's chunk-0 key of the same call (v3 keeps it)
    st2 = R.Stores(R.SRC_STORES); n2, rr2, sib2 = st2.open(k0)    # same store index k (v3n<k> = m1n<k>)
    r2 = [i for i, c in enumerate(conv_of[k0]) if f"ALOE/s2handover-none/{c}/c00" == call0 and int(rr2[i]["chunk_index"]) == 0]
    assert len(r2) == 1, (call0, r2)
    row2 = rr2[r2[0]]
    assert row2["row_key"] == call0
    m_ref = R.speech_micro_v2(row2, int(row2["row"]), sib2, R.aloe_cfg(), RANK, s0)
    m_new = stx.conv(it0, CFG["data"], RANK, audio=False, vox=False)
    n1, Pr, Pn = int(row2["n_frames"]), int(m_ref["P"]), int(m_new["P"])
    f_r, f_n = m_ref["sel"] - Pr, m_new["sel"] - Pn
    k_r, k_n = f_r < n1 - 2, f_n < n1 - 2
    chk = {"lanes": np.array_equal(m_ref["ids"][:, Pr:Pr + n1 - 1], m_new["ids"][:, Pn:Pn + n1 - 1]),
           "codes": np.array_equal(m_ref["depth"][:, Pr:Pr + n1 + D.CD], m_new["depth"][:, Pn:Pn + n1 + D.CD]),
           "cb0": np.array_equal(np.asarray(m_ref["cb0"])[Pr:Pr + n1 + D.CD], np.asarray(m_new["cb0"])[Pn:Pn + n1 + D.CD]),
           "targets": np.array_equal(f_r[k_r], f_n[k_n]) and all(np.array_equal(np.asarray(m_ref[q])[k_r], np.asarray(m_new[q])[k_n]) for q in ("y", "word", "pad", "eob", "nw")),
           "light": np.array_equal(m_ref["light"] - Pr, (m_new["light"] - Pn)[(m_new["light"] - Pn) < n1 + D.CD])}
    allc = [None] * WORLD; dist.all_gather_object(allc, (all(chk.values()), Pr == Pn, n1, int(m_new["P"]) + int(stx.open(k0)[0]["length"][r0])))
    log(f"[same-as-v2] step-0 item of all {WORLD} ranks: v3 micro == v2 micro of the same call on its first 7,500 frames (lanes to frame 7,498, "
        f"codes, cb0, B targets, light): {sum(c[0] for c in allc)} / {WORLD}; prefix identical on {sum(c[1] for c in allc)} ranks (else the voice "
        f"clip source differs: v2 later chunk / v3 tail); rank 0 checks {chk}")
    assert all(c[0] for c in allc), [c for c in allc if not c[0]][:3]
    del m_ref, m_new, R, PR, st2
'''
s = s[:i0] + DATA + s[i1:]
E.append("data section")

# ---- eval micros: v2's eval cfg + held-out voice-clip log
rep('''    ev = A.eval_micros(ev_items, CFG["data"], RANK)''',
    '''    ev = A.eval_micros(ev_items, CFG["eval_data"], RANK)
    evc = allsum([float(sum(int(m["voice_clip"]) for m in ev)), float(sum(int(m["voice_drawn"]) for m in ev)), float(len(ev))], dev)
    log(f"[eval] v3 held-out voice clips ({CFG['data']['clip_src']} source): drawn {int(evc[1])} / {int(evc[2])}, realised {int(evc[0])} / {int(evc[2])} = {evc[0] / max(evc[2], 1):.4f}")''')

# ---- training loop: v2 light-ratio band + cumulative voice clips; v3 per-step max frames
rep('''    times = []
    stopped = None
    for step in range(CFG["steps"]):''', '''    times = []
    stopped = None
    vc_cum = [0.0, 0.0, 0.0]                                      # v2: realised voice clips, drawn, real micros (cumulative over the run)
    for step in range(CFG["steps"]):''')
rep('''        tot = allsum([float(x) for x in oc[:3]], dev)''', '''        tot = allsum([float(x) for x in oc[:3]], dev)
        lr_ratio = tot[2] / max(tot[1], 1.0)                      # v2 (light_frac 0.1): the selected frames = ~10 % of B's coded (cb0) frames
        assert CFG["light_ratio_lo"] <= lr_ratio <= CFG["light_ratio_hi"], ("light frames / B coded frames outside the 10 % band", lr_ratio, tot)
        vc = allsum([float(b.get("voice_clip", 0)), float(b.get("voice_drawn", 0)), 0.0 if b.get("dummy", 0) else 1.0], dev)
        vc_cum = [x + y for x, y in zip(vc_cum, vc)]''')
rep('''        pk = torch.tensor([torch.cuda.max_memory_allocated() / GiB, dt, wait, t_micro, t_enc], device=dev)''',
    '''        pk = torch.tensor([torch.cuda.max_memory_allocated() / GiB, dt, wait, t_micro, t_enc, float(len(b["pid"])), torch.cuda.max_memory_reserved() / GiB], device=dev)''')
rep('''                  "train/light_frames": tot[2]})''', '''                  "train/light_frames": tot[2], "train/light_ratio": lr_ratio, "train/voice_clip_frac_cum": vc_cum[0] / max(vc_cum[2], 1.0),
                  "train/voice_drawn_frac_cum": vc_cum[1] / max(vc_cum[2], 1.0), "train/max_T": float(pk[5]), "train/peak_reserved_gib": float(pk[6])})''')
rep('''            + f" | ETA {eta / 60:.1f} min")''', '''            + f" | T max {int(pk[5])} reserved {float(pk[6]):.1f} GiB light {lr_ratio:.4f} clip {vc_cum[0] / max(vc_cum[2], 1.0):.4f} | ETA {eta / 60:.1f} min")''')
rep('''"src": "aloe-ft train_aloe export_local"''', '''"src": "aloe-v3-ft train_aloe_v3 export_local"''')
open(f"{C}/train_aloe_v3.py", "w").write(s)
print("train_aloe_v3.py written;", len(E), "edits")
