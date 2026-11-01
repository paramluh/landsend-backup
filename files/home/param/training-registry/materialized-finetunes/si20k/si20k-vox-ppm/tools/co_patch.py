"""si20k-vox CO-TRAINING patch (user 10-18 PT: + unseen pretraining podcast rows at 1:1), applied by make_trainers.py to the generated trainer
text (exact replacements, each anchor exactly once). Adds: podcast plan / sources / loader / eval (podcast_rows), prep_pod (native A), the
step loop's extra micro (pooled tokens_step normalisers, decoder update after the last micro), separate train / eval summaries
(train_pod_speech, eval_pod_speech), and the step-0 eval WITHOUT the user input (eval_noA_speech)."""

SETUP = '''    # ---- co-training podcast rows (podcast_rows): this host's pretraining rank j's UNSEEN continuation, native A, one micro per si20k micro
    POD = None
    if CFG["podcast"]["on"]:
        PPL = PR.load_plan(CFG["podcast"]["plan"])
        assert HOST in PPL["hosts"], ("this host holds no pretraining slice of the podcast plan: place the run on the plan's hosts", HOST, sorted(PPL["hosts"]))
        src_ = PR.Source(PPL, HOST, LOCAL, CFG["podcast"]["data"])
        assert len(src_.codes) >= CFG["steps"] and len(set(src_.codes)) == len(src_.codes), ("unseen podcast rows for this rank", len(src_.codes), CFG["steps"])
        pk_ = [None] * WORLD; dist.all_gather_object(pk_, (HOST, LOCAL, src_.prod_rank, src_.mode))
        log(f"[podcast] rows per host: " + ", ".join(sorted({f"{x[0]} {x[3]}" for x in pk_})))
        assert len(set(x[2] for x in pk_)) == WORLD, ("two fine-tune ranks map to the same pretraining rank", pk_)
        m0_ = src_.micro(src_.codes[0], audio=False)
        POD = {"src": src_, "plan": PPL}
        log(f"[podcast] co-training ON: {CFG['podcast']['plan']} (sha {CFG['podcast']['plan_sha']}; {PPL['pretrain_run']} stopped at step {PPL['stop_step']}); "
            f"rank 0 = pretraining rank {src_.prod_rank} on {HOST}: {len(src_.codes)} unseen rows, step-0 row {json.loads(m0_['prov'])['row_key']} "
            f"(P {int(m0_['P'])}, T {len(m0_['pid'])}, swap {m0_['swap']}, vin {m0_['vin']}); {LWORLD} si20k + {WORLD} podcast rows per step; A input {CFG['podcast']['a_input']}; "
            f"plan totals {PPL.get('totals')}; staged overlay {PPL.get('staged_overlay')}; this host's rows: {src_.mode} (native slice or staged shards)")
'''

PREP_POD = '''def prep_pod(b, dev, S_text, S_cb0):
    """co-training: one podcast micro (the model's pretraining speech-micro maths, NATIVE A: PPM partner words + Mimi; INTL partner words in
    the user_text slot through the base path) -> device dict = prep() without the VOX keys and without the channel-A emptiness checks."""
    assert b["kind"] == "speech" and b.get("src") == "podcast", (b.get("kind"), b.get("src"))
    if "il_check" in globals():
        il_check(b, user_words_ok=True)
    t = {"kind": "speech", "src": "speech", "segs": [int(x) for x in b["segs"]]}
    for k in ARR:
        t[k] = torch.as_tensor(np.asarray(b[k])).to(dev, non_blocking=True)
    t["pid"] = t["pid"][None]
    w = {k: t[k].float() for k in ("word", "pad", "eob", "nw")}
    S = float(S_text)
    t["wt"] = ((CFG["w_cat_word"] * w["word"] + CFG["w_tim"] * w["pad"] + CFG["w_eob"] * w["eob"] + CFG["w_nw"] * w["nw"]) / S).float()
    t["wn"] = (CFG["w_text"] * w["word"] / S).float()
    t["cb0_w"] = (t["cb0"] >= 0).float() / float(S_cb0)
    t["fwd"] = {"path": "base"} if "il_check" in globals() else {}
    t["dummy"] = bool(b.get("dummy", 0))
    if t["dummy"]:
        t["wt"], t["wn"], t["cb0_w"] = t["wt"] * 0.0, t["wn"] * 0.0, t["cb0_w"] * 0.0
    return t


'''

LOADERS = '''    ldp = iter(DataLoader(PR.PodDS(CFG["podcast"]["plan"], HOST, LOCAL, CFG["steps"], CFG["podcast"]["data"]), batch_size=None, shuffle=False,
                          num_workers=3, prefetch_factor=4, persistent_workers=True, pin_memory=False, worker_init_fn=_wi,
                          collate_fn=lambda x: x)) if POD else None
    evp = PR.eval_micros(POD["plan"], HOST, LOCAL, CFG["podcast"]["eval_data"]) if POD else []
    t_ = torch.tensor([float(len(evp))], device=dev); dist.all_reduce(t_, op=dist.ReduceOp.MAX); n_evp = int(t_)
    if POD:
        nr_ = allsum([float(len(evp))], dev)[0]
        if not evp:                                               # staged host (its slice is gone): no pretraining eval rows -> zero-weight dummies
            evp = [PR.as_micro(POD["src"].micro(POD["src"].codes[0]), POD["src"].codes[0], dummy=True)]
        evp += [dict(evp[0], dummy=1, own=np.zeros_like(evp[0]["own"])) for _ in range(n_evp - len(evp))]
        log(f"[eval] podcast = the pretraining eval_speech rows of this run's {WORLD} pretraining ranks: {int(nr_)} micros ({n_evp} per rank max, dummies pad)")
'''

EVAL_HEAD_OLD = '''    def evaluate(step):
        model.eval()
        t0_ = time.time()
        acc = {0: None, 1: None}
        for m in ev:
            oc = np.asarray(m["own"]).sum(0)
            lat = encode(front, m)
            v, _, _ = run_micro(model, prep(m, dev, max(oc[0], 1.0), max(oc[1], 1.0), lat=lat), rot, dos, grad=False)
'''
EVAL_HEAD_NEW = '''    def evaluate(step, noA=False):
        """noA (step 0): the si20k eval with NO user input (PPM: adapter output x 0, channel A carries nothing; INTL: the base path with
        A = PAD -> emb_A(PAD), the 14B's absent-user rule) -> eval_noA_speech/*. Else + the podcast eval (eval_pod_speech/*) when co-training."""
        model.eval()
        t0_ = time.time()
        acc = {0: None, 1: None}
        if noA and "il_check" not in globals():
            model.vadapter_test_scale = 0.0
        for m in ev:
            oc = np.asarray(m["own"]).sum(0)
            lat = encode(front, m)
            bt_ = prep(m, dev, max(oc[0], 1.0), max(oc[1], 1.0), lat=lat)
            if noA and "il_check" in globals():
                bt_["fwd"] = {"path": "base"}
            v, _, _ = run_micro(model, bt_, rot, dos, grad=False)
'''
EVAL_TAIL_OLD = '''        g0, g1 = allsum(acc[0], dev), allsum(acc[1], dev)
        d = summarize([x + y for x, y in zip(g0, g1)], "eval")
        d.update(summarize(g0, "eval_asstB")); d.update(summarize(g1, "eval_userB"))
'''
EVAL_TAIL_NEW = '''        g0, g1 = allsum(acc[0], dev), allsum(acc[1], dev)
        if noA:
            model.vadapter_test_scale = None
            d = summarize([x + y for x, y in zip(g0, g1)], "eval_noA")
            model.train()
            d["eval_noA/time_s"] = time.time() - t0_
            emit(d, step)
            log(f"EVAL step {step} WITHOUT the user input: eval_noA_speech/word_ce_norm {d['eval_noA_speech/word_ce_norm']:.4f}")
            return d
        d = summarize([x + y for x, y in zip(g0, g1)], "eval")
        d.update(summarize(g0, "eval_asstB")); d.update(summarize(g1, "eval_userB"))
        if POD:                                                   # the pretraining podcast eval (native A): maintenance of pretraining ability
            accp = None
            for m in evp:
                oc = np.asarray(m["own"]).sum(0)
                v, _, _ = run_micro(model, prep(m, dev, max(oc[0], 1.0), max(oc[1], 1.0), lat=encode(front, m)), rot, dos, grad=False)
                if m.get("dummy"):
                    v = [0.0] * len(v)
                accp = v if accp is None else [x + y for x, y in zip(accp, v)]
            d.update(summarize(allsum(accp, dev), "eval_pod"))
'''

LOOP_IN = '''        if POD:                                                   # co-training: + this rank's podcast micro of step t (1:1 with the si20k micro)
            tw = time.time()
            bp = next(ldp)
            wait += time.time() - tw
            assert int(bp["step"]) == step and bp.get("src") == "podcast", ("podcast micro order", bp.get("step"), step)
            bs.append(bp)
        ocs = [np.asarray(b["own"]).sum(0) for b in bs]
        if step == 0:                                             # user 10-18 PT: decoder K 1 on ~10 % of the light audio for BOTH micro types
            r_ = allsum([float(sum(oc[i] for oc, b in zip(ocs, bs) if b.get("src") != "podcast")) for i in (1, 2)]
                        + [float(sum(oc[i] for oc, b in zip(ocs, bs) if b.get("src") == "podcast")) for i in (1, 2)], dev)
            log(f"[light] step 0, decoder K {CFG['dec_k']}: si20k micros light frames {int(r_[1])} / B coded frames {int(r_[0])} = "
                f"{r_[1] / max(r_[0], 1.0):.4f}; podcast micros {int(r_[3])} / {int(r_[2])} = {r_[3] / max(r_[2], 1.0):.4f} "
                f"(train light_frac si20k {CFG['data']['light_frac']}, podcast {CFG['podcast'].get('data', {}).get('light_frac')})")
'''

MICRO_OLD = '''        for g, b in enumerate(bs):
            t_e = time.time()
            lat = encode(front, b)                                # VOX: frozen Voxtral on this micro's A waveform
            torch.cuda.synchronize(); t_enc += time.time() - t_e
            if step == 0 and CFG["losscheck"]:                    # every logical rank's step-0 micro (pad slots: computed, ignored)
'''
MICRO_NEW = '''        for g, b in enumerate(bs):
            pod_ = b.get("src") == "podcast"
            t_e = time.time()
            lat = encode(front, b)                                # VOX: frozen Voxtral on this micro's A waveform (si20k user / podcast partner)
            torch.cuda.synchronize(); t_enc += time.time() - t_e
            if step == 0 and CFG["losscheck"] and not pod_:       # every logical rank's step-0 si20k micro (pad slots: computed, ignored)
'''

EVL_BUILD_OLD = """    evc = allsum([float(sum(int(m["voice_clip"]) for m in ev)), float(sum(int(m["voice_drawn"]) for m in ev)), float(len(ev)),"""
EVL_BUILD_NEW = """    evl = []                                                      # SI20K_A_GAIN=normal: the low-mic eval split (same rows, target mean - 2 SD)
    if CFG.get("eval_low_data"):
        for g in real:
            evl += A.eval_micros(RIT[g][1], CFG["eval_low_data"], LRANKS[g])
    evc = allsum([float(sum(int(m["voice_clip"]) for m in ev)), float(sum(int(m["voice_drawn"]) for m in ev)), float(len(ev)),"""
EVL_PAD_OLD = """    ev += [dict(ev[0], dummy=1, own=np.zeros_like(ev[0]["own"])) for _ in range(n_ev - len(ev))]   # same forwards on every rank (FSDP)\n"""
EVL_PAD_NEW = EVL_PAD_OLD + """    if CFG.get("eval_low_data"):
        evl += [dict(evl[0] if evl else ev[0], dummy=1, own=np.zeros_like(ev[0]["own"])) for _ in range(n_ev - len(evl))]
        log(f"[eval] low-mic split: {sum(0 if m.get('dummy') else 1 for m in evl)} micros on this rank at {CFG['eval_low_data']['a_gain']} "
            f"(main eval at {CFG['eval_data']['a_gain']}; train {CFG['data']['a_gain']} N({CFG['data']['a_level_mean']}, {CFG['data']['a_level_sd']}) dBFS)")
"""
EVL_EVAL_OLD = """            d.update(summarize(allsum(accp, dev), "eval_pod"))\n"""
EVL_EVAL_NEW = EVL_EVAL_OLD + """        if CFG.get("eval_low_data"):                              # quiet-mic robustness: the si20k eval rows at the low target level
            accl = None
            for m in evl:
                oc = np.asarray(m["own"]).sum(0)
                v, _, _ = run_micro(model, prep(m, dev, max(oc[0], 1.0), max(oc[1], 1.0), lat=encode(front, m)), rot, dos, grad=False)
                if m.get("dummy"):
                    v = [0.0] * len(v)
                accl = v if accl is None else [x + y for x, y in zip(accl, v)]
            d.update(summarize(allsum(accl, dev), "eval_lowmic"))
"""
GAIN_LOG_OLD = """                     float(sum(b.get("persona_drawn", 0) for b in bsi))], dev)
"""
GAIN_LOG_NEW = """                     float(sum(b.get("persona_drawn", 0) for b in bsi))], dev)
        ag = allsum([float(sum(b.get("a_gain_limited", 0) for b in bsi)), float(sum(b.get("a_gain_no_level", 0) for b in bsi)),
                     float(sum(b.get("a_gain_db", 0.0) for b in bsi))], dev)
        AG_CUM[0] += ag[0]; AG_CUM[1] += ag[1]; AG_CUM[2] += ag[2]
"""
GAIN_D_OLD = """                  "train/persona_drawn_frac_cum": vc_cum[4] / max(vc_cum[2], 1.0)})
"""
GAIN_D_NEW = """                  "train/persona_drawn_frac_cum": vc_cum[4] / max(vc_cum[2], 1.0),
                  "train/a_gain_limited_frac_cum": AG_CUM[0] / max(vc_cum[2], 1.0), "train/a_gain_no_level_frac_cum": AG_CUM[1] / max(vc_cum[2], 1.0),
                  "train/a_gain_db_mean_cum": AG_CUM[2] / max(vc_cum[2], 1.0)})
"""


CPU_MAIN = """def main():
    _cpus = sorted(os.sched_getaffinity(0))                       # SPEED FIX v2 (si20k, no maths change): the CPU-heavy collator (decode +
    COLL_CPUS = set(_cpus[-4:]) if len(_cpus) >= 6 else set(_cpus)  # qqa-corrupt-v2 servers, niced) on the rank's LAST 4 CPUs only; the main /
    #                                                               NCCL / CUDA threads keep all the rank's CPUs, 3 of them collator-free (v1 pinned main to 2: starved)
"""
WI_OLD = """    def _wi(_):
        torch.set_num_threads(1); os.environ["OMP_NUM_THREADS"] = "1"
"""
WI_NEW = """    def _wi(_):
        torch.set_num_threads(1); os.environ["OMP_NUM_THREADS"] = "1"
        os.sched_setaffinity(0, COLL_CPUS)                        # si20k SPEED FIX: collator CPUs (inherited by the corruption servers)
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))  # SPEED FIX v3: collator + corruption run only on idle CPU time (applied live 22:42 PT)
"""

COMMON = [
    ('        out = model(b, rot, rot, info=info)\n', '        out = model(b, rot, rot, info=info, **(b.get("fwd") or {}))\n'),
    ('TK = ("word_ce", "word_norm", "n_word", "word_hit", "pad_ce", "n_pad", "nw_ce", "n_nw", "eob_ce", "n_eob")\n',
     PREP_POD + 'TK = ("word_ce", "word_norm", "n_word", "word_hit", "pad_ce", "n_pad", "nw_ce", "n_nw", "eob_ce", "n_eob")\n'),
    ("    ev, ev_first = [], [None] * GMAX                              # every logical rank's eval micros (built with its logical rank)\n",
     LOADERS + "    ev, ev_first = [], [None] * GMAX                              # every logical rank's eval micros (built with its logical rank)\n"),
    (EVAL_HEAD_OLD, EVAL_HEAD_NEW),
    (EVAL_TAIL_OLD, EVAL_TAIL_NEW),
    ('    evaluate(0)\n    import gc\n', '    evaluate(0)\n    evaluate(0, noA=True)                                          # user 10-18 PT: step-0 eval with vs without the user input\n    import gc\n'),
    ('        ocs = [np.asarray(b["own"]).sum(0) for b in bs]\n', LOOP_IN),
    ('        assert all(not (int(b["padslot"]) and not b.get("dummy", 0)) for b in bs), "a pad slot must be a dummy micro"\n',
     '        assert all(not (int(b.get("padslot", 0)) and not b.get("dummy", 0)) for b in bs), "a pad slot must be a dummy micro"\n'),
    ('        v, loss_v, ex_ = None, 0.0, None\n', '        v, vp, loss_v, ex_ = None, None, 0.0, None\n'),
    (MICRO_OLD, MICRO_NEW),
    ('            bt = prep(b, dev, tot[0] / WORLD, tot[1] / WORLD, lat=lat)   # tokens_step normalisers (physical world: FSDP / decoder average over WORLD)\n',
     '            bt = prep(b, dev, tot[0] / WORLD, tot[1] / WORLD, lat=lat)   # tokens_step over ALL micros of the step (podcast A = Voxtral too)\n'),
    ('            L_ = LRANKS[g] if LRANKS[g] is not None else LRANKS[0]\n',
     '            L_ = (1000 + RANK) if pod_ else (LRANKS[g] if LRANKS[g] is not None else LRANKS[0])   # podcast micro: its own decoder-permutation seed\n'),
    ('"first": g == 0, "last": g == GMAX - 1, "lrank": L_, "step": step})', '"first": g == 0, "last": g == len(bs) - 1, "lrank": L_, "step": step})'),
    ('            v = v_ if v is None else [x + y for x, y in zip(v, v_)]\n', '''            if pod_:
                vp = v_ if vp is None else [x + y for x, y in zip(vp, v_)]
            else:
                v = v_ if v is None else [x + y for x, y in zip(v, v_)]
'''),
    ('            ex_ = ex_g if g == GMAX - 1 else ex_\n', '            ex_ = ex_g if g == len(bs) - 1 else ex_\n'),
    ('        d = summarize(sv[:-1], "train")\n', '''        d = summarize(sv[:-1], "train")                            # si20k micros only (= the reference's curve); podcast below
        if POD:
            d.update(summarize(allsum(vp, dev), "train_pod"))
'''),
    ('''        vc = allsum([float(sum(b.get("voice_clip", 0) for b in bs)), float(sum(b.get("voice_drawn", 0) for b in bs)),
                     float(sum(0.0 if b.get("dummy", 0) else 1.0 for b in bs)), float(sum(b.get("persona_used", 0) for b in bs)),
                     float(sum(b.get("persona_drawn", 0) for b in bs))], dev)
''', '''        bsi = [b for b in bs if b.get("src") != "podcast"]             # prompt counters over the si20k micros only
        vc = allsum([float(sum(b.get("voice_clip", 0) for b in bsi)), float(sum(b.get("voice_drawn", 0) for b in bsi)),
                     float(sum(0.0 if b.get("dummy", 0) else 1.0 for b in bsi)), float(sum(b.get("persona_used", 0) for b in bsi)),
                     float(sum(b.get("persona_drawn", 0) for b in bsi))], dev)
'''),
    ("    # ---- model\n", SETUP + "    # ---- model\n"),
    (EVL_BUILD_OLD, EVL_BUILD_NEW), (EVL_PAD_OLD, EVL_PAD_NEW), (EVL_EVAL_OLD, EVL_EVAL_NEW), (GAIN_LOG_OLD, GAIN_LOG_NEW), (GAIN_D_OLD, GAIN_D_NEW),
    ("batch_size=None, shuffle=False, num_workers=W_, prefetch_factor=2 if W_ else None,", "batch_size=None, shuffle=False, num_workers=W_, prefetch_factor=4 if W_ else None,"),   # corruption in the collator: deeper prefetch (engineering)
    ("def main():\n", CPU_MAIN), (WI_OLD, WI_NEW),
    ("STOP = {\"flag\": False}\n", "STOP = {\"flag\": False}\nAG_CUM = [0.0, 0.0, 0.0]                                          # SI20K_A_GAIN: peak-limited, no-level, gain dB (cumulative)\n"),
]

PER = {
    "ppm": [('    assert SKIPPED_INIT.get("pose_head."), "expected the D32 pose_head keys in the PPM init (skipped)"\n',
             '    assert SKIPPED_INIT.get("tower_p.") and set(SKIPPED_INIT) <= set(POSE_PREFIX), \\\n'
             '        "expected the init\'s pose parts (skipped, not built); dcp-12829 has no D32 pose_head (11488 had 2 pose_head tensors: the only key difference)"\n')],
    "intl": [('def il_check(b):\n', 'def il_check(b, user_words_ok=False):\n'),
             ('    assert (ids[0] == D.PAD_ID).all(), "nextvox: a user word id on channel A (the user stream must be Voxtral latents only)"\n',
              '    assert user_words_ok or (ids[0] == D.PAD_ID).all(), "nextvox: a user word id on channel A (the user stream must be Voxtral latents only; podcast micros: native A)"\n'),
             ('            ichk = {"check_user_slot": True} if step == 0 else None    # nextvox DATA CHECK on every rank\'s first train micro\n',
              '            ichk = {"check_user_slot": True} if step == 0 and not pod_ else None    # nextvox DATA CHECK on every rank\'s first si20k train micro\n'),
             ('            if step == 0:\n                ichk = getattr(model, "user_slot_check", None)',
              '            if step == 0 and not pod_:\n                ichk = getattr(model, "user_slot_check", None)')],
}


def apply(name, s):
    for a, b in COMMON + PER[name]:
        n = s.count(a)
        if a.startswith("    # ---- model") and name == "intl":
            a, b = "    # ---- model (next-arch 14B", SETUP + "    # ---- model (next-arch 14B"
            n = s.count(a)
        assert n == 1, ("co_patch anchor must occur exactly once", name, n, a[:100])
        s = s.replace(a, b)
    return s
