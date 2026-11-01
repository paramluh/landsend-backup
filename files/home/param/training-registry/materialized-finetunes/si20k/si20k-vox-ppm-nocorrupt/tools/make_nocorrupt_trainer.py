"""si20k-vox-ppm-nocorrupt (copy of si20k-vox-ppm11488/tools/make_m11488_trainer.py): generate code/train_si20k_vox_nocorrupt.py from the run's copy of si20k-vox-ppm's trainer (byte-identical to
si20k-vox-ppm/code/train_si20k_vox.py, never edited there) by EXACT replacements: the ONLY recipe change is the backbone init (QQA_INIT =
local-11488, set by the sbatch: the backbone the ASR adapter w-1972 was trained on), plus bookkeeping: the no-A eval at every eval, the fp32
vgain-theta export fix, the ASR-adapter load check, a cfg diff against j24920's own final cfg. Diff -> tests/trainer_nocorrupt_vs_vox.diff."""
import difflib

R = "/data/home/amartyavarma/ppm/si20k-vox-ppm-nocorrupt"
SRC, DST = f"{R}/code/train_si20k_vox.py", f"{R}/code/train_si20k_vox_nocorrupt.py"
VOX = "/data/home/amartyavarma/ppm/si20k-vox-ppm/code/train_si20k_vox.py"
s = open(SRC).read()
assert s == open(VOX).read()


def rep(a, b, n=1):
    global s
    c = s.count(a)
    assert c == n, (c, n, a[:160])
    s = s.replace(a, b)


rep('"""si20k-vox TRAINER (10-19, user:', '''"""si20k-vox-ppm-nocorrupt TRAINER (10-19, user-approved: "the si20k speech-in, speech-out PPM fine-tune WITHOUT audio corruption, to
test why speech-in trails text-in on si20k") = the finished si20k-vox-ppm j24920 trainer below, IDENTICAL recipe (init dcp-12829 + ASR w-1972);
the only change is SI20K_A_CORRUPT=none (env): qqa-corrupt-v2 OFF on the si20k user audio AND the podcast A audio (the gauss:6 mic gain stays). Bookkeeping added (no maths change): the si20k eval WITHOUT the user input (adapter output
x 0) at EVERY eval + the gap, the vgain theta saved in fp32 in every export, the ASR-adapter load check (per-tensor norms == checkpoint),
the cfg diff against j24920's own final cfg (only run dirs, init, store root, ledger dir and the runtime calibration keys may differ).
si20k-vox TRAINER (10-19, user:''')
# no-A eval every eval + gap
rep('''        if (CFG["eval_every"] and done % CFG["eval_every"] == 0) or done == CFG["steps"] or done in CFG["export_steps"]:
            evaluate(done)
''', '''        if (CFG["eval_every"] and done % CFG["eval_every"] == 0) or done == CFG["steps"] or done in CFG["export_steps"]:
            evaluate(done)
            evaluate(done, noA=True)                              # (coordinator 10-19) A zeroed at EVERY eval -> the listening gap
''')
rep('''            log(f"EVAL step {step} WITHOUT the user input: eval_noA_speech/word_ce_norm {d['eval_noA_speech/word_ce_norm']:.4f}")''',
    '''            log(f"EVAL step {step} WITHOUT the user input: eval_noA_speech/word_ce_norm {d['eval_noA_speech/word_ce_norm']:.4f} "
                f"(with A {LAST_EVAL.get('eval_speech/word_ce_norm', float('nan')):.4f}; gap noA - A {d['eval_noA_speech/word_ce_norm'] - LAST_EVAL.get('eval_speech/word_ce_norm', float('nan')):+.4f})")
            emit({"eval_noA/gap_noA_minus_A": d["eval_noA_speech/word_ce_norm"] - LAST_EVAL.get("eval_speech/word_ce_norm", float("nan"))}, step)''')
rep('''        log(f"EVAL step {step}: " + " | ".join(f"{k} {v:.4f}" for k, v in d.items()))''',
    '''        LAST_EVAL.clear(); LAST_EVAL.update(d)
        log(f"EVAL step {step}: " + " | ".join(f"{k} {v:.4f}" for k, v in d.items()))''')
rep('''STOP = {"flag": False}\n''', '''STOP = {"flag": False}\nLAST_EVAL = {}                                                    # the latest with-A eval (for the noA gap)\n''')
# fp32 theta export fix
rep('''    dcp.save({k: v.to(torch.bfloat16) for k, v in msd.items()}, checkpoint_id=path)''',
    '''    assert "vgain.theta" in msd, "export: vgain.theta missing from the model state dict"
    th_ = float(model.vgain.theta.full_tensor().reshape(-1)[0]) if hasattr(model.vgain.theta, "full_tensor") else float(model.vgain.theta.reshape(-1)[0])
    dcp.save({k: (v.to(torch.float32) if k == "vgain.theta" else v.to(torch.bfloat16)) for k, v in msd.items()}, checkpoint_id=path)   # theta fp32 (bf16 shifted s by ~1 %)''')
rep('''        json.dump({"step": int(step), "decp_bb": 0, "pose": 0, "nopose_model": 1, "run": CFG["run_tag"], "init": CFG["init"], "cfg": CFG},''',
    '''        json.dump({"step": int(step), "decp_bb": 0, "pose": 0, "nopose_model": 1, "vgain_theta_fp32": th_, "vgain_s_fp32": math.exp(th_),
                   "vgain_theta_dtype": "float32", "run": CFG["run_tag"], "init": CFG["init"], "cfg": CFG},''')
rep('''        if hasattr(p, "to_local"):
            sh[k] = p.to_local().detach().to(torch.bfloat16).cpu().contiguous().clone()
        elif RANK == 0:
            rp[k] = p.detach().to(torch.bfloat16).cpu().contiguous().clone()''',
    '''        dt_ = torch.float32 if k == "vgain.theta" else torch.bfloat16  # theta (log s) in fp32 (bf16 shifted s by ~1 %)
        if hasattr(p, "to_local"):
            sh[k] = p.to_local().detach().to(dt_).cpu().contiguous().clone()
        elif RANK == 0:
            rp[k] = p.detach().to(dt_).cpu().contiguous().clone()''')
rep('''    hosts = [None] * WORLD; dist.all_gather_object(hosts, HOST)''', '''    hosts = [None] * WORLD; dist.all_gather_object(hosts, HOST)
    sx_ = s_value(model)                                          # collective: exp(theta fp32), recorded in meta.json''')
rep('''"run_tag": CFG["run_tag"], "init": CFG["init"], "cfg": CFG, "save_time": now_pt()}, open(f"{path}/meta.json", "w"), indent=1, default=str)''',
    '''"run_tag": CFG["run_tag"], "init": CFG["init"], "cfg": CFG, "save_time": now_pt(),
                   "vgain_s_fp32": sx_, "vgain_theta_fp32": math.log(sx_), "vgain_theta_dtype": "float32"}, open(f"{path}/meta.json", "w"), indent=1, default=str)''')
# ASR adapter load check
rep('''    model.vadapter.target_rms.fill_(float("nan"))
    set_s(model, CFG["vox_a_scale_init"])''', '''    rows_ = []                                                    # the loaded fp32 masters == the checkpoint tensors (asserted)
    for n, p in model.vadapter.named_parameters(prefix="vadapter"):
        ss = torch.tensor([float(M._loc(p).double().pow(2).sum())], device=dev, dtype=torch.float64)
        if hasattr(p, "to_local"):
            dist.all_reduce(ss)
        nm, nc = math.sqrt(float(ss)), float(sd[n].double().norm())
        rows_.append((n, nm, nc))
        assert abs(nm - nc) <= 1e-5 * max(nc, 1e-12), ("ASR adapter tensor NOT loaded", n, nm, nc)
    log("[init] ASR adapter LOADED, per-tensor L2 norm model / checkpoint: " + "; ".join(f"{n} {a:.6f} / {b:.6f}" for n, a, b in rows_))
    model.vadapter.target_rms.fill_(float("nan"))
    set_s(model, CFG["vox_a_scale_init"])''')
rep('''log(f"si20k-vox (PPM, ref aloe-v2vox-raw j23081)''', '''log(f"si20k-vox-ppm-nocorrupt (= j24920 without qqa-corrupt-v2; PPM, ref aloe-v2vox-raw j23081)''')
# export wait fix (10-19, xfer-tt-ss lost its final: rank 0 waited for /data space while every other rank sat in ONE broadcast past the
# 30-min collective timeout): the wait is now collective, one small broadcast every 15 s.
rep('''    ok = [None]
    if RANK == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tw = time.time()
        while True:
            free = shutil.disk_usage(os.path.dirname(path)).free / 1e9
            if free - size_gb >= min_free_gb or time.time() - tw > 60.0 * CFG["export_wait_min"] or os.path.exists(path):
                break
            time.sleep(15.0)
        if time.time() - tw > 1.0:
            log(f"[export] waited {time.time() - tw:.0f}s for /data space ({free:.0f} GB free)")
        ok = [(free - size_gb >= min_free_gb and not os.path.exists(path), free)]
    dist.broadcast_object_list(ok, src=0)''', '''    tw = time.time()
    while True:                                                   # collective wait: every rank joins one small broadcast every 15 s
        st_ = [None]
        if RANK == 0:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            free_ = shutil.disk_usage(os.path.dirname(path)).free / 1e9
            st_ = [(free_ - size_gb >= min_free_gb or time.time() - tw > 60.0 * CFG["export_wait_min"] or os.path.exists(path), free_,
                    free_ - size_gb >= min_free_gb and not os.path.exists(path))]
        dist.broadcast_object_list(st_, src=0)
        if st_[0][0]:
            break
        time.sleep(15.0)
    if RANK == 0 and time.time() - tw > 1.0:
        log(f"[export] waited {time.time() - tw:.0f}s for /data space ({st_[0][1]:.0f} GB free)")
    ok = [(st_[0][2], st_[0][1])]''')
# export robustness (10-19 bug fix, no maths change: si20k-vox-ppm11488 lost its final when /scratch filled during export_local): (1) the
# consolidated /data final is written FIRST, the node-local final copy after it; (2) every export_local checks /scratch space on every node
# first (collective) and SKIPS the node-local copy with a loud log line instead of crashing the job.
rep('''    t0 = time.time()
    os.makedirs(path, exist_ok=True)
    sh, rp = {}, {}''', '''    t0 = time.time()
    anc_ = path
    while not os.path.isdir(anc_):
        anc_ = os.path.dirname(anc_)
    need_gb = (sum(M._loc(p_).numel() * 2 for p_ in model.parameters() if hasattr(p_, "to_local")) * LW          # this node's 8 ranks' bf16 shards
               + (sum(p_.numel() * 2 for p_ in model.parameters() if not hasattr(p_, "to_local")) if RANK // LW == 0 else 0)) / 1e9 + 3.0   # + rep.pt on rank 0's node
    free_gb = shutil.disk_usage(anc_).free / 1e9
    ok_ = torch.tensor([1.0 if free_gb >= need_gb else 0.0], device=next(model.parameters()).device)
    dist.all_reduce(ok_, op=dist.ReduceOp.MIN)
    if float(ok_) < 1.0:
        log(f"!!! [export-local] SKIPPED {path}: a node has < {need_gb:.1f} GB free on its /scratch (rank 0's node {free_gb:.1f} GB); node-local copy not written")
        return None
    os.makedirs(path, exist_ok=True)
    sh, rp = {}, {}''')
rep('''        export_local(model, CFG["steps"], f"{CFG['local_export_root']}/export-{CFG['steps']}")   # aloe: node-local copy first (never refused)
        export_weights(model, CFG["steps"], CFG["export"], CFG["export_min_free_gb"])''', '''        export_local(model, CFG["steps"], f"{CFG['local_export_root']}/export-{CFG['steps']}")   # node-local copy first (skipped, never crashing, if /scratch is short)
        export_weights(model, CFG["steps"], CFG["export"], CFG["export_min_free_gb"])   # then /data (10-19 fix: collective 15 s polls while waiting for space)''')
# cfg diff vs j24920's final cfg
a0 = s.index("SI20K_ALLOWED = (")
a1 = s.index("\n\n\ndef cfg_diff_si20k():", a0)
s = s[:a0] + '''SI20K_ALLOWED = ("run_tag", "log_root", "plan_dir", "control_dir", "export", "local_export_root", "si20k.ref_cfg",   # run dirs / ref path
                 "data.a_corrupt", "podcast.data.a_corrupt",                         # THE change: qqa-corrupt-v2 -> none (si20k user + podcast A)
                 "stores", "si20k.ledger_dir",                                       # stores read in place from /data; ledger dir
                 "vox_a_scale_init_recal", "vox_calib_raw_rms")                      # set at runtime (after j24920's cfg diff)''' + s[a1:]
rep('''        a, b = ref.get(k, "<absent>"), cur.get(k, "<absent>")
        if a != b:
            ok = any(k == x or k.startswith(x + ".") for x in SI20K_ALLOWED)
            rows.append(f"  {'ok ' if ok else 'BAD'} {k}: ref {a} -> si20k {b}")''',
    '''        a, b = ref.get(k, "<absent>"), cur.get(k, "<absent>")
        if a != b and not (isinstance(a, float) and isinstance(b, float) and a != a and b != b):   # NaN == NaN here
            ok = any(k == x or k.startswith(x + ".") for x in SI20K_ALLOWED)
            rows.append(f"  {'ok ' if ok else 'BAD'} {k}: j24920 {a} -> nocorrupt {b}")''')
open(DST, "w").write(s)
d = difflib.unified_diff(open(VOX).read().splitlines(1), s.splitlines(1), "si20k-vox-ppm/train_si20k_vox.py", "si20k-vox-ppm-nocorrupt/train_si20k_vox_nocorrupt.py")
open(f"{R}/tests/trainer_nocorrupt_vs_vox.diff", "w").writelines(d)
print("wrote", DST, len(s.splitlines()), "lines")
