"""Transcript conditioning with the production Xabi media updates.

The archived run.env is the hyperparameter source. Only the data layout, world/
batch, text objective, logging and operational checkpoint policy differ.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "xabi-main-17850"
sys.path.insert(0, str(HERE.parents[1]))
from finetune.storage_paths import canonical_run_dir, prepare_training_storage


def arguments():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("train", "benchmark-old", "benchmark-new"), default="train")
    p.add_argument("--recipe", choices=("aligned-media", "compact-all", "no-text", "xabi-cb1", "pose-cb01"), default="aligned-media")
    p.add_argument("--micro-batch-rows", type=int, default=None,
                   help="no-text / xabi-cb1: original logical rows per GPU forward (default2)")
    p.add_argument("--gradient-accumulation", type=int, default=None,
                   help="no-text / xabi-cb1: forwards per optimizer step (default1)")
    p.add_argument("--cache", type=Path, default=Path("/home/param/datasets/xabi-v2pure-cache"))
    p.add_argument("--base", type=Path, default=Path("/home/param/models/ppm-811-17850-base-bf16"))
    p.add_argument("--resume", choices=("auto", "never"), default="auto")
    p.add_argument("--no-eval", action="store_true", help="Skip all evaluations, including initial and final")
    p.add_argument("--final-checkpoint-only", action="store_true", help="Save full state and weights only at the final training step")
    p.add_argument("--benchmark-warmup", type=int, default=1)
    p.add_argument("--benchmark-steps", type=int, default=2)
    p.add_argument("--benchmark-stress", action="store_true",
                   help="Original-row benchmark: full7500-frame rows and maximum token padding; cb1 also varies shapes")
    p.add_argument("--wandb-project", default="param-pose-llm")
    return p.parse_args()


def configure(args):
    """Read literal KEY=VALUE assignments; never source archived shell code."""
    for line in (BASE / "run.env").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        os.environ[key] = value
    root = canonical_run_dir(args.run_dir).resolve()
    model_root = prepare_training_storage(root, ('saves', 'exports'))
    compact = getattr(args, "recipe", "aligned-media") == "compact-all"
    pose_only = getattr(args, "recipe", "aligned-media") == "pose-cb01"
    cb1 = getattr(args, "recipe", "aligned-media") in ("xabi-cb1", "pose-cb01")
    if getattr(args, "recipe", "aligned-media") != "aligned-media" and args.mode == "benchmark-old":
        raise ValueError("benchmark-old only supports the original aligned-media configuration")
    overrides = {
        "QQA_RUN": root.name, "QQA_LOG_ROOT": str(root.parent),
        "QQA_INIT": str(args.base / "weights"), "QQA_PLAN_DIR": str(root),
        "QQA_CONTROL_DIR": str(root / "control"), "QQA_EXPORT": str(model_root / "exports/export-200"),
        "ALOE_LOCAL_EXPORT_ROOT": str(model_root / "exports"), "ALOE_SAVE_ROOT": str(model_root / "saves"),
        "ALOE_STORE_DIR": str(args.cache / "provenance/original/ppm/aloe-v2pure-ft/stores"),
        "ALOE_RESUME": args.resume, "ALOE_FINAL_SAVE": "1",
        "QQA_LW_WORD": "1" if pose_only else "10" if args.mode == "benchmark-old" or cb1 else "1" if compact else "0",
        "QQA_POSE_CB1": "1" if cb1 else "0",
        "QQA_POSE_LIGHT": "0" if pose_only else "1",
        "QQA_AUDIO_TRAINING": "0" if pose_only else "1",
        "QQA_LW_SPEECH": "0" if pose_only else "1",
        "QQA_LR_TOWER_P1": "3e-4" if pose_only else os.environ["QQA_LR_TOWER_P"],
        # Equivalent packed data math is checked in the adapter tests. Avoid the
        # legacy gate's many full-size model forwards before a quick benchmark.
        "QQA_LOSSCHECK": "0",
    }
    if compact or pose_only:
        # Category CE + conditional lexical CE = ordinary vocabulary CE when
        # both coefficients are one; remove Xabi's extra lexical-token boost.
        overrides["QQA_W_TEXT_CE"] = "1"
    if getattr(args, "no_eval", False):
        overrides["QQA_EVAL_EVERY"] = "0"
    if getattr(args, "final_checkpoint_only", False):
        overrides["QQA_EXPORT_STEPS"] = ""
    os.environ.update(overrides)
    os.environ.pop("QQA_REF_CFG", None)
    for path in (BASE / "code/lib", BASE / "code"):
        sys.path.insert(0, str(path))
    return root


def media_metrics(T, values, prefix="train", codebooks=True, text_loss=False):
    d = T.summarize(values, "media")
    audio = d["media_speech/cb0_ce"] + d["media_speech/light_ce"]
    pose = d["media_speech/pose_cb0_ce"] + d["media_speech/pose_light_ce"]
    if T.CFG.get("pose_cb1"):
        pose += d["media_speech/pose_cb1_ce"]
    audio_training = T.CFG.get("audio_training", True)
    pose_light = T.CFG.get("pose_light", True)
    out = {f"{prefix}/pose_loss": pose, f"{prefix}/loss": pose}
    if audio_training:
        out[f"{prefix}/audio_loss"] = audio
        out[f"{prefix}/loss"] += audio
    if text_loss:
        out[f"{prefix}/text_loss"] = d["media_speech/text_ce"]
        # Original Xabi boosts conditional lexical CE by 5, then the whole
        # text stream by 10. Report the actual weighted objective explicitly.
        weighted = T.CFG["lw_word"] * (d["media_speech/text_ce"] +
                    (T.CFG["w_text"] - 1) * d["media_speech/word_ce_norm"] *
                    d["media_speech/n_word"] / max(sum(values[j] for j in (2, 5, 7, 9)), 1.0))
        out[f"{prefix}/weighted_text_loss"] = weighted
        out[f"{prefix}/loss"] += weighted
    if T.CFG.get("pose_cb1"):
        out[f"{prefix}/pose_cb0_loss"] = d["media_speech/pose_cb0_ce"]
        out[f"{prefix}/pose_cb1_loss"] = d["media_speech/pose_cb1_ce"]
        if pose_light:
            out[f"{prefix}/pose_light_loss"] = d["media_speech/pose_light_ce"]
    if codebooks:
        out["pose_codebooks/00"] = d["media_speech/pose_cb0_ce"]
        if audio_training:
            out["audio_codebooks/00"] = d["media_speech/cb0_ce"]
            for cb in T.CB_S:
                out[f"audio_codebooks/{cb:02d}"] = d[f"media_speech/light_ce_cb{cb}"]
        if T.CFG.get("pose_cb1"):
            out["pose_codebooks/01"] = d["media_speech/pose_cb1_ce"]
        for cb in T.CB_P:
            out[f"pose_codebooks/{cb:02d}"] = d[f"media_speech/pose_light_ce_cb{cb}"]
    return out


def main():
    args = arguments()
    root = configure(args)
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    import train_aloe_pose as T
    from data_transcript_pose import CompactXabi, StepDataset, eval_micros, plan_digest, stress_no_text_plan

    M, C = T.M, T.CFG
    T.CB_S = T.CB_P = tuple(range(1, 32))
    bench = args.mode != "train"
    compact = args.recipe == "compact-all"
    no_text = args.recipe == "no-text"
    pose_only = args.recipe == "pose-cb01"
    cb1 = args.recipe in ("xabi-cb1", "pose-cb01")
    original_rows = no_text or cb1
    if cb1:
        T.CB_P = tuple(range(2, 32))
    if pose_only:
        T.CB_S = T.CB_P = ()
    if args.benchmark_stress and (not bench or not original_rows):
        raise ValueError("--benchmark-stress requires a no-text or xabi-cb1 benchmark")
    data_mode = "xabi-cb1" if cb1 else "no-text" if no_text else "original" if args.mode == "benchmark-old" else "compact-transcript" if compact else "transcript"
    ga = args.gradient_accumulation if args.gradient_accumulation is not None else (1 if compact or original_rows else 2)
    mbs = args.micro_batch_rows if args.micro_batch_rows is not None else (2 if original_rows else 1)
    if not original_rows and (args.micro_batch_rows is not None or args.gradient_accumulation is not None):
        raise ValueError("Explicit batch/accumulation options require no-text or xabi-cb1")
    limit = args.benchmark_warmup + args.benchmark_steps if bench else C["steps"]
    assert 0 < limit <= C["steps"] and args.benchmark_warmup >= 0 and args.benchmark_steps > 0
    dev = torch.device("cuda", T.LOCAL)
    torch.cuda.set_device(dev)
    torch.set_num_threads(1)
    torch.manual_seed(C["dec_seed"])
    dist.init_process_group("nccl", device_id=dev, timeout=datetime.timedelta(minutes=45))
    expected_world = 32 if original_rows else 56
    assert T.WORLD == expected_world and T.LW == 8, ("Unexpected distributed layout", T.WORLD, T.LW, expected_world)
    if original_rows:
        assert mbs * ga * T.WORLD == 64 and mbs in (1, 2) and ga in (1, 2)
    root.mkdir(parents=True, exist_ok=True)
    (root / "control").mkdir(exist_ok=True)
    signal.signal(signal.SIGUSR1, T._usr1)
    M.QWEN32 = str(args.base / "configs/Qwen3-32B")
    M.QWEN4B_CFG = str(args.base / "configs/Qwen3-4B")
    data = CompactXabi(args.cache, M.QWEN32)
    # One planner per node; all other ranks wait, then read the same serialized plan.
    plan_path = root / "plan.json"
    if T.LOCAL == 0 and not plan_path.exists():
        plan = data.make_plan(T.WORLD, steps=C["steps"], ga=ga, budget=7500,
                              mode=data_mode, oversize_policy="split", micro_batch_rows=mbs)
        if args.benchmark_stress:
            plan = stress_no_text_plan(plan)
        tmp = plan_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(plan, separators=(",", ":")))
        tmp.replace(plan_path)
    dist.barrier()
    plan = json.loads(plan_path.read_text())
    assert plan["digest"] == plan_digest(plan)
    assert plan["source_plan_sha256"] == data.source_plan_sha256
    assert plan["data_cfg"] == data.cfg, "Saved plan uses different data conditioning"
    assert plan["world"] == T.WORLD and plan["steps"] == C["steps"]
    assert plan["mode"] == data_mode
    assert bool(plan.get("benchmark_stress")) == args.benchmark_stress
    if original_rows:
        assert plan["ga"] == ga and plan["micro_batch_rows"] == mbs and plan["global_batch_rows"] == 64
    elif data_mode != "original":
        assert plan["ga"] == ga and plan["budget"] == 7500
    digests = [None] * T.WORLD
    dist.all_gather_object(digests, plan["digest"])
    assert len(set(digests)) == 1, "Plan differs between nodes"
    if compact:
        C["export_steps"] = []  # This short second run saves automatically only at the end.
    C.update(task="pose-cb01-text-only-objective" if pose_only else "xabi-second-pose-tower-cb1" if cb1 else "no-text-audio-pose" if no_text else "compact-transcript-text-audio-pose" if compact else "aligned-transcript-audio-pose",
             recipe=args.recipe, sequence_budget=plan["budget"], gradient_accumulation=plan["ga"],
             source_cache=str(args.cache), plan_digest=plan["digest"], mode=args.mode,
             checkpoint_steps=C["export_steps"] + [C["steps"]], retain_full_checkpoints=2,
             checkpoint_policy="final-only" if compact or args.final_checkpoint_only else "xabi",
             evaluation_enabled=not args.no_eval,
             cuda_allocator=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
             compiler_cache=os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
             triton_cache=os.environ.get("TRITON_CACHE_DIR"))
    if no_text:
        C.update(micro_batch_rows=mbs, global_batch_rows=64, logical_row_media_budget=7500,
                 text_policy=plan["text_policy"], output_text_policy="constant PAD", text_loss_enabled=False)
    if cb1:
        C.update(micro_batch_rows=mbs, global_batch_rows=64, logical_row_media_budget=7500,
                 text_policy=plan["text_policy"], text_loss_enabled=True,
                 pose_loss_definition="cb0 CE + cb1 CE + mean CE(cb2..31)",
                 pose_cb1_init="tower_p shared weights copied; current cb0 embedding and upstream hidden projection freshly initialized",
                 pose_light_conditioning="cb1 tower hidden + original backbone link; cb0/cb1 prefill",
                 initialization_checkpoint_step=17850)
    if pose_only:
        C.update(pose_loss_definition="cb0 CE + cb1 CE", loss_definition="text CE + pose cb0 CE + pose cb1 CE",
                 pose_light_conditioning=None, audio_conditioning="all original audio codes retained in backbone inputs",
                 active_decoders=[], main_forward_passes_per_micro=1, cb1_peak_lr=C["lr"]["tower_p1"])
    if T.RANK == 0:
        (root / "config.json").write_text(json.dumps(C, indent=2, default=str))
        T.log(f"[plan] {plan['digest']} mode={plan['mode']} coverage={plan.get('coverage')} rounds={sorted(set(plan['R']))}")
    c32, c4, c06 = M.cfg32(64), M.cfg4b(36), M.cfg06(28)
    model, _ = M.build(c32, c4, c06, dev, T.log, partial=False, proj_r=2048,
                       tower_vocab=M.TOWER_VOCAB, init=False, dec_ac=False,
                       dec_mode="replicate", pose_head=False, decp_bb=not pose_only, pose_cb1=cb1,
                       pose_light=not pose_only, audio_training=not pose_only)
    model.head.media_only = C["lw_word"] == 0
    if hasattr(model, "decp_bb"):
        model.decp_bb_alpha = C["decp_bb_alpha"]
    opt, sched, by, frozen, decn = T.make_opt(model)
    dos = {} if pose_only else {"dec_s": T.DecOpt(model.dec_s, C["dec_lr"], C["dec_clip"]),
                               "dec_p": T.DecOpt(model.dec_p, C["dec_p_lr"], C["dec_p_clip"])}
    start = T.find_resume() if args.resume == "auto" else None
    if start is not None:
        T.load_full(model, opt, sched, dos, start, plan["digest"], dev)
    else:
        start = 0
        meta, gb = T.load_init(model, C["init"], dev)
        assert int(meta["step"]) == 17850
        if cb1:
            from cb1_init import initialize_cb1
            initialize_cb1(model)
            T.log("[init] cb1 shared tower weights copied from cb0; new conditioning weights freshly initialized")
        T.log(f"[init] pretrain step17850; source world={meta['world']}; {gb:.1f} GB loaded across ranks")
    if C["dec_fast"]:
        for name in dos:
            M.compile_decoder(getattr(model, name), C["dec_compile_mode"], whole=True, dynamic=C["dec_dyn"])
    rot = M.Rot(c32, dev)
    wb = None
    if not bench and T.RANK == 0:
        import wandb
        idfile = root / "wandb-id"
        runid = idfile.read_text().strip() if idfile.exists() else uuid.uuid4().hex[:8]
        idfile.write_text(runid + "\n")
        wb = wandb.init(project=args.wandb_project, id=runid, resume="allow", name=root.name,
                        dir=str(root), config=C, settings=wandb.Settings(
                            x_disable_stats=True, x_disable_meta=True, x_disable_machine_info=True,
                            disable_git=True, disable_code=True, disable_job_creation=True, console="off"))
        (root / "wandb-url").write_text(wb.url + "\n")
        T.log(f"[wandb] {wb.url}")
    metrics_file = open(root / "metrics.jsonl", "a") if T.RANK == 0 else None

    def emit(metrics, step):
        assert all(math.isfinite(v) for v in metrics.values()), ("Nonfinite metrics", metrics)
        if T.RANK == 0:
            metrics_file.write(json.dumps(dict(step=step, **metrics)) + "\n")
            metrics_file.flush()
            if wb:
                # Evaluation at the same step joins this record; the next step
                # commits it. Avoid dropping eval by advancing W&B too early.
                wb.log(metrics, step=step, commit=False)

    def evaluate(step):
        model.eval()
        ev = list(eval_micros(data, plan, T.RANK))
        n = torch.tensor([len(ev)], device=dev)
        dist.all_reduce(n, op=dist.ReduceOp.MAX)
        fallback = ev[0] if ev else StepDataset(data, plan, T.RANK)[0][0]
        ev += [dict(fallback, dummy=1, own=np.zeros_like(fallback["own"])) for _ in range(int(n) - len(ev))]
        acc = None
        with torch.no_grad():
            for micro in ev:
                oc = np.asarray(micro["own"]).sum(0)
                batch = T.prep(micro, dev, max(oc[0], 1), max(oc[1], 1), max(oc[3], 1))
                v, _, _ = T.run_micro(model, batch, rot, dos, grad=False)
                acc = v if acc is None else [a + b for a, b in zip(acc, v)]
        result = media_metrics(T, T.allsum(acc, dev), prefix="eval", codebooks=cb1, text_loss=compact or cb1)
        if cb1:
            # Evaluation codebook series must not overwrite training series.
            result = {("eval/" + k if k.startswith(("audio_codebooks/", "pose_codebooks/")) else k): v for k, v in result.items()}
        emit(result, step)
        T.log(f"[eval] step={step} {result}")
        model.train()

    def worker_init(_):
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"

    loader = iter(DataLoader(StepDataset(data, plan, T.RANK, start=start), batch_size=None, shuffle=False,
                             num_workers=C["dl_workers"], prefetch_factor=2, persistent_workers=True,
                             worker_init_fn=worker_init, collate_fn=lambda x: x))
    if not bench and C["evaluation_enabled"] and start == 0:
        evaluate(0)
    model.train()
    gc.collect()
    times, peaks, reserved_peaks, allocator_retries, allocator_ooms = [], [], [], [], []
    stop_done = None
    for step in range(start, limit):
        torch.cuda.synchronize()
        began = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        micros = next(loader)
        assert len(micros) == plan["R"][step] and all(int(b["step"]) == step for b in micros)
        assert all(len(b["pid"]) <= plan["budget"] for b in micros)
        own = np.sum([np.asarray(b["own"]).sum(0) for b in micros], axis=0)
        total = T.allsum([float(x) for x in own[:5]], dev)
        batches = [T.prep(b, dev, max(total[0] / T.WORLD, 1), total[1] / T.WORLD, total[3] / T.WORLD) for b in micros]
        values, _, extra = T.train_step_multi(model, batches, rot, dos)
        if not pose_only:
            assert abs(extra["dec_s_N"] - total[2]) < .5 and abs(extra["dec_p_N"] - total[4]) < .5
        norms = {}
        for group in opt.param_groups:
            norm = torch.nn.utils.clip_grad_norm_(group["params"], C["clip"][group["name"]])
            norms[group["name"]] = float(norm.full_tensor() if hasattr(norm, "full_tensor") else norm)
        lr = {group["name"]: group["lr"] for group in opt.param_groups}
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        memory = torch.cuda.memory_stats()
        timing = torch.tensor([time.perf_counter() - began, torch.cuda.max_memory_allocated() / 2**30,
                               torch.cuda.max_memory_reserved() / 2**30, memory.get("num_alloc_retries", 0),
                               memory.get("num_ooms", 0)], device=dev)
        dist.all_reduce(timing, op=dist.ReduceOp.MAX)
        times.append(float(timing[0])); peaks.append(float(timing[1]))
        reserved_peaks.append(float(timing[2])); allocator_retries.append(int(timing[3])); allocator_ooms.append(int(timing[4]))
        metrics = media_metrics(T, T.allsum(values, dev), text_loss=compact or cb1)
        metrics.update({f"train/lr_{k}": v for k, v in lr.items()})
        metrics.update({f"grad_norm/{k}": v for k, v in norms.items()})
        for name, do in dos.items():
            metrics[f"train/lr_{name}"] = extra[f"{name}_lrs"][-1]
            metrics[f"grad_norm/{name}"] = max(do.grad_norms)
        done = step + 1
        emit(metrics, done)
        if T.RANK == 0:
            runtime = dict(step=done, step_seconds=times[-1], peak_allocated_gib=peaks[-1],
                           peak_reserved_gib=reserved_peaks[-1], allocator_retries=allocator_retries[-1],
                           allocator_ooms=allocator_ooms[-1],
                           updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            tmp_runtime = root / "runtime.json.tmp"
            tmp_runtime.write_text(json.dumps(runtime) + "\n")
            tmp_runtime.replace(root / "runtime.json")
        T.log(f"step {done}/200: {times[-1]:.2f}s, peak={peaks[-1]:.1f}GiB pose={metrics['train/pose_loss']:.4f}"
              + (f" audio={metrics['train/audio_loss']:.4f}" if not pose_only else "")
              + (f" text={metrics['train/text_loss']:.4f}" if compact or cb1 else "")
              + (f" pcb0={metrics['train/pose_cb0_loss']:.4f} pcb1={metrics['train/pose_cb1_loss']:.4f}" if cb1 else ""))
        del batches, micros
        gc.collect(1)
        stop = torch.tensor([int(T.STOP["flag"] or (root / "control/STOP_AFTER_SAVE").exists())], device=dev)
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        requested_stop = bool(int(stop))
        if not bench:
            if done in C["checkpoint_steps"] or (requested_stop and not args.final_checkpoint_only):
                T.save_full(model, opt, sched, dos, done, "stop" if requested_stop else "scheduled", plan["digest"])
                T.export_local(model, done, str(Path(os.environ['ALOE_LOCAL_EXPORT_ROOT']) / f"export-{done}"))
                if T.LOCAL == 0:
                    (root / "LATEST").write_text(str(Path(os.environ['ALOE_SAVE_ROOT']) / f"step-{done:05d}") + '\n')
            if requested_stop:
                if T.RANK == 0:
                    saved = done in C["checkpoint_steps"] or not args.final_checkpoint_only
                    marker = "STOPPED" if saved else "STOPPED_WITHOUT_SAVE"
                    (root / "control" / marker).write_text(f"{done}\n")
                    T.log(f"[stop] step={done} checkpoint_saved={saved}")
                stop_done = done
                break
            if C["evaluation_enabled"] and (done % C["eval_every"] == 0 or done in C["checkpoint_steps"]):
                evaluate(done)
    if bench and T.RANK == 0:
        measured = times[args.benchmark_warmup:]
        assert measured
        report = dict(mode=args.mode, world=T.WORLD, warmup_steps=args.benchmark_warmup,
                      measured_step_seconds=measured, mean_step_seconds=float(np.mean(measured)),
                      estimated_200_steps_seconds=200 * float(np.mean(measured)),
                      excludes="startup, compilation, evaluation and checkpoint I/O", peak_gib=max(peaks),
                      all_step_seconds=times, plan_digest=plan["digest"], peak_reserved_gib=max(reserved_peaks),
                      allocator_retries=max(allocator_retries), allocator_ooms=max(allocator_ooms),
                      cuda_allocator=C["cuda_allocator"], compiler_cache=C["compiler_cache"],
                      completed_updates=len(times), max_shape_steps=[s + 1 for s in plan.get("stress_steps", []) if s < limit],
                      per_step_peak_gib=peaks, per_step_reserved_gib=reserved_peaks)
        report.update(benchmark_stress=args.benchmark_stress, micro_batch_rows=mbs, gradient_accumulation=ga,
                      tokens_per_micro=plan["budget"], media_frames_per_step=plan["media_frames_per_step"][:limit])
        (root / "benchmark.json").write_text(json.dumps(report, indent=2))
        T.log(f"[benchmark] {report}")
    if T.RANK == 0:
        metrics_file.close()
        if wb:
            wb.finish()
        if not bench and stop_done is None:
            (root / "TRAINING_COMPLETE").write_text("200\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
