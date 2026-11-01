#!/usr/bin/env python3
"""Native monoplex rollout with completed-frame pose interventions and keyed RNG.

The first complete batch is unperturbed. The second replays the same prefix,
batch shape and indexed uniforms, with independent single-codebook branches.
Instrumentation observes native kernels; it does not replace transformer math.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import types
from unittest.mock import patch

import numpy as np
import torch

CBS, DEPTH, CD, FPS, BATCH = 2048, 32, 2, 12.5, 36


def keyed_uniform(seed, frame, modality, codebook, stream="shared"):
    """Stable 52-bit midpoint variate, exactly inside (0,1) in float64."""
    key = f"pose-ar-v1:{int(seed)}:{int(frame)}:{modality}:{int(codebook)}:{stream}"
    bits = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") >> 12
    return (bits + .5) / (1 << 52)


def inverse_cdf(logits, uniforms, exclude=None):
    z = logits.double()
    if exclude is not None:
        z = z.clone()
        z.scatter_(1, exclude[:, None], float("-inf"))
    width = z.shape[-1]
    padding = (-width) % 128
    if padding:
        z = torch.nn.functional.pad(z, (0, padding), value=float("-inf"))
    p = torch.softmax(z, -1)
    cdf = p.cumsum(-1)
    # Renormalize cumulative roundoff, preserving zero-mass plateaus (including
    # excluded IDs and alignment padding). The final positive mass ends at1.
    cdf = cdf / cdf[:, -1:]
    u = torch.as_tensor(uniforms, dtype=torch.float64, device=z.device)
    q = torch.searchsorted(cdf.contiguous(), u[:, None], right=False)[:, 0]
    if bool((q >= width).any()) or (exclude is not None and bool((q == exclude).any())):
        raise ValueError("Inverse CDF selected a zero-mass or padded token")
    return q, p[:, :width]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_torch(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def log(event, **kw):
    print(json.dumps({"event": event, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}), flush=True)


class Recorder:
    """Store CPU tensors indexed by model frame, retaining native dtypes."""

    def __init__(self):
        self.active = False
        self.frame = -1
        self.pending = {}
        self.values = defaultdict(list)
        self.frames = defaultdict(list)
        self.reference = None
        self.baseline_pass = False
        self.branch_frame = 27
        self.checks = {"baseline_rows_equal": True, "repeat_exact": True,
                       "noop_exact": True, "branch_prefix_exact": True,
                       "independent_prefix_exact": True, "checked_tensors": 0}

    def begin(self, frame):
        self.frame, self.pending, self.active = frame, {}, True

    def push(self, key, value):
        if self.active:
            if key in self.pending:
                raise ValueError(f"Duplicate trace key {key} at frame {self.frame}")
            if value.shape[0] != BATCH:
                raise ValueError(f"Wrong batch axis for {key}: {tuple(value.shape)}")
            self.pending[key] = value.detach().clone()

    def finish(self):
        self.active = False
        for key, value in self.pending.items():
            cpu = value.cpu()
            if self.baseline_pass:
                if not torch.equal(cpu, cpu[:1].expand_as(cpu)):
                    raise ValueError(f"Identical baseline rows differ: {key}, frame {self.frame}")
                cpu = cpu[:1].clone()
            elif self.reference is not None:
                ref_frames = self.reference["frames"][key]
                index = ref_frames.index(self.frame)
                ref = self.reference["tensors"][key][index]
                if not torch.equal(cpu[:3], ref.expand_as(cpu[:3])):
                    delta = (cpu[:3].double() - ref.double()).abs().max().item()
                    raise ValueError(f"Repeat/no-op differs: {key}, frame {self.frame}, delta {delta}")
                if self.frame < self.branch_frame and not torch.equal(cpu[3:35], ref.expand_as(cpu[3:35])):
                    raise ValueError(f"Intervention prefix differs: {key}, frame {self.frame}")
                if self.frame <= self.branch_frame and not torch.equal(cpu[35:], ref.expand_as(cpu[35:])):
                    raise ValueError(f"Independent continuation prefix differs: {key}, frame {self.frame}")
            self.checks["checked_tensors"] += 1
            self.values[key].append(cpu)
            self.frames[key].append(self.frame)
        self.pending = {}

    def result(self):
        return {"frames": dict(self.frames), "tensors": {k: torch.stack(v) for k, v in self.values.items()},
                "checks": dict(self.checks)}


class Instrumentation:
    def __init__(self, engine, module, recorder):
        self.engine, self.FP, self.rec = engine, module, recorder
        self.seed = 0
        self.continuation_noise = False
        self.counts = {}
        self.handles = []
        self.originals = []
        self.layers = {}
        for name, stack in (("backbone", engine.bb), ("speech_tower", engine.ts), ("pose_tower", engine.tp)):
            for i, layer in enumerate(stack.L):
                self.layers[id(layer)] = (name, i, len(stack.L))
        self.install()

    def replace(self, obj, name, replacement):
        self.originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    def install(self):
        e, rec = self.engine, self.rec
        native_layer = e._layer_step

        def layer_step(_engine, layer, h, *args):
            name, i, count = self.layers[id(layer)]
            if i == 0:
                rec.push(f"input/{name}", h[0])
            out = native_layer(layer, h, *args)
            taps = {0: "early", (count - 1) // 2: "middle", count - 1: "final"}
            if i in taps:
                rec.push(f"hidden/{name}/{taps[i]}", out[0])
            return out

        self.replace(e, "_layer_step", types.MethodType(layer_step, e))
        self.replace(e, "_text", types.MethodType(self.text, e))
        for modality, decoder in (("speech", e.dec_s), ("pose", e.dec_p)):
            native_step = decoder.step
            def depth_step(x, j, native=native_step, modality=modality):
                rec.push(f"input/{modality}_light/{j + 1}", x)
                result = native(x, j)
                rec.push(f"hidden/{modality}_light/{j + 1}/final_norm", result)
                return result
            self.replace(decoder, "step", depth_step)
        for name, module in (("backbone.pose", e.m.inp.pose), ("backbone.speech_A", e.m.inp.aud[0]),
                             ("backbone.speech_B", e.m.inp.aud[1]),
                             ("pose_tower.pose", e.m.tower_p.code_emb), ("speech_tower.speech", e.m.tower_s.code_emb)):
            def code_hook(module, args, output, name=name):
                if not rec.active:
                    return
                codes = args[0]
                rec.push(f"component/{name}/sum", output)
                rec.push(f"component/{name}/codes", codes)
                rows = torch.nn.functional.embedding(codes.clamp_min(0), module.weight).float()
                rows = rows * (codes >= 0)[..., None]
                rec.push(f"component/{name}/per_code_rms", rows.square().mean(-1).sqrt())
                rec.push(f"component/{name}/per_code_delta_rms", (rows - rows[:1]).square().mean(-1).sqrt())
            self.handles.append(module.register_forward_hook(code_hook))
        for name, module in (("backbone.text_A", e.m.inp.emb[0]), ("backbone.text_B", e.m.inp.emb[1]),
                             ("pose_tower.text", e.m.tower_p.text_emb), ("pose_tower.conditioning", e.m.tower_p.cond),
                             ("speech_tower.text", e.m.tower_s.text_emb), ("speech_tower.conditioning", e.m.tower_s.cond)):
            def component_hook(module, args, output, name=name):
                rec.push(f"component/{name}/value", output)
            self.handles.append(module.register_forward_hook(component_hook))

    def uniforms(self, modality, codebook):
        frame = self.rec.frame
        shared = keyed_uniform(self.seed, frame, modality, codebook)
        u = [shared] * BATCH
        if self.continuation_noise and frame > self.rec.branch_frame:
            u[35] = keyed_uniform(self.seed, frame, modality, codebook, "independent-continuation")
        return u

    def draw(self, logits, modality, codebook):
        q, p = inverse_cdf(logits, self.uniforms(modality, codebook))
        # Native head GEMM outputs are BF16 values upcast to FP32; this saves
        # storage losslessly. Reject any unexpected higher-precision logits.
        compact = logits.to(torch.bfloat16)
        if not torch.equal(compact.float(), logits.float()):
            raise ValueError(f"Non-BF16 native logits for {modality}/{codebook}")
        self.rec.push(f"logits/{modality}/{codebook}", compact)
        self.rec.push(f"sampled/{modality}/{codebook}", q)
        self.rec.push(f"sample_probability/{modality}/{codebook}", p.gather(1, q[:, None])[:, 0])
        aligned_p = torch.nn.functional.pad(p, (0, (-p.shape[-1]) % 128))
        self.rec.push(f"entropy/{modality}/{codebook}", -(aligned_p * aligned_p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(-1))
        self.rec.push(f"uniform/{modality}/{codebook}", torch.tensor(self.uniforms(modality, codebook), dtype=torch.float64, device=q.device))
        return q

    def sample(self, logits, generator, temperature, topk, topp=None):
        e = self.engine
        modality = "pose" if generator is e.gen_p else "speech" if generator is e.gen_a else None
        if modality is None:
            raise ValueError("Unexpected sampling generator")
        cb = self.counts.get(modality, 0)
        self.counts[modality] = cb + 1
        if cb >= DEPTH:
            raise ValueError("Unexpected depth sampling order")
        return self.draw(logits, modality, cb)

    def text(self, e):
        raw = e.text_logits(e.hn_prev[:e.NI])
        e.zt.copy_(raw)
        q = self.draw(raw[:, :e.n_in], "text", -1)
        T = e.Tv[e.dl][:e.NI]
        e.laneB.scatter_(1, T[:, None], q[:, None])
        e.tok_now.copy_(q)
        e.adapt[:, 0].zero_()
        e.adapt[:, 1:].fill_(1.)
        e.ring.push(q)

    def close(self):
        for handle in self.handles:
            handle.remove()
        for obj, name, original in reversed(self.originals):
            setattr(obj, name, original)


def neutral_params(FP):
    p = dict(FP.DEFAULTS)
    for key in p:
        if key.endswith("temp"):
            p[key] = 1.
        elif key.endswith("topk"):
            p[key] = 0
    p.update(text_topp=1., min_p=0., freq_pen=0., text_rep=1., audio_rep_pen=1.,
             pose_rep_cb0=1., pose_rep_light=1., word_greedy=0, adaptive_rep_on=0,
             pose_cfg_on=0, pose_cfg_s=1., pose_cfg_cb0=0, pose_cfg_light=0)
    return p


def interventions(baseline, seed, physical_frame):
    f = physical_frame + CD
    key = "tokens/pose"
    original_frame = baseline["tensors"][key][baseline["frames"][key].index(f), 0].clone()
    forced = original_frame[None].expand(BATCH, -1).clone()
    details = []
    for cb in range(DEPTH):
        key = f"logits/pose/{cb}"
        z = baseline["tensors"][key][baseline["frames"][key].index(f), :1].double()
        original = int(original_frame[cb] - cb * CBS)
        u = keyed_uniform(seed, f, "pose", cb, "intervention-exclude-original")
        replacement, conditional = inverse_cdf(z, [u], torch.tensor([original]))
        replacement = int(replacement[0])
        p = z.softmax(-1)[0]
        assert replacement != original
        forced[3 + cb, cb] = replacement + cb * CBS
        details.append({"row": 3 + cb, "codebook": cb, "original": original, "replacement": replacement,
                        "p_original": float(p[original]), "p_replacement": float(p[replacement]),
                        "excluded_mass": float(1 - p[original]), "conditional_replacement_probability": float(conditional[0, replacement]),
                        "uniform": u, "stream": "intervention-exclude-original"})
    return forced.numpy(), details


def run_pass(engine, FP, inst, seed, params, n_frames, branch_frame, *, baseline=None, force=None):
    rec = Recorder()
    rec.baseline_pass = baseline is None
    rec.reference = baseline
    rec.branch_frame = branch_frame + CD
    instrumentation = Instrumentation(engine, FP, rec)
    instrumentation.seed = seed
    instrumentation.continuation_noise = baseline is not None
    try:
        engine.start([inst] * BATCH, seed, params, f0=0)
        with patch.object(FP, "t_sample", instrumentation.sample):
            for f in range(n_frames + CD):
                started = time.perf_counter()
                rec.begin(f)
                instrumentation.counts = {}
                intervention = {"pose": force} if force is not None and f == branch_frame + CD else None
                engine.step(f, graphs=False, cold_pad=False, force=intervention)
                rec.push("hidden/backbone/final_norm", engine.hn_prev)
                rec.push("hidden/pose_tower/final_norm", engine.hv_p)
                rec.push("hidden/speech_tower/final_norm", engine.hv_s)
                rec.push("tokens/text", engine.tok_now)
                if f >= CD:
                    expected = {"pose": DEPTH, "speech": DEPTH}
                    if instrumentation.counts != expected:
                        raise ValueError(f"Sampling order differs: {instrumentation.counts}")
                    rec.push("tokens/pose", engine.pkp[engine.dp])
                    rec.push("tokens/speech", engine.pk[engine.dl][:, 1:])
                rec.finish()
                log("rollout_frame", pass_kind="baseline" if baseline is None else "branches", seed=seed,
                    model_frame=f, physical_frame=f-CD, seconds=round(time.perf_counter()-started, 3))
    finally:
        instrumentation.close()
    return rec.result()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("/home/param/models/imported-weights/ppm-aloe-monoplex-pose-raw-bandfix-355/weights/export-355"))
    p.add_argument("--engine-dir", type=Path, default=Path("/home/param/websites/pose-monoplex-engine/ppm"))
    p.add_argument("--contexts", type=Path, default=Path("/home/param/outputs/pose-ar-diagnostics-20261030/contexts/manifest.json"))
    p.add_argument("--context-ids", default="c0-side0,c1-side0,c2-side0,c3-side0")
    p.add_argument("--devices", default="0,1")
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--frames", type=int, default=51)
    p.add_argument("--branch-frame", type=int, default=25)
    p.add_argument("--output-dir", type=Path, default=Path("/home/param/outputs/pose-ar-rollout-20261030"))
    args = p.parse_args()
    if not 0 <= args.branch_frame < args.frames:
        p.error("Branch frame must be within the physical rollout")
    os.environ.setdefault("PPM_CD", "2")
    os.environ.setdefault("PPM_AUTOTUNE", "0")
    os.environ.setdefault("FA3_DIR", "/home/param/pose-repo/.cache/kernels/vllm-flash-attn3")
    sys.path.insert(0, str(args.engine_dir))
    import fast_ppm as FP
    import transformers
    devices = [torch.device(f"cuda:{int(x)}") for x in args.devices.split(",")]
    torch.cuda.set_device(devices[0])
    requested = args.context_ids.split(",")
    contexts = {c["id"]: c for c in json.loads(args.contexts.read_text())["contexts"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = neutral_params(FP)
    report = {"status": "running", "schema_version": 1, "checkpoint": str(args.checkpoint), "model_id": "mono355",
              "methodology": {"physical_frames": args.frames, "fps": FPS, "code_delay": CD,
                              "branch_physical_frame": args.branch_frame, "branch_model_frame": args.branch_frame + CD,
                              "branch_time_seconds": args.branch_frame / FPS, "strictly_following_frames": args.frames - args.branch_frame - 1,
                              "batch_size": BATCH, "row_mapping": {"0": "baseline_repeat", "1": "duplicate_control", "2": "forced_original_noop",
                                                                      "3..34": "intervene codebook row-3", "35": "independent_continuation"},
                              "mutation": "Completed sampled pose frame: change exactly one CB, preserve other same-frame tokens; force before current backbone input",
                              "conditioning": "Native monoplex voice prefix only, absent channel A, generated text/speech/pose thereafter",
                              "sampling": "Temperature1, raw native heads, text restricted only to valid input IDs, no CFG/repetition/filtering/cold PAD",
                              "randomness": "SHA256 pose-ar-v1:{seed}:{model_frame}:{modality}:{codebook}:{stream}, first52bits with exact half-bin offset; FP64 inverse CDF",
                              "independent_continuation": "Row35 shares history through branch frame; distinct stream from the next model frame onward",
                              "light_taps": "Native BF16 head-input final_norm only; fused internal residuals are not changed or instrumented",
                              "logit_storage": "Lossless BF16 encoding of native BF16 GEMM logits upcast to FP32; FP64 softmax/CDF/entropy use aligned multiples of128 with zero-probability padding outside the valid support",
                              "component_vectors": "Native summed component vectors plus consumed IDs and per-code RMS/delta-to-row0 RMS; per-code full vectors reconstruct from exact checkpoint",
                              "params": params},
              "runtime": {"python": sys.version, "pytorch": torch.__version__, "transformers": transformers.__version__,
                          "cuda": torch.version.cuda, "platform": platform.platform(), "devices": [str(d) for d in devices],
                          "gpus": [torch.cuda.get_device_name(d) for d in devices], "command": [sys.executable, *sys.argv],
                          "source_hashes": {str(q): sha(q) for q in [Path(__file__), args.engine_dir/"fast_ppm.py", args.engine_dir/"model_ppm.py",
                                                                      args.engine_dir/"light_exact_ppm.py", args.contexts]}}, "experiments": []}
    atomic_json(args.output_dir/"rollout-manifest.json", report)
    with torch.no_grad():
        model, layer_devices, cfgs, info = FP.build_model(str(args.checkpoint), devices)
        report["native_model"] = info
    with torch.inference_mode():
        engine = FP.FastPPM(model, layer_devices, cfgs, devices, n_inst=BATCH, cfg_rows=False,
                            max_s=max(6., args.frames/FPS), p_max=max(contexts[c]["P"] for c in requested),
                            mimi_stream=False, front=False)
        for index, context_id in enumerate(requested):
            c = contexts[context_id]
            seed = args.seed + 1009 * index
            with np.load(c["path"], allow_pickle=False) as z:
                inst = {k: z[k].copy() for k in ("ids", "depth", "present", "pdepth")}
            inst.update(P=c["P"], a_mode="empty")
            exp_dir = args.output_dir/context_id
            exp_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            baseline = run_pass(engine, FP, inst, seed, params, args.frames, args.branch_frame)
            atomic_torch(exp_dir/"baseline.pt", baseline)
            forced, details = interventions(baseline, seed, args.branch_frame)
            branches = run_pass(engine, FP, inst, seed, params, args.frames, args.branch_frame, baseline=baseline, force=forced)
            atomic_torch(exp_dir/"branches.pt", branches)
            prefix = {k: torch.from_numpy(inst[k][:, :c["P"]].copy() if k != "pdepth" else inst[k][:c["P"]].copy())
                      for k in ("ids", "depth", "present", "pdepth")}
            atomic_torch(exp_dir/"prefix.pt", prefix)
            experiment = {"context_id": context_id, "seed": seed, "P": c["P"], "source_context": c["path"],
                          "source_sha256": sha(c["path"]), "baseline": str(exp_dir/"baseline.pt"), "branches": str(exp_dir/"branches.pt"),
                          "prefix": str(exp_dir/"prefix.pt"), "interventions": details,
                          "checks": {"baseline": baseline["checks"], "branches": branches["checks"]},
                          "wall_seconds": time.perf_counter()-started}
            atomic_json(exp_dir/"experiment.json", experiment)
            report["experiments"].append(experiment)
            atomic_json(args.output_dir/"rollout-manifest.json", report)
            log("rollout_context_complete", context_id=context_id, seconds=experiment["wall_seconds"])
            del baseline, branches
        report["status"] = "complete"
        atomic_json(args.output_dir/"rollout-manifest.json", report)
        log("rollout_complete", output=str(args.output_dir))


if __name__ == "__main__":
    main()
