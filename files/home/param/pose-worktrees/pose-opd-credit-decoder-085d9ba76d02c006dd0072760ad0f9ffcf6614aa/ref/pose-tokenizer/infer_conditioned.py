#!/usr/bin/env python3
"""Inference for the prompt-conditioned model trained by train.py.

Row layout the model was trained on (see build_simple_from_prompted.py):

    [SOP] <question tokens> [EOP] <pose_frame_0> [SOG] t0 a0 pose_frame_0 t1 a1 pose_1 ... [EOG]

This script builds that prompt up to (and including) SOG and lets the model
generate the body. The pose_frame_0 conditioning seed is taken from a RANDOM
example in the eval split (the dataset/split named in TRAIN.yaml), so every run
poses a fixed set of questions against a real frame-0 pose.

It reuses infer.py for token-band parsing, Mimi audio decode and pose rendering.

Usage:
    export HF_TOKEN=...                       # read model + (private) dataset
    python infer_conditioned.py \
        --model /mnt/somfs/.../run_1/checkpoint-500 \
        --output-dir cond_infer_out \
        --render                              # optional: decode pose mp4 + mimi wav

The questions default to the five requested ones; override with --questions.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import infer  # token-band helpers + decode/render machinery

DEFAULT_QUESTIONS = [
    "What is the capital of France?",
    "What are you wearing to work today?",
    "Who is the oldest person in the world?",
    "When did the USA become a great power?",
    "What is the main difference between plants and fungi?",
]

# Conditioned-prompt special tokens (must match build_simple_from_prompted.py).
BASE = 151936
SOP = BASE + 3   # 151939
EOP = BASE + 4   # 151940
SOG = BASE + 1   # 151937


def _token() -> str | None:
    return (os.environ.get("HF_TOKEN") or os.environ.get("hf_token")
            or os.environ.get("xabi_token") or os.environ.get("canopy_token"))


def load_yaml(path: str | Path) -> dict:
    if not path or not Path(path).is_file():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def extract_pose0(input_ids: list[int]) -> list[int] | None:
    """The NUM_POSE_CODEBOOKS pose tokens immediately before the first SOG =
    the frame-0 pose conditioning seed. Returns None if not well-formed."""
    try:
        k = input_ids.index(SOG)
    except ValueError:
        return None
    n = infer.NUM_POSE_CODEBOOKS
    pose0 = [int(t) for t in input_ids[k - n:k]]
    if len(pose0) == n and all(infer.POSE_START <= t < infer.POSE_END for t in pose0):
        return pose0
    return None


def pick_pose0_from_eval(dataset: str, split: str, token: str | None,
                         seed: int, max_scan: int = 200) -> tuple[list[int], int]:
    """Load the eval split and return a random example's frame-0 pose."""
    ds = load_dataset(dataset, split=split, token=token)
    rng = random.Random(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    for idx in order[:max_scan]:
        pose0 = extract_pose0([int(t) for t in ds[idx]["input_ids"]])
        if pose0 is not None:
            return pose0, idx
    raise RuntimeError(f"No well-formed frame-0 pose found in {dataset}[{split}] "
                       f"(scanned {min(max_scan, len(ds))} rows)")


def build_prompt_ids(tokenizer, question: str, pose0: list[int],
                     include_sog: bool = True) -> list[int]:
    qids = [int(t) for t in tokenizer(question, add_special_tokens=False)["input_ids"]
            if t < BASE]
    ids = [SOP] + qids + [EOP] + pose0
    if include_sog:
        ids.append(SOG)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/INFERENCE.yaml",
                    help="Inference YAML for generation/render defaults.")
    ap.add_argument("--train-config", default="configs/TRAIN.yaml",
                    help="Training YAML for dataset/eval_split/model defaults.")
    ap.add_argument("--model", required=True,
                    help="Checkpoint dir or HF repo (REQUIRED). e.g. "
                         "/mnt/somfs/.../run_1/checkpoint-500")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--num-pose-codebooks", type=int, default=None)
    ap.add_argument("--dataset", default=None,
                    help="Eval dataset repo (default: TRAIN.yaml dataset).")
    ap.add_argument("--eval-split", default=None,
                    help="Eval split (default: TRAIN.yaml eval_split or 'train[-50:]').")
    ap.add_argument("--questions", nargs="*", default=None,
                    help="Override the default 5 questions.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for picking the pose-conditioning example + sampling.")
    ap.add_argument("--same-pose", action="store_true", default=True,
                    help="Use the SAME random pose0 for every question (default).")
    ap.add_argument("--per-question-pose", dest="same_pose", action="store_false",
                    help="Draw a fresh random pose0 per question.")
    ap.add_argument("--no-sog", action="store_true",
                    help="Do not append SOG to the prompt (let the model emit it).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-dir", default="cond_infer_out")
    ap.add_argument("--render", action="store_true",
                    help="Decode generated pose -> mp4 and Mimi speech -> wav.")
    # generation overrides (else taken from INFERENCE.yaml)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=None)
    args = ap.parse_args()

    icfg = load_yaml(args.config)
    tcfg = load_yaml(args.train_config)
    gen = icfg.get("generation", {})
    rcfg = icfg.get("render", {})

    npose = int(args.num_pose_codebooks or icfg.get("num_pose_codebooks")
                or tcfg.get("num_pose_codebooks") or 4)
    infer.configure_token_layout(npose)

    device = args.device or icfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    token = _token()
    tok_name = args.tokenizer or icfg.get("tokenizer") or tcfg.get("tokenizer_name") or tcfg.get("model_name")
    model_name = args.model

    dataset = args.dataset or tcfg.get("dataset")
    eval_split = args.eval_split or tcfg.get("eval_split") or "train[-50:]"
    questions = args.questions if args.questions else DEFAULT_QUESTIONS

    gen_kwargs = dict(
        max_new_tokens=int(args.max_new_tokens or gen.get("max_new_tokens", 1024)),
        do_sample=bool(gen.get("do_sample", True)),
        temperature=float(args.temperature if args.temperature is not None else gen.get("temperature", 0.7)),
        top_k=int(args.top_k if args.top_k is not None else gen.get("top_k", 50)),
        top_p=float(args.top_p if args.top_p is not None else gen.get("top_p", 1.0)),
        repetition_penalty=float(args.repetition_penalty if args.repetition_penalty is not None
                                 else gen.get("repetition_penalty", 1.05)),
    )

    print(f"[setup] model={model_name}\n        tokenizer={tok_name}\n"
          f"        dataset={dataset} [{eval_split}]  pose_cb={npose}  device={device}")

    if not icfg.get("enable_cudnn", False):
        torch.backends.cudnn.enabled = False

    tokenizer = AutoTokenizer.from_pretrained(tok_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, token=token,
        attn_implementation="flash_attention_2",
    ).to(device)
    infer.ensure_tokenizer_vocab(tokenizer, model)
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shared_pose0, shared_idx = pick_pose0_from_eval(dataset, eval_split, token, args.seed)
    print(f"[pose0] seed example row={shared_idx}  pose0={shared_pose0}")

    results = []
    for qi, question in enumerate(questions):
        if args.same_pose:
            pose0, pose_row = shared_pose0, shared_idx
        else:
            pose0, pose_row = pick_pose0_from_eval(dataset, eval_split, token, args.seed + qi + 1)

        prompt_ids = build_prompt_ids(tokenizer, question, pose0, include_sog=not args.no_sog)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)

        torch.manual_seed(args.seed + qi)
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=gen_kwargs["max_new_tokens"],
                do_sample=gen_kwargs["do_sample"],
                temperature=gen_kwargs["temperature"] if gen_kwargs["do_sample"] else None,
                top_k=gen_kwargs["top_k"] if gen_kwargs["do_sample"] and gen_kwargs["top_k"] > 0 else None,
                top_p=gen_kwargs["top_p"] if gen_kwargs["do_sample"] else None,
                repetition_penalty=gen_kwargs["repetition_penalty"],
                eos_token_id=infer.EOG, pad_token_id=infer.TEXT_PAD,
            )[0].detach().cpu().tolist()

        continuation = out[len(prompt_ids):]
        cont_summary = infer.summarize(continuation, tokenizer)
        rec = {
            "index": qi,
            "question": question,
            "pose_seed_row": pose_row,
            "pose0": pose0,
            "prompt_ids": prompt_ids,
            "continuation_ids": continuation,
            "continuation": cont_summary,
        }

        print(f"\n=== Q{qi}: {question}")
        print(f"    bands={cont_summary['band_counts']}")
        txt = cont_summary["decoded_text"].strip()
        if txt:
            print(f"    text: {txt[:300]}")

        if args.render:
            speech_tokens, pose_tokens, frames = infer.parse_speech_pose_tokens_from_ids(continuation)
            stem = out_dir / f"q{qi}"
            try:
                if pose_tokens and pose_tokens[0]:
                    infer.render_pose_on_black(
                        pose_tokens,
                        rcfg.get("pose_tokenizer", "xabirizar9/pose-tokenizer-2x-12cb-lossshift1"),
                        stem.with_suffix(".mp4"), device,
                        int(rcfg.get("width", 768)), int(rcfg.get("height", 768)),
                        float(rcfg.get("fps", 12.5)),
                        float(rcfg.get("synthetic_shoulder_width", 0.16)),
                        float(rcfg.get("confidence_threshold", 0.3)),
                        int(rcfg.get("pose_decode_shift_frames", 1)),
                        None,
                    )
                    rec["pose_mp4"] = str(stem.with_suffix(".mp4"))
                if speech_tokens and speech_tokens[0]:
                    infer.decode_mimi_tokens(speech_tokens, stem.with_name(f"q{qi}_mimi.wav"), device)
                    rec["mimi_wav"] = str(stem.with_name(f"q{qi}_mimi.wav"))
            except Exception as exc:  # noqa: BLE001
                rec["render_error"] = str(exc)
                print(f"    [render error] {exc}")

        (out_dir / f"q{qi}.json").write_text(json.dumps(rec, indent=2))
        results.append(rec)

    (out_dir / "summary.json").write_text(json.dumps(
        {"model": model_name, "dataset": dataset, "eval_split": eval_split,
         "num_pose_codebooks": npose, "generation": gen_kwargs,
         "results": [{k: r[k] for k in ("index", "question", "pose_seed_row",
                                         "continuation")} for r in results]},
        indent=2))
    print(f"\n[done] wrote {len(results)} results to {out_dir}/ (summary.json)")


if __name__ == "__main__":
    main()
