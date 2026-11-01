#!/usr/bin/env python3
"""GenghisBench "before" number: base Qwen3-32B (the PPM init) with its own chat template (enable_thinking=False, no system prompt), vLLM
bf16, the same sim_20q v5 protocol as gb_ppm.py / the Qwen GB runs (OPEN turns, last_question, oracle v5, 20-question cap, 30-turn loop,
T 0.7, no top-k / top-p / min-p, max 1024 tokens per reply = the Qwen engines' max_pieces, context 12,288 tokens = their --max-len: a turn
whose prompt + 1024 would cross it ends the game as a fail). All games advance in lock-step, one vLLM batch per round (per-request seeds;
NOT batch-invariant).
  python gb_base_vllm.py --out <dir> --seeds 100-119 [--tp 2]"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gb_ppm import load_sim, parse_seeds  # noqa: E402

MODEL = "/data/home/amartyavarma/models/Qwen3-32B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="100-119")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    S = load_sim()
    os.makedirs(a.out, exist_ok=True)
    llm = LLM(MODEL, tensor_parallel_size=a.tp, dtype="bfloat16", max_model_len=a.max_len + 512, gpu_memory_utilization=0.9,
              enable_prefix_caching=True)
    tk = llm.get_tokenizer()
    seeds = parse_seeds(a.seeds)
    G = {s: {"seed": s, "msgs": [], "conv": [], "asked": 0, "unknown": [], "pending": list(S.OPEN), "done": None, "round": 0} for s in seeds}

    def next_user(g):
        """sim_20q.attempt after a model turn -> the next user text, or a final result."""
        if g["pending"]:
            return g["pending"].pop(0)
        t = g["conv"][-1]["model"]
        if S.is_win_text(t):
            g["done"] = True; return None
        if g["asked"] >= 20 or len(g["conv"]) - len(S.OPEN) >= 30:
            g["done"] = S.is_win_text(t); return None
        q = S.last_question(t)
        if q is None:
            return "Please ask your next yes or no question."
        g["asked"] += 1
        ans = S.oracle(q)
        if ans is None:
            g["unknown"].append(q); ans = "I'm not sure."
        return ans
    t0 = time.time()
    while True:
        batch, prompts = [], []
        for g in G.values():
            if g["done"] is not None:
                continue
            u = next_user(g)
            if u is None:
                continue
            g["msgs"].append({"role": "user", "content": u})
            g["conv"].append({"user": u, "model": None})
            p = tk.apply_chat_template(g["msgs"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if len(tk(p)["input_ids"]) + a.max_tokens > a.max_len:
                g["done"], g["error"] = False, "context full"
                g["conv"].pop()
                continue
            batch.append(g); prompts.append(p)
        if not batch:
            break
        sps = [SamplingParams(temperature=0.7, top_k=-1, top_p=1.0, min_p=0.0, max_tokens=a.max_tokens, seed=g["seed"] * 1000 + g["round"])
               for g in batch]
        outs = llm.generate(prompts, sps, use_tqdm=False)
        for g, o in zip(batch, outs):
            txt = o.outputs[0].text
            g["conv"][-1]["model"] = txt
            g["msgs"].append({"role": "assistant", "content": txt})
            g["round"] += 1
        print(f"[gb-base] round done: {len(batch)} active, {time.time() - t0:.0f}s", flush=True)
    res = []
    for g in G.values():
        r = {"seed": g["seed"], "success": bool(g["done"]), "questions": g["asked"] if "error" not in g else -1, "conv": g["conv"],
             "unknown": g["unknown"]}
        if "error" in g:
            r["error"] = g["error"]
        res.append(r)
    s = sum(r["success"] for r in res)
    qs = [r["questions"] for r in res if r["success"]]
    out = {"label": "qwen3-32b-base (chat template, no think)", "oracle": S.ORACLE_VERSION, "successes": s, "n": len(res),
           "median_questions_success": statistics.median(qs) if qs else None, "attempts": res}
    json.dump(out, open(f"{a.out}/gb_s{a.seeds}.json", "w"), indent=1)
    lines = [f"qwen3-32b-base: {s}/{len(res)} (oracle v5), median questions to success {out['median_questions_success']}"]
    lines += [f"  seed {r['seed']}: {'SUCCESS' if r['success'] else 'fail   '} q {r['questions']:3d}" + (f"  {r['error']}" if r.get('error') else "") for r in res]
    open(f"{a.out}/summary_s{a.seeds}.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
