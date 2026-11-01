#!/usr/bin/env python3
"""GenghisBench (GB) for PPM checkpoints: sim_20q's 20 questions (v5 rule oracle thinking of Genghis Khan) played through the PPM text
chat exactly as infer_ppm.run_chat plays it, one game at a time per 2-GPU worker (batch-invariant by construction: every game has its own
seeds, so a game's result does not depend on which worker plays it or what else runs).

  python gb_ppm.py worker --ckpt <ckpt> --out <dir> --seeds 100-119 [--devs 0,1]     claim unplayed seeds from <dir> one by one, play them
  python gb_ppm.py merge  --out <dir> [--label L]                                     <dir>/games/*.json -> <dir>/gb_s<seeds>.json + summary_s<seeds>.txt
  python gb_ppm.py parity --ckpt <ckpt> --out <dir>                                   ChatSession vs run_chat on the same turns (bitwise)
  (wrapper: gb_ppm.sh <ckpt> [n] [seed0] launches workers on slinky-7 / slinky-15 + a merge job)

PROTOCOL (= the Qwen GB runs, gbench_batched.Bench.play / sim_20q.attempt, line for line):
  oracle       gb_v5/sim_20q_v5.py (ORACLE_VERSION v5, 73/73 unit tests): OPEN (the 2 opening user turns), last_question (ONLY the final
               question sentence is read), oracle(q) (unknown -> "I'm not sure."), is_win_text; no question -> "Please ask your next yes
               or no question."; 20-question cap, 30-turn loop; success = the model names Genghis / Chinggis / Temujin.
  sampling     text temperature 0.7, no top-k (Qwen GB: --temp 0.7 --top-k 0), no top-p, no min-p, CFG off, bf16. Lane grammar = run_chat's
               ('chat': word pieces / PAD / <|im_end|>, no NEW_WORD). One torch.Generator per game seeded with the game seed.
  chat render  infer_ppm.run_chat: no system prompt (Qwen GB sys None), PREPEND frame, user turn k = chat_user_frames(None, u, seed + k)
               on lane A with B forced PAD (first turn after its render lead, later turns after a fixed 10-frame gap), then B free-runs
               frame by frame until <|im_end|> (or --max-turn-s 120 s = 1,500 frames, the run_chat default; the turn is then cut and the
               game goes on, like the Qwen engines' max_pieces cap). The reply text = the decoded word pieces of B's lane.
  context      7,500 frames: a user turn or a reply that would cross it ends the game as a fail (sim_20q GameOver, questions -1)."""
import argparse
import glob
import json
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ORACLE = "/data/home/amartyavarma/qwen-1ch/infer/gb_v5/sim_20q_v5.py"
CTX = 7500


def load_sim():
    sys.path.insert(0, "/data/home/amartyavarma/qwen-1ch/infer/gb_v5")
    from oracle_lib import load
    S = load(ORACLE, "sim_v5")
    assert S.ORACLE_VERSION == "v5", S.ORACLE_VERSION
    return S


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


class GameOver(Exception):
    pass


class ChatSession:
    """infer_ppm.run_chat's loop, one user turn per call (same pushes, same backbone calls, same sampling call order)."""

    def __init__(self, IP, g, seed, temp, topk, max_turn_s, system=""):
        import numpy as np
        self.IP, self.g, self.np = IP, g, np
        self.seed, self.temp, self.topk = seed, temp, topk
        self.tk = IP.tok()
        self.ids, self.pres = [[], []], [[], []]
        self.maxf = int(round(max_turn_s * IP.FPS))
        self.lane = IP.Grammar(g.V, g.dl, "chat", True)
        self.T, self.hlast, self.k = 0, None, 0
        if system:
            for t in self.tk(system, add_special_tokens=False)["input_ids"]:
                self.push(IP.PAD, int(t), 0.0, 1.0)
        self.push(IP.PAD, IP.PAD)                                          # the 1ch PREPEND frame
        g.gen.manual_seed(int(seed))

    def push(self, ta, tb, pa=1.0, pb=1.0):
        self.ids[0].append(ta); self.ids[1].append(tb); self.pres[0].append(pa); self.pres[1].append(pb)

    def say(self, u):
        IP, g, np = self.IP, self.g, self.np
        ua, gap, lead = IP.chat_user_frames(None, u, self.seed + self.k)
        npre = (lead - 1) if self.k == 0 else 10
        if len(self.ids[0]) + npre + len(ua) >= CTX:
            raise GameOver(f"context full ({len(self.ids[0])} + {npre} + {len(ua)} user frames >= {CTX})")
        for _ in range(npre):
            self.push(IP.PAD, IP.PAD)
        for t in ua:
            self.push(int(t), IP.PAD)
        n0 = len(self.ids[0])
        arr = np.array(self.ids, np.int64)[:, self.T:n0]
        pr = np.array(self.pres, np.float32)[:, self.T:n0]
        h = g.bb.run(g.embed(arr, None, pr, None), self.T)
        self.hlast = g.hnorm(h)[-1:]
        self.T = n0
        start, reply = self.T, []
        for _ in range(self.maxf):
            if self.T >= CTX:
                raise GameOver(f"context full during the reply ({self.T} frames)")
            z = g.text_logits(self.hlast)
            tb = IP.sample(z, self.temp, self.topk, g.gen, self.lane.mask(self.ids[1][-1]))
            self.push(IP.PAD, tb)
            reply.append(tb)
            e = g.embed(np.array([[IP.PAD], [tb]], np.int64), None, np.ones((2, 1), np.float32), None)
            self.hlast = g.hnorm(g.bb.run(e, self.T))
            self.T += 1
            if tb == IP.EOB:
                break
        self.k += 1
        words = [t for t in reply if t < IP.TE]
        return {"text": self.tk.decode(words), "frames": len(reply), "ended_with_im_end": bool(reply and reply[-1] == IP.EOB),
                "user_frames": len(ua), "start_frame": start}


def play(IP, g, S, seed, a):
    """sim_20q.attempt / gbench_batched.Bench.play with /say = ChatSession.say."""
    cs = ChatSession(IP, g, seed, a.temp, a.topk, a.max_turn_s)
    cl, asked, unknown, meta = [], 0, [], []

    def say(u):
        r = cs.say(u)
        meta.append({k: r[k] for k in ("frames", "ended_with_im_end", "user_frames", "start_frame")})
        return r["text"]
    try:
        for u in S.OPEN:
            t = say(u); cl.append({"user": u, "model": t})
        for _ in range(30):
            t = cl[-1]["model"]
            if S.is_win_text(t):
                return {"seed": seed, "success": True, "questions": asked, "conv": cl, "unknown": unknown, "turn_meta": meta, "frames": cs.T}
            if asked >= 20:
                break
            q = S.last_question(t)
            if q is None:
                u = "Please ask your next yes or no question."
            else:
                asked += 1
                ans = S.oracle(q)
                if ans is None:
                    unknown.append(q); ans = "I'm not sure."
                u = ans
            cl.append({"user": u, "model": say(u)})
        ok = S.is_win_text(cl[-1]["model"])
        return {"seed": seed, "success": ok, "questions": asked, "conv": cl, "unknown": unknown, "turn_meta": meta, "frames": cs.T}
    except GameOver as e:
        return {"seed": seed, "success": False, "questions": -1, "conv": cl, "unknown": unknown, "turn_meta": meta, "frames": cs.T,
                "error": str(e)}


def load_engine(a):
    import torch
    import infer_ppm as IP
    assert os.environ.get("GEMM_PRECISION", "bf16") == "bf16", "bf16 only"
    devs = [torch.device(f"cuda:{int(x)}") for x in a.devs.split(",")]
    torch.cuda.set_device(devs[-1])
    ci = IP.ckpt_info(a.ckpt)
    a.seed = 0
    m, ldev, cfgs = IP.build_model(ci, devs)
    g = IP.PPMGen(m, ldev, cfgs, devs, CTX + 200, a)
    return IP, g, ci


def cmd_worker(a):
    import torch
    seeds = parse_seeds(a.seeds)
    gd, cd = f"{a.out}/games", f"{a.out}/claims-{a.launch or 'x'}"
    os.makedirs(gd, exist_ok=True); os.makedirs(cd, exist_ok=True)
    todo = [s for s in seeds if not os.path.exists(f"{gd}/{s}.json") and not os.path.exists(f"{cd}/{s}")]
    if not todo:
        print("[gb-ppm] nothing to play", flush=True); return
    S = load_sim()
    with torch.inference_mode():
        IP, g, ci = load_engine(a)
        IP.log(f"[gb-ppm] {socket.gethostname()} devs {a.devs}: {ci['tag']} step {ci['step']} loaded; seeds {a.seeds}")
        for s in todo:
            if os.path.exists(f"{gd}/{s}.json"):
                continue
            try:
                os.mkdir(f"{cd}/{s}")                                     # atomic claim
            except FileExistsError:
                continue
            t0 = time.time()
            r = play(IP, g, S, s, a)
            r.update({"wall_s": round(time.time() - t0, 1), "host": socket.gethostname(), "ckpt": ci["path"], "step": ci["step"],
                      "tag": ci["tag"], "oracle": S.ORACLE_VERSION, "temp": a.temp, "top_k": a.topk, "max_turn_s": a.max_turn_s, "ctx": CTX})
            tmp = f"{gd}/.{s}.json.tmp"
            json.dump(r, open(tmp, "w"), indent=1)
            os.replace(tmp, f"{gd}/{s}.json")
            IP.log(f"[gb-ppm] seed {s}: {'SUCCESS' if r['success'] else 'fail'} after {r['questions']} questions; {r['frames']} frames; "
                   f"{r['wall_s']:.0f}s" + (f"; {r['error']}" if r.get("error") else ""))


def cmd_merge(a):
    import statistics
    fs = sorted(glob.glob(f"{a.out}/games/*.json"), key=lambda p: int(os.path.basename(p).split(".")[0]))
    want = parse_seeds(a.seeds) if a.seeds else None
    res = [json.load(open(f)) for f in fs]
    if want:
        res = [r for r in res if r["seed"] in want]
        missing = sorted(set(want) - {r["seed"] for r in res})
    else:
        missing = []
    s = sum(r["success"] for r in res)
    qs = [r["questions"] for r in res if r["success"]]
    med = statistics.median(qs) if qs else None
    noeob = sum(1 for r in res for t in r.get("turn_meta", []) if not t["ended_with_im_end"])
    nturn = sum(len(r.get("turn_meta", [])) for r in res)
    label = a.label or (res[0]["tag"] if res else os.path.basename(a.out))
    out = {"label": label, "oracle": "v5", "successes": s, "n": len(res), "median_questions_success": med, "missing_seeds": missing,
           "context_full": sum(1 for r in res if r.get("error")), "turns_without_im_end": noeob, "turns": nturn,
           "unknown_questions": sum(len(r["unknown"]) for r in res), "attempts": res}
    rng = f"_s{a.seeds}" if a.seeds else ""
    json.dump(out, open(f"{a.out}/gb{rng}.json", "w"), indent=1)
    lines = [f"{label}: {s}/{len(res)} successes (oracle v5), median questions to success {med}; context-full {out['context_full']}; "
             f"turns without <|im_end|> {noeob}/{nturn}; unknown questions {out['unknown_questions']}" + (f"; MISSING seeds {missing}" if missing else "")]
    for r in res:
        lines.append(f"  seed {r['seed']}: {'SUCCESS' if r['success'] else 'fail   '} q {r['questions']:3d}  frames {r['frames']:5d}"
                     + (f"  {r['error']}" if r.get("error") else ""))
    open(f"{a.out}/summary{rng}.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_parity(a):
    """ChatSession must reproduce infer_ppm.run_chat bitwise (same lanes) on the same user turns and seed."""
    import torch
    S = load_sim()
    os.makedirs(a.out, exist_ok=True)
    users = list(S.OPEN) + ["Yes.", "No.", "I'm not sure."]
    with torch.inference_mode():
        IP, g, ci = load_engine(a)
        a.user, a.system, a.chat_file, a.interactive = users, "", "", False
        a.text_temp, a.text_topk = a.temp, a.topk
        a.seed = 123
        g.gen.manual_seed(123)
        IP.run_chat(a, g, a.out)
        ref = json.load(open(f"{a.out}/text.json"))
        cs = ChatSession(IP, g, 123, a.temp, a.topk, a.max_turn_s)
        texts = [cs.say(u) for u in users]
        same = cs.ids[0] == ref["lane_A"] and cs.ids[1] == ref["lane_B"]
        IP.log(f"[gb-ppm parity] run_chat vs ChatSession lanes identical: {same} ({len(ref['lane_A'])} vs {len(cs.ids[0])} frames)")
        for u, t in zip(users, texts):
            print(f"USER: {u}\nMODEL: {t['text']}   [frames {t['frames']}, im_end {t['ended_with_im_end']}]", flush=True)
        json.dump({"identical": same, "turns": texts}, open(f"{a.out}/parity.json", "w"), indent=1)
        assert same, "ChatSession diverges from run_chat"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("worker", "merge", "parity"))
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--launch", default="", help="worker: launch id (claims are per launch, so a rerun replays games a dead worker left)")
    ap.add_argument("--devs", default="0,1")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--max-turn-s", type=float, default=120.0)
    a = ap.parse_args()
    {"worker": cmd_worker, "merge": cmd_merge, "parity": cmd_parity}[a.cmd](a)


if __name__ == "__main__":
    main()
