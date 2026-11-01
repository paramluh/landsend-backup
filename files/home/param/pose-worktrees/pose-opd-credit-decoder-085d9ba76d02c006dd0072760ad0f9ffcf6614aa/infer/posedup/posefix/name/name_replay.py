#!/usr/bin/env python3
"""posefix NAME-A (10-27, user: "why is the name lauren at 90% chance"): offline replay of the speech twin's Kate call 20261027-003457
(vox17127 export-249, live_ppm_twin engine, Kate 06 voice clip = SAM opening mimi_local[:, :125], conversation-start prefill 26 frames,
A = the call's recorded mic after the server's level stage). Reference run: B's lane TEACHER-FORCED to the call's lane (session tok), B's
Mimi codes = the engine's own draws (recorded); counterfactual runs force BOTH the reference lane and codes, so the history up to the name
is identical and only the listed input changes. At the name frame: B's text-lane distribution of the cond row (eng.zt row 0): raw softmax
over the whole lane, and the word-piece-only renormalised view (the factorised sampler's word stage before temperature / top-k / min-p).
Inputs pinned under posefix/name/pinroot (Kate's live package is being rebuilt). usage: name_replay.py <out json> [cond ...]"""
import hashlib
import json
import os
import sys

TW = "/data/home/amartyavarma/ppm/aloe-m1vox-live"
sys.path.insert(0, TW)
os.environ.setdefault("PPMLIVE_DEFAULTS", "nopersona")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import live_ppm_twin as LP  # noqa: E402
import prompts_ppm as PR  # noqa: E402
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-posedup-live")
import prefill_clip as PC  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

# read-only instrumentation: the engine's own factorised draw (t_text_draw) re-computed WITHOUT any RNG call to record the distribution the
# word piece is actually drawn from (penalties -> temperature -> word min-p -> top-k -> top-p) and the WORD-category probability
_TTD = LP.t_text_draw
DRAW = {}


def _rec_ttd(z, is_word, cnt, cnt2, fact, ct, tt, tk, tp, mp, mrel, fp, rp, wg, gen_t, gen_c):
    w = is_word[None]
    zw = z.masked_fill(~w, LP.NINF)
    useW = (fact > 0) & torch.isfinite(zw).any(-1, keepdim=True)
    zp = LP.t_penalties(torch.where(useW, zw, z), cnt, fp, rp, is_word, cnt2)
    zt_ = zp / tt.clamp_min(1e-4)
    DRAW["ps"] = torch.softmax(LP._topp_mask(LP._topk_mask(LP.t_word_min_p(zt_, is_word, mp, mrel), tk), tp), -1).clone()
    DRAW["ps_rel"] = torch.softmax(LP._topp_mask(LP._topk_mask(LP.t_word_min_p(zt_, is_word, mp, torch.ones_like(mrel)), tk), tp), -1).clone()
    DRAW["ps_nominp"] = torch.softmax(LP._topp_mask(LP._topk_mask(zt_, tk), tp), -1).clone()
    DRAW["zp"] = zp.clone()
    DRAW["in"] = dict(z2=torch.where(useW, zw, z).clone(), cnt=cnt.clone(), cnt2=None if cnt2 is None else cnt2.clone(), tt=tt.clone(), tk=tk.clone(),
                      tp=tp.clone(), mp=mp.clone(), mrel=mrel.clone(), fp=fp.clone(), rp=rp.clone(), ct=ct.clone(), z=z.clone())
    zc = torch.cat([torch.logsumexp(z[:, :LP.TE], -1, keepdim=True), z[:, LP.TE:]], -1) / ct.clamp_min(1e-4)
    DRAW["pc_word"] = torch.softmax(zc, -1)[:, 0].clone()
    return _TTD(z, is_word, cnt, cnt2, fact, ct, tt, tk, tp, mp, mrel, fp, rp, wg, gen_t, gen_c)


LP.t_text_draw = _rec_ttd
OUT = sys.argv[1]
CONDS = sys.argv[2:] or ["ref", "noclip", "clip09", "prefill_off"]
CK = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/vox/export-vox17127-249"
SESS = f"{TW}/twin/sessions/20261027-003457"
ROOT = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/name/pinroot"
KATE = "06_74807fe0"
SPF, CD, LOOK = LP.SPF, LP.CD, LP.VS.LOOK24
SHA = {"clip": "aadfaa2e3a46d02ac4e02ae1931d7be401168c4f1554169597f12cf11055a5fc", "codes": "3c734ab278a7569d20d07133dc8214645932de3de30c174ff0b2ff2a0b31539f",
       "lane": "b521c6539b50f4d1ae286b8be3e37796424cce8bd541f9abba51eaee90dfaf58"}


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes()).hexdigest()


def prefix_with(codes):
    if codes is None:
        return PR.build_prefix("", None, prose=False)[0]
    if isinstance(codes, dict):                                   # a clip dict WITH its word lane (kate_intro_words)
        return PR.build_prefix("", codes, prose=False)[0]
    c = PR.upload_clip(np.asarray(codes, np.int64))
    c.update(src="conversation_start")
    return PR.build_prefix("", c, prose=False)[0]


def main():
    sj = json.load(open(f"{SESS}/session.json"))
    z = np.load(f"{SESS}/session.npz")
    A, lane_rec = z["a"].astype(np.float32), z["tok"].astype(np.int64)
    params = {k: v for k, v in sj["params"].items() if k in LP.PARAMS}
    seed = int(sj["seed"])
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    with torch.no_grad():
        m, ldev, cfgs, info = LP.build_model(CK, devs)
        eng = LP.LivePPM(m, ldev, cfgs, devs, max_s=60.0)
        tok = LP.IP.tok()
        kclip, kci = PC.clip_for(KATE, "start", ROOT)
        pf_codes, pf_lane, _, pfi = PC.prefill_from_clip(KATE, 2.0, ROOT, False, LP, tok, clip_src="start")
        checks = {"clip": sha(kclip) == SHA["clip"], "codes": sha(pf_codes.T if pf_codes.shape[0] != 32 else pf_codes) == SHA["codes"]
                  or sha(pf_codes) == SHA["codes"], "lane": sha(pf_lane) == SHA["lane"]}
        print("[name] input sha checks", checks, "prefill", pfi.get("frames"), flush=True)
        names = {n: tok.encode(" " + n, add_special_tokens=False) for n in ("Lauren", "Rosie", "Kate", "Katie", "Kat", "Laura", "Lauryn")}
        print("[name] token ids", names, flush=True)
        L0 = names["Lauren"][0]
        hits = np.flatnonzero(lane_rec == L0)
        assert len(hits), ("no Lauren token in the recorded lane", names["Lauren"])
        tn = int(hits[0])                                       # conversation frame where B's lane emits " Lauren" (first piece)
        print("[name] Lauren at frame", tn, "lane context:", tok.decode([t for t in lane_rec[max(0, tn - 60):tn] if t < LP.TE]), flush=True)
        is_word = (torch.arange(eng.zt.shape[-1], device=eng.dl) < LP.TE)
        pf = {"lane": pf_lane, "codes": np.ascontiguousarray(pf_codes.T) if pf_codes.shape[0] == 32 else pf_codes}

        def run(clip_codes, prefill, a_wave, force_codes=None, lane=None):
            lane = lane_rec if lane is None else lane
            pre = prefix_with(clip_codes)
            P = int(pre["P"])
            eng.start(pre, seed, dict(params), prefill=prefill)
            eng.set_params(dict(params))
            wrote, codes_rec = 0, []
            for f in range(tn + 1):
                T = P + f
                need = (f + 1) * SPF + LOOK
                eng.write_a(wrote, a_wave[wrote:need]); wrote = need
                if force_codes is not None:
                    cf = force_codes[f]
                elif prefill is not None and CD <= f < CD + len(prefill["codes"]):
                    cf = prefill["codes"][f - CD] + LP.OFF                 # step(force=...) skips the engine's own prefill-code forcing
                else:
                    cf = None
                dec = eng.step(T, graphs=False, force=(int(lane[f]), cf))
                eng.result(T, dec)
                codes_rec.append(eng.depB[T].clone().cpu().numpy() if f >= CD else None)
            z_ = eng.zt[0].float()
            pr = torch.softmax(z_, -1)
            zw = z_.masked_fill(~is_word, float("-inf"))
            pw = torch.softmax(zw, -1)
            top = lambda p: [(tok.decode([int(i)]), round(float(p[i]), 4)) for i in torch.topk(p, 10).indices]  # noqa: E731
            res = {"P": P, "top10_raw": top(pr), "top10_word_stage": top(pw),
                   **{f"p_raw_{n}": round(float(pr[ids[0]]), 5) for n, ids in names.items()},
                   **{f"p_word_{n}": round(float(pw[ids[0]]), 5) for n, ids in names.items()},
                   "p_word_category": round(float(pr[is_word].sum()), 4)}
            if "ps" in DRAW:                                       # the engine's actual draw at the name frame (row 0)
                ps, psr, psn = DRAW["ps"][0], DRAW["ps_rel"][0], DRAW["ps_nominp"][0]
                pcw = float(DRAW["pc_word"][0])
                res.update(draw_word_category=round(pcw, 4), draw_n_words_kept_absminp=int((ps[:LP.TE] > 0).sum()),
                           draw_top5_absminp=top(ps)[:5], draw_top5_penalised_no_minp=top(psn)[:5],
                           **{f"draw_{n}_absminp": round(float(ps[ids[0]]), 5) for n, ids in names.items() if n in ("Lauren", "Rosie", "Kate")},
                           **{f"draw_{n}_relminp": round(float(psr[ids[0]]), 5) for n, ids in names.items() if n in ("Lauren", "Rosie", "Kate")},
                           **{f"draw_{n}_nominp": round(float(psn[ids[0]]), 5) for n, ids in names.items() if n in ("Lauren", "Rosie", "Kate")})
            return res, codes_rec

        out = {"session": SESS, "name_frame": tn, "names": names, "checks": checks, "conds": {}}
        ref, ref_codes = run(kclip, pf, A)
        if CONDS == ["minp"]:
            I = DRAW["in"]
            isw = is_word
            gen = torch.Generator(device=eng.dl); gen.manual_seed(12345)

            def dist(mp, mrel, fp):
                zp = LP.t_penalties(I["z2"], I["cnt"], fp, I["rp"], isw, I["cnt2"]) / I["tt"].clamp_min(1e-4)
                zs = zp if mp is None else LP.t_word_min_p(zp, isw, mp, mrel)
                return torch.softmax(LP._topp_mask(LP._topk_mask(zs, I["tk"]), I["tp"]), -1)[0]
            zc = torch.cat([torch.logsumexp(I["z"][:, :LP.TE], -1, keepdim=True), I["z"][:, LP.TE:]], -1) / I["ct"].clamp_min(1e-4)
            pc = torch.softmax(zc, -1)[0]
            one = torch.ones_like(I["mrel"]); zero_fp = torch.zeros_like(I["fp"])
            V = {"call (abs min-p 0.05, freq_pen 0.25)": dist(I["mp"], I["mrel"], I["fp"]),
                 "relative min-p 0.05": dist(I["mp"], one, I["fp"]),
                 "min-p off": dist(None, None, I["fp"]),
                 "freq_pen 0 (abs min-p as call)": dist(I["mp"], I["mrel"], zero_fp)}
            mm = {"call_knobs": {k: float(I[k].flatten()[0]) for k in ("mp", "mrel", "fp", "rp", "tt", "tk", "tp", "ct")},
                  "P_WORD_category": round(float(pc[0]), 4)}
            for nm, ps in V.items():
                keep = torch.nonzero(ps[:LP.TE] > 0).flatten()
                order = keep[torch.argsort(ps[keep], descending=True)]
                surv = [(tok.decode([int(i)]), round(float(ps[i]), 4)) for i in order[:12]]
                dr = torch.multinomial(ps[None].expand(20, -1), 1, replacement=True, generator=gen)[:, 0]
                hist = {}
                for t in dr.tolist():
                    k = tok.decode([t]); hist[k] = hist.get(k, 0) + 1
                mm[nm] = {"n_surviving": int(len(keep)), "top12": surv, "draws20": dict(sorted(hist.items(), key=lambda x: -x[1]))}
                print(f"[minp] {nm}: n={len(keep)} top={surv[:6]} draws={mm[nm]['draws20']}", flush=True)
            out["minp"] = mm
            json.dump(out, open(OUT, "w"), indent=1)
            print("[minp] P(WORD category)", mm["P_WORD_category"], "knobs", mm["call_knobs"], flush=True)
            return
        out["conds"]["ref_selfcodes"] = ref
        print("[name] ref (lane forced, own codes):", json.dumps(ref), flush=True)
        fc = [None if c is None else c for c in ref_codes]
        clips = {"ref": kclip}
        for pid in ("09_d9ef3c94", "12_72cb45d3", "19_47473a46"):
            if os.path.isfile(f"{ROOT}/{pid}/voice_clip.safetensors"):
                clips[pid] = PC.clip_for(pid, "start", ROOT)[0]
        kv = np.asarray(load_file(f"{ROOT}/{KATE}/voice_clip.safetensors")["codes"], np.int64)
        table = {"ref": (kclip, pf, A), "noclip": (None, pf, A), "clip09": (clips.get("09_d9ef3c94"), pf, A),
                 "clip12": (clips.get("12_72cb45d3"), pf, A), "clip19": (clips.get("19_47473a46"), pf, A),
                 "prefill_off": (kclip, None, A), "a_silent": (kclip, pf, np.zeros_like(A)), "clip_rule": (kv, pf, A)}
        # prefill OFF with B silent at the start: the first CD + N lane frames PAD, B codes = digital-silence codes (pose_cfg_silence.npy)
        KI = "/data/home/amartyavarma/ppm/sam-sep-handoff/test/06_kate_fdba6c86"
        ki = np.asarray(load_file(f"{KI}/voice_clip_kate_intro.safetensors")["codes"], np.int64)
        kw = json.load(open(f"{KI}/voice_clip_kate_intro_words.json"))
        wl, pid, po = [], [], [0]
        for w in kw:                                                # [start, end, text]: the clip's own timed words (training row format)
            st, txt = float(w[0]), str(w[2])
            ids = tok.encode(" " + txt, add_special_tokens=False)
            wl.append({"start": st, "end": float(w[1]), "start_frame": int(st * 12.5), "w": txt}); pid += ids; po.append(len(pid))
        ki_words = {"w0": 0, "L": int(ki.shape[1]), "src_side": "assistant", "side": "assistant", "blank_lane": False,
                    "src_row": {"codes_assistant": ki, "words_assistant": wl, "piece_ids_assistant": np.asarray(pid, np.int64),
                                "piece_offsets_assistant": np.asarray(po, np.int64)}}
        table["kate_intro"] = (ki, pf, A)
        table["kate_intro_words"] = (ki_words, pf, A)
        Npf = len(pf["codes"])
        sil = np.load("/data/home/amartyavarma/ppm/aloe-posedup-live/pose_cfg_silence.npy")[:, :Npf].T + LP.OFF
        lane_sil = lane_rec.copy(); lane_sil[:Npf + CD] = LP.PAD
        fc_sil = list(fc)
        for f in range(CD, CD + Npf):
            fc_sil[f] = sil[f - CD]
        for c in CONDS:
            if c == "prefill_off_silentB":
                r, _ = run(kclip, None, A, force_codes=fc_sil, lane=lane_sil)
                out["conds"][c] = r
                print(f"[name] {c}:", json.dumps({k: v for k, v in r.items() if k.startswith(("p_", "top10_word"))}), flush=True)
                continue
            if c not in table:
                continue
            cc, pp, aa = table[c]
            r, _ = run(cc, pp, aa, force_codes=fc)
            out["conds"][c] = r
            print(f"[name] {c}:", json.dumps({k: v for k, v in r.items() if k.startswith(("p_", "top10_word"))}), flush=True)
            json.dump(out, open(OUT, "w"), indent=1)
        json.dump(out, open(OUT, "w"), indent=1)
        print("[name] DONE", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
