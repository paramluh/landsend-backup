"""step-0 checks 1 + 2 on the FULL-SIZE PPM (64-layer Qwen3-32B backbone, 2 x 36-layer 4B towers, 2 x 28-layer 0.6B decoders), FSDP2 over
one node (torchrun --nproc_per_node 8). No optimizer.
  1. backbone == HF Qwen3-32B: the plain token sequence (step0_tokens) as a text-only row (channel B = the tokens, channel A absent,
     plain RoPE pid 0..N-1) -> raw lm_head logits of P_B(final norm(h)) (P_B = identity at init) vs hf_ref.py's stock HF logits.
  2. init losses on real micros of this node (pose eval row, FULL podcast eval row with its system prompt, text-duplex eval pack 0):
     speech / pose cb0 CE (4B towers) and decoder CE (0.6B, 31 codebooks) vs ln 2048 = 7.6246; text CE (word, normalised word, PAD, NW)
     of the duplex rendering vs base Qwen's plain-text next-token CE on the SAME words (channel B's pieces, concatenated).
usage: torchrun ... tests/step0_full.py HF_REF_FILE"""
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PPM_RUN", "ppm-step0-full")
import train_ppm as T  # noqa: E402
import step0_tokens as S0  # noqa: E402

M, D = T.M, T.D
REF = sys.argv[1]


def main():
    dev = torch.device("cuda", T.LOCAL)
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    t0 = time.time()
    c32, c4, c06 = M.cfg32(), M.cfg4b(), M.cfg06()
    model, info = M.build(c32, c4, c06, dev, T.log)
    model.eval()
    rot = M.Rot(c32, dev)
    T.log(f"built full PPM in {time.time() - t0:.0f}s; peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")
    res = {"init_info": {k: v for k, v in info.items()}}
    # ---- check 1 (vs stock HF bf16 AND vs HF fp32 = the precision reference: is PPM's bf16 error the same size as HF-bf16's?)
    ref = torch.load(REF)
    ref32 = torch.load(REF.replace(".pt", "_fp32.pt")) if os.path.isfile(REF.replace(".pt", "_fp32.pt")) else None
    ids = ref["ids"].to(dev)
    N = ids.numel()
    b = {"kind": "text", "ids": torch.stack([torch.full_like(ids, M.PAD), ids]), "present": torch.stack([torch.zeros(N, device=dev), torch.ones(N, device=dev)]),
         "pid": torch.arange(N, device=dev)[None], "segs": [N], "sel": torch.arange(N, device=dev)}
    with torch.no_grad():
        lg = model(b, rot, rot, want_logits=True)["logits"]

    def cmp(x, y, tag):
        d = (x - y).abs()
        lp, lq = torch.log_softmax(y, -1), torch.log_softmax(x, -1)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        pm = d.max(-1).values
        i = int(pm.argmax()); j = int(d[i].argmax())
        r = {"max_abs": d.max().item(), "mean_abs": d.mean().item(), "max_abs_excl_pos0": d[1:].max().item(), "p99_abs": torch.quantile(d[::8].flatten()[::97], 0.99).item(),
             "argmax_pos": i, "argmax_col": j, "vals_at_max": [x[i, j].item(), y[i, j].item()], "top1_agree": (x.argmax(-1) == y.argmax(-1)).float().mean().item(),
             "kl_mean": kl.mean().item(), "kl_max": kl.max().item(), "ce_x": torch.nn.functional.cross_entropy(x[:-1], ids[1:]).item(),
             "ce_y": torch.nn.functional.cross_entropy(y[:-1], ids[1:]).item()}
        T.log(f"[check1] {tag}: {json.dumps(r)}")
        return r
    hf = ref["logits"].to(dev).float()
    res["check1"] = {"tokens": N, "ppm_vs_hf_bf16": cmp(lg, hf, "PPM vs HF bf16")}
    if ref32 is not None:
        h32 = ref32["logits"].to(dev).float()
        res["check1"]["ppm_vs_hf_fp32"] = cmp(lg, h32, "PPM vs HF fp32")
        res["check1"]["hf_bf16_vs_hf_fp32"] = cmp(hf, h32, "HF bf16 vs HF fp32")
        del h32
    del hf, lg
    # plain-text CE on the whole B word sequence of eval pack 0 (base Qwen on the same words as the duplex micro below)
    pt = torch.as_tensor(S0.plain_tokens(10 ** 9)).to(dev)
    Np = pt.numel()
    bp = {"kind": "text", "ids": torch.stack([torch.full_like(pt, M.PAD), pt]), "present": torch.stack([torch.zeros(Np, device=dev), torch.ones(Np, device=dev)]),
          "pid": torch.arange(Np, device=dev)[None], "segs": [Np], "sel": torch.arange(Np - 1, device=dev)}
    with torch.no_grad():
        lgp = model(bp, rot, rot, want_logits=True)["logits"]
        cep = torch.nn.functional.cross_entropy(lgp, pt[1:], reduction="none")
        cen = torch.logsumexp(lgp[:, :151643], -1) - lgp.gather(1, pt[1:, None])[:, 0]
    res["plain_text"] = {"tokens": Np, "ce": cep.mean().item(), "ce_norm_word_vocab": cen.mean().item()}
    del lgp
    # ---- check 2 (real micros of this node)
    dcfg = T.CFG["data"]
    pnode = D.PS.load_node(T.CFG["pose_root"])
    pplan = D.PS.plan(pnode, T.CFG["pose_seed"], T.LW, T.LOCAL, 1, T.LW, T.LOCAL, 1)
    snode = D.R3.load_node(T.CFG["speech_root"])
    splan = D.speech_plan(snode, D.DS.load_index(T.CFG["speech_root"]), T.CFG["speech_seed"], T.LW, T.LOCAL, T.CFG["speech_eval_frac"], T.CFG["speech_eval_salt"], 1)
    recs, _ = S0.pack0(dcfg)
    micros = {"pose": D.PoseDS(pnode, pplan["eval_rows"], dcfg, T.RANK)[0], "speech": D.SpeechDS(T.CFG["speech_root"], splan["eval_rows"], dcfg, T.RANK)[0],
              "text": D.text_micro(recs, dcfg)}
    dos = {"dec_s": None, "dec_p": None}
    info0 = {}
    for kind, mb in micros.items():
        v, _ = T.run_micro(model, T.prep(mb, dev), rot, dos, grad=False, info=info0 if kind == "speech" else None)
        s = T.summarize(kind, T.allsum(v, dev), "init")
        res[f"check2_{kind}"] = {**s, "P": json.loads(mb["prov"]).get("P") if kind == "speech" else None, "N": int(len(mb["pid"]))}
        T.log(f"[check2] {kind}: " + " ".join(f"{k} {x:.4f}" for k, x in s.items()))
    res["tower_input_rms"] = info0
    res["ln2048"] = math.log(2048)
    T.log(f"[check2] tower input rms (speech row, rank 0): {info0}; ln 2048 = {math.log(2048):.4f}; plain-text CE on the same words {res['plain_text']}")
    if T.RANK == 0:
        os.makedirs("/data/home/amartyavarma/ppm/runs/step0", exist_ok=True)
        json.dump(res, open("/data/home/amartyavarma/ppm/runs/step0/step0_full.json", "w"), indent=1, default=str)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
