"""EXACT-DTYPE 0.6B DEPTH DECODER kernels for the PPM live / playground engines (posefix 10-26, main c7's order "match the trainer exactly").
Kernels = motlive's ppm/mot/live/light_exact.py (copied 10-26 06:06 UTC), the attention kernel extended to S independent rows (pose CFG
[cond | uncond], fast_ppm's NI instances).

WHY  the engines' FusedDec ran the decoder with bf16-module semantics (bf16 residual stream, RMSNorm rounded to bf16 before the weight
     multiply, bf16 q / k norm weights, bf16 cos / sin, bf16 slot 0 / embeddings). The trainer runs the replicated decoders (dec_s, dec_p)
     as fp32-master modules under torch.autocast(bf16) (model_ppm.Decoder.forward -> HF Qwen3Model, sdpa), so its dtype flow is:
  inputs     x = cat(proj(hv) [autocast Linear: bf16] .to(fp32) (+ zb.to(fp32), D33 pose decoder), cb0_emb(c0) [fp32], embed_tokens [fp32])
  RMSNorm    fp32 in: w32 * (x * rsqrt(mean(x^2) + eps)) -> fp32; consumed by Linears only (autocast: ONE bf16 rounding)
  q / k      q_proj / k_proj (bf16) -> q_norm / k_norm: w32 * bf16(x * rsqrt(..)) -> fp32 -> RoPE in fp32 (fp32 cos / sin) -> SDPA bf16
  attention  SDPA (flash: fp32 scores / softmax, P rounded to bf16 for P.V, row sum from fp32 P) -> bf16 -> o_proj (bf16)
  residual   fp32 + bf16 -> fp32 (the residual stream stays fp32)
  MLP        gate / up (bf16) -> silu (bf16) -> * (bf16) -> down (bf16)
  final      model.norm (fp32) -> heads einsum under autocast: bf16(hl) @ W -> bf16 logits -> .float()
Parameters: the export's bf16 values (= the trainer's masters rounded to bf16; the masters themselves are not on disk). Every GEMM weight
is exactly the trainer's autocast cast of its master; the fp32 tensors (RMSNorm weights, embedding rows) are exact fp32 upcasts of the
bf16 export. Remaining non-maths differences: fp32 reduction order, Triton exp / rsqrt vs CUDA (<= ~2 ulp fp32 before the bf16 roundings)."""
import os

import torch
import triton
import triton.language as tl

BF = torch.bfloat16
F32 = torch.float32
ROUND_P = int(os.environ.get("LIGHT_EXACT_ROUND_P", "1"))        # 1 = flash-SDPA semantics (P -> bf16 for P.V)


@triton.jit
def _add_rms_f32_k(R, A, S, Y, W, N, eps, HAS_ADD: tl.constexpr, BLOCK: tl.constexpr):
    """s = r (fp32) [+ a (bf16 -> fp32)] in fp32; y = bf16(w32 * (s * rsqrt(mean(s^2) + eps))) (= HF RMSNorm on an fp32 stream, then the
    autocast cast of the Linear input)."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(R + row * N + offs, mask=m, other=0.0).to(tl.float32)
    if HAS_ADD:
        a = tl.load(A + row * N + offs, mask=m, other=0.0).to(tl.float32)
        x = x + a
        tl.store(S + row * N + offs, x, mask=m)
    var = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(W + offs, mask=m, other=0.0).to(tl.float32)
    xn = x * rstd
    tl.store(Y + row * N + offs, (w * xn).to(tl.bfloat16), mask=m)


def add_rms_f32(res, a, w32, eps):
    """res fp32 [L, N]; a bf16 [L, N] or None -> (new residual fp32 [L, N] (res itself when a is None), normed bf16 [L, N])."""
    L, N = res.shape
    y = torch.empty(L, N, dtype=BF, device=res.device)
    with torch.cuda.device(res.device):                            # Triton launches on the CURRENT device: pin it to the tensors'
        if a is None:
            _add_rms_f32_k[(L,)](res, res, res, y, w32, N, float(eps), HAS_ADD=False, BLOCK=triton.next_power_of_2(N), num_warps=4)
            return res, y
        s = torch.empty(L, N, dtype=F32, device=res.device)
        _add_rms_f32_k[(L,)](res, a.reshape(L, N), s, y, w32, N, float(eps), HAS_ADD=True, BLOCK=triton.next_power_of_2(N), num_warps=4)
    return s, y


def final_norm_f32(res, a, w32, eps):
    """the decoder's last add + model.norm -> bf16 (the heads einsum's autocast cast of the fp32 norm output)."""
    return add_rms_f32(res, a, w32, eps)[1]


@triton.jit
def _qk_exact(base, WN1, WN2, o1, HALF: tl.constexpr, D: tl.constexpr, eps, c1, c2, s1, s2):
    """HF q_norm / k_norm on a bf16 projection row (w32 * bf16(x * rsqrt(mean(x^2) + eps)), fp32), then RoPE in fp32 -> (e1, e2) fp32."""
    x1 = tl.load(base + o1).to(tl.float32)
    x2 = tl.load(base + HALF + o1).to(tl.float32)
    rstd = tl.rsqrt((tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / D + eps)
    k1 = WN1 * (x1 * rstd).to(tl.bfloat16).to(tl.float32)
    k2 = WN2 * (x2 * rstd).to(tl.bfloat16).to(tl.float32)
    e1 = k1 * c1 - k2 * s1                                         # (k * cos) + (rotate_half(k) * sin), first half: -k2 * sin
    e2 = k2 * c2 + k1 * s2
    return e1, e2


@triton.jit
def _attn_exact_k(QKV, QN, KN, COS, SIN, KC, VC, O, eps, scale, t0, RS,
                  L: tl.constexpr, NH: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr, BLK: tl.constexpr,
                  RP: tl.constexpr):
    """grid (kv head, q head in the GQA group, row): row r's L new positions t0 .. t0 + L - 1 (qkv rows r L .. r L + L - 1) -> its cache
    (row stride RS elements) -> causal attention over positions 0 .. t0 + l (= motlive's single-row kernel per row)."""
    h = tl.program_id(0)
    gq = tl.program_id(1)
    r = tl.program_id(2).to(tl.int64)
    G: tl.constexpr = NH // KVH
    W: tl.constexpr = (NH + 2 * KVH) * D
    QKV += r * L * W
    KC += r * RS
    VC += r * RS
    O += r * L * (NH * D)
    o1 = tl.arange(0, HALF)
    kn1 = tl.load(KN + o1)
    kn2 = tl.load(KN + HALF + o1)
    qn1 = tl.load(QN + o1)
    qn2 = tl.load(QN + HALF + o1)
    for l in tl.static_range(L):                                   # the chunk's keys / values -> cache (bf16 = SDPA's autocast cast)
        pos = t0 + l
        base = QKV + l * W
        c1 = tl.load(COS + l * D + o1)
        c2 = tl.load(COS + l * D + HALF + o1)
        s1 = tl.load(SIN + l * D + o1)
        s2 = tl.load(SIN + l * D + HALF + o1)
        e1, e2 = _qk_exact(base + NH * D + h * D, kn1, kn2, o1, HALF, D, eps, c1, c2, s1, s2)
        cb = (pos * KVH + h) * D
        tl.store(KC + cb + o1, e1.to(tl.bfloat16))
        tl.store(KC + cb + HALF + o1, e2.to(tl.bfloat16))
        tl.store(VC + cb + o1, tl.load(base + (NH + KVH) * D + h * D + o1))
        tl.store(VC + cb + HALF + o1, tl.load(base + (NH + KVH) * D + h * D + HALF + o1))
    tl.debug_barrier()
    j = tl.arange(0, BLK)
    kb = (j * KVH + h) * D
    qh = h * G + gq
    for l in tl.static_range(L):
        pos = t0 + l
        jm = j <= pos
        K1 = tl.load(KC + kb[:, None] + o1[None, :], mask=jm[:, None], other=0.0).to(tl.float32)
        K2 = tl.load(KC + kb[:, None] + HALF + o1[None, :], mask=jm[:, None], other=0.0).to(tl.float32)
        V1 = tl.load(VC + kb[:, None] + o1[None, :], mask=jm[:, None], other=0.0).to(tl.float32)
        V2 = tl.load(VC + kb[:, None] + HALF + o1[None, :], mask=jm[:, None], other=0.0).to(tl.float32)
        c1 = tl.load(COS + l * D + o1)
        c2 = tl.load(COS + l * D + HALF + o1)
        s1 = tl.load(SIN + l * D + o1)
        s2 = tl.load(SIN + l * D + HALF + o1)
        e1, e2 = _qk_exact(QKV + l * W + qh * D, qn1, qn2, o1, HALF, D, eps, c1, c2, s1, s2)
        r1 = e1.to(tl.bfloat16).to(tl.float32)
        r2 = e2.to(tl.bfloat16).to(tl.float32)
        sc = (tl.sum(K1 * r1[None, :], axis=1) + tl.sum(K2 * r2[None, :], axis=1)) * scale
        sc = tl.where(jm, sc, float("-inf"))
        p = tl.exp(sc - tl.max(sc, axis=0))
        den = tl.sum(p, axis=0)
        if RP:
            p = p.to(tl.bfloat16).to(tl.float32)
        ob = O + l * (NH * D) + qh * D
        tl.store(ob + o1, (tl.sum(V1 * p[:, None], axis=0) / den).to(tl.bfloat16))
        tl.store(ob + HALF + o1, (tl.sum(V2 * p[:, None], axis=0) / den).to(tl.bfloat16))


def attn_exact(qkv, qn32, kn32, eps, cos32, sin32, KC, VC, t0, L, nh, kvh, D, scale, round_p=None):
    """qkv [S L, (nh + 2 kvh) D] bf16 (row-major: row r's L positions together); cos32 / sin32 [L, D] fp32 (positions t0 .. t0 + L - 1,
    the same for every row); KC / VC [S, MP, kvh, D] bf16 -> [S L, nh D] bf16."""
    S, MP = KC.shape[0], KC.shape[1]
    assert qkv.shape[0] == S * L and KC.is_contiguous() and VC.is_contiguous(), (qkv.shape, KC.shape)
    o = torch.empty(S * L, nh * D, dtype=BF, device=qkv.device)
    with torch.cuda.device(qkv.device):
        _attn_exact_k[(kvh, nh // kvh, S)](qkv, qn32, kn32, cos32, sin32, KC, VC, o, float(eps), float(scale), int(t0), int(KC.stride(0)),
                                            L=L, NH=nh, KVH=kvh, D=D, HALF=D // 2, BLK=triton.next_power_of_2(MP),
                                            RP=int(ROUND_P if round_p is None else round_p), num_warps=4)
    return o


def layer_weights(dec):
    """the decoder's per-layer tensors in the trainer's dtypes: fused bf16 GEMM weights (qkv, gate|up; = the autocast casts), fp32 RMSNorm
    weights (exact upcasts of the bf16 export)."""
    out = []
    for L in dec.model.layers:
        at, mlp = L.self_attn, L.mlp
        assert at.q_proj.bias is None and at.o_proj.bias is None and at.k_proj.bias is None and at.v_proj.bias is None
        out.append(dict(ln1=L.input_layernorm.weight.float().contiguous(), e1=float(L.input_layernorm.variance_epsilon),
                        wqkv=torch.cat([at.q_proj.weight, at.k_proj.weight, at.v_proj.weight], 0).contiguous(), wo=at.o_proj.weight,
                        qn=at.q_norm.weight.float().contiguous(), kn=at.k_norm.weight.float().contiguous(),
                        eqk=float(at.q_norm.variance_epsilon), sc=float(at.scaling),
                        ln2=L.post_attention_layernorm.weight.float().contiguous(), e2=float(L.post_attention_layernorm.variance_epsilon),
                        wgu=torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0).contiguous(), wd=mlp.down_proj.weight))
    return out


def rope_f32(dec, dev, max_pos):
    """HF rotary with an fp32 activation -> fp32 cos / sin [max_pos, D] (the trainer's: rotary_emb returns x.dtype)."""
    with torch.cuda.device(dev):
        c, s = dec.model.rotary_emb(torch.zeros(1, 1, 8, dtype=F32, device=dev), torch.arange(max_pos, device=dev)[None])
    assert c.dtype == F32
    return c[0].contiguous(), s[0].contiguous()


def trunk_step(layers, nw, ne, K, V, cos, sin, x, j, nh, kv, D, lin, silu_mul_gu):
    """one decoder step for S rows: x [S, L, Hl] fp32 input rows (j = 0: L = 2 (slot 0, cb0) at positions 0, 1; else L = 1 at j + 1) ->
    the final-normed hidden of each row's last position, bf16 [S, Hl]."""
    S, L, Hl = x.shape
    t0 = 0 if j == 0 else j + 1
    c, s = cos[t0:t0 + L], sin[t0:t0 + L]
    res, xn = add_rms_f32(x.reshape(S * L, Hl).float().contiguous(), None, layers[0]["ln1"], layers[0]["e1"])
    mo = None
    for i, Ly in enumerate(layers):
        if i:
            res, xn = add_rms_f32(res, mo, Ly["ln1"], Ly["e1"])
        qkv = lin(xn, Ly["wqkv"])
        a = attn_exact(qkv, Ly["qn"], Ly["kn"], Ly["eqk"], c, s, K[i], V[i], t0, L, nh, kv, D, Ly["sc"])
        a = lin(a, Ly["wo"])
        res, h2 = add_rms_f32(res, a, Ly["ln2"], Ly["e2"])
        mo = lin(silu_mul_gu(lin(h2, Ly["wgu"])), Ly["wd"])
    hl = final_norm_f32(res, mo, nw, ne)
    return hl.view(S, L, Hl)[:, -1]
