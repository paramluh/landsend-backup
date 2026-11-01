# MoT 14B base (w-14810)

This repo holds the base checkpoint of the **MoT 14B**, a two-channel speech-duplex model. It is the run `ppm-mot` at optimizer step **14,810**, exported as torch DCP in bf16.

The model is a Mixture-of-Transformers version of our interleaved ("INTL") 14B speech model. Each 12.5 Hz frame is split into three backbone positions:

- **U**: the user's word lane;
- **A_t**: the assistant's word lane;
- **A_s**: the assistant's Mimi audio codes.

Each slot has its own full Qwen3-14B expert. All three experts share one residual stream and one global causal attention.

This checkpoint is the base for the MoT ASR-adapter runs (motasr2) and for motvox (w-15639).

## Repo layout

| path | contents |
|---|---|
| `w-14810/` | the export, byte-identical to the copy on disk: `.metadata`, `COMPLETE` (holds `14810`), `__0_0.distcp` … `__127_0.distcp` (130 files, 86,433,443,557 bytes) |
| `_backup/MANIFEST.w-14810.tsv` | the upload manifest: path, size and source path of every file |
| `README.md` | this file |

The same export is also stored in `InternalCan/ppm-mot-14b/w-14810/`, alongside the run's other steps.

## Architecture

**Layout.** Row frame *p* becomes three backbone positions: U[p] = 3p, A_t[p] = 3p+1 and A_s[p] = 3p+2.

**Inputs**
- U: the embedding of channel A's (user's) word-lane token (`inp.emb.0`).
- A_t: the embedding of channel B's (assistant's) word-lane token (`inp.emb.1`).
- A_s: the sum of the embeddings of B's 32 Mimi codes, at code delay CD = 2 frames (`inp.aud.1`).
- The user's audio is not an input. `inp.aud.0` is present but frozen and unused.

**Experts.** Each of the 40 backbone layers holds three full Qwen3-14B decoder layers, `layers.i.experts.k` (k = 0 U, 1 A_t, 2 A_s). Every expert has its own:
- input RMSNorm;
- Q/K/V/O projections, including q_norm and k_norm;
- post-attention RMSNorm;
- MLP.

The dimensions are hidden 5120, 40 query heads, 8 KV heads and MLP 17,408.

**Routing and attention**
- Routing is static and strided: expert k processes exactly the positions 3p+k.
- Attention is one causal attention over all 3T positions of a packed segment, using FA3 varlen.
- RoPE frequencies are divided by 3, so frame p's three slots sit at p, p+1/3 and p+2/3.

**Reads (v2)**
- The text head reads U[p]'s output, normalised by `head.norm` (the U expert's final norm). It predicts the assistant word-lane token at A_t[p], the next position.
- The cb0 head and the light decoder read A_t[p]'s output, normalised by `head.norm_t`. They predict frame p's 32 Mimi codes, which are the codes input at A_s[p].
- The A_s output has no loss and no final norm. As a result, the last layer's A_s expert gets no gradient in its q_proj, q_norm, o_proj, post-attention norm or MLP.

**Heads**
- Text head: Qwen3 `lm_head` (151,936 × 5120), a gated low-rank projection (rank 2048), and three extra output rows for PAD, NEW_WORD and EOB.
- cb0 head: 5120 → 2048.
- Speech decoder `dec_s`: a 14-layer Qwen3-0.6B-shaped decoder (hidden 1024). It is conditioned on the A_t output (projected 5120 → 1024) and on cb0, and it predicts codebooks 1–31 as a 32-position sequence.

### Parameter counts

These are from the training build log; the totals match the export's `.metadata`.

| part | params |
|---|---|
| `inp`: word-lane embeddings A/B (151,936 × 5120 each) + Mimi code embeddings A/B (65,536 = 32 × 2048 rows × 5120 each) | 2,226,913,280 |
| `layers`: 40 layers × 3 experts | 39,637,432,320 |
| of which, per expert (U / A_t / A_s, each) | 13,212,477,440 |
| `head` (text head and final norms) | 835,609,600 |
| `cb0_head` | 10,485,760 |
| `dec_s` (speech light decoder; 220,233,216 in its 14 layers) | 357,598,720 |
| **total** | **43,068,039,680** |

## Tokens and streams

- **Frame rate:** 12.5 Hz.
- **Audio:** Mimi, with 32 codebooks of 2,048 entries each, for 65,536 code embeddings per channel. Code delay is CD = 2.
- **Text:** the Qwen3 tokenizer and vocabulary (151,936 rows), carried as a per-frame word lane.
  - A word's first piece sits on frame floor(start × 12.5); later pieces go on the following frames.
  - The other lane tokens are PAD (151670), NEW_WORD (151671, on the frame before a word's first piece) and EOB (151645, `<|im_end|>`).
- **Channels:** channel A is the user, channel B is the assistant. Roles are as the data labels them, with no A/B swap.
- **System prompt:** speech rows start with a prefix of at most 2,000 positions: a header, voice clips, persona text and a footer.

## Training

| item | value |
|---|---|
| init | Every expert is a copy of Qwen3-14B, so the three are bit-identical at step 0. Every other tensor uses the INTL 14B's step-0 rule (seed 20261011). |
| hardware | 16 nodes × 8 GPUs (world 128), FSDP2, bf16 compute with fp32 master weights. No FP8. |
| steps | 14,810 optimizer steps of a planned 17,366, which is about one pass over the speech rows. Step 14,810 ≈ 85% of that pass. |
| batch per step | Gradient accumulation of (speech, speech, text) per GPU: 256 speech rows + 128 text packs per optimizer step |
| speech rows | About 4.45M two-channel podcast-conversation rows: Mimi codes for both speakers plus word-timed transcripts, up to 7,500 frames (600 s) each, including the system-prompt prefix. The eval episodes (salt `s3eval-v1`) are held out. |
| text rows | Each step and rank draws 50/50 between two sources, packed to ≤ 7,500 frames: Gemma-generated text-duplex conversations (2,585,267 packs, same word-lane format, no audio) and corpus15m chat (1,055,497 packs). |
| losses | Factorised text CE on the assistant word lane: word, PAD/timing, NEW_WORD and EOB terms (×1 each), plus a word term weighted ×5. Also Mimi cb0 CE (×1) and decoder cb1–31 CE (×1). |
| optimizer | AdamW (β 0.9 / 0.95, weight decay 0). Trunk LR 5e-5 (experts, word embeddings, lm_head, final norms), constant after a 100-step warmup (no decay). Other groups: PAD/NEW_WORD/EOB rows 6e-5, head projection 3e-5, Mimi code tables 2e-4, cb0 head 3e-4. Grad clip 1.0. |
| decoder updates | 8 updates per micro over a disjoint partition of all frames. LR 4e-4 after a 400-update warmup, constant, clip 0.5. |
| period | 2026-10-23 06:16 PT (step 0) → 2026-10-27 ~11:16 PT (step 14,810), Slurm jobs 28071, 28155, 29514 and 30289 |

## Eval

The eval set is `eval_speech_s3`: 103 held-out podcast rows. Each row is scored with either speaker as the assistant (lanes A and B), and the results are pooled.

| metric (step 14,810) | pooled | lane A | lane B |
|---|---|---|---|
| normalised word CE | **1.9992** | 2.0152 | 1.9829 |

This is the trainer's logged value. On 2026-10-28, loading this export with the loader below reproduced it as 1.999170.

## File format

- **Format:** a torch Distributed Checkpoint (DCP) written by the trainer's consolidated export: one `.metadata` file plus 128 `__<rank>_0.distcp` shards.
- **Contents:** 1,495 tensors, all `torch.bfloat16`, with the model's own parameter names (`inp.*`, `layers.<i>.experts.<k>.*`, `head.*`, `cb0_head.*`, `dec_s.*`).
- **Not included:** optimizer state and fp32 masters.
- **Readable with:** `torch.distributed.checkpoint` at any world size, including a single process.

## How to load

**Code**
- The MoT trainer's code tree is `/data/home/amartyavarma/ppm/mot/code`. It contains `model_mot.py`, `model_ilp.py`, `model_ppm.py`, `lib/` and `vendor/`.
- Python: `/data/home/amartyavarma/venvs/qwen-s2/bin/python`.
- The Qwen3-14B config is read from the local copy.

**Environment variables:** `PPM_CD=2`, `PPM_DEC_ATTN=attn32` and `HF_HUB_OFFLINE=1`.

**Download**

```python
from huggingface_hub import snapshot_download
root = snapshot_download("InternalCan/mot-14b-base", allow_patterns=["w-14810/*"])
EXPORT = f"{root}/w-14810"
```

**Single process, two 80 GB GPUs (inference)**

This is the loader used by the MoT live engine, `ppm/mot/live/engine.py` (`build_model` / `load_into`). It loads every tensor, including the speech decoder.

```python
import os, sys, torch
os.environ.setdefault("PPM_CD", "2"); os.environ.setdefault("PPM_DEC_ATTN", "attn32"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
CODE = "/data/home/amartyavarma/ppm/mot/code"
for p in (f"{CODE}/vendor", f"{CODE}/lib", CODE):
    sys.path.insert(0, p)
import model_ppm as M
import model_mot as MOT
import torch.distributed.checkpoint as dcp

c14, c4, c06 = M.cfg_backbone("qwen3-14b", 40), M.cfg4b(36), M.cfg06(14)
d0, d1 = torch.device("cuda:0"), torch.device("cuda:1")       # ~86 GB of bf16 weights -> split over two GPUs
torch.set_default_dtype(torch.bfloat16)
with torch.device("meta"):
    m = MOT.MoT(c14, c4, c06, 2048, 151682, rope_pos_div=3, read_pos="v2")
torch.set_default_dtype(torch.float32)
for name, ch in m.named_children():                         # layers 0-19 + inputs on GPU 0, the rest on GPU 1
    if name == "layers":
        for i, L in enumerate(ch):
            L.to_empty(device=d0 if i < 20 else d1)
    else:
        ch.to_empty(device=d0 if name == "inp" else d1)
with torch.cuda.device(d1):
    m.dec_s.model.rotary_emb = type(m.dec_s.model.rotary_emb)(config=c06, device=d1)
params = dict(m.named_parameters())                         # == the export's 1,495 keys, all bf16
dcp.load(params, checkpoint_id=EXPORT, no_dist=True)
m.eval().requires_grad_(False)
```

**FSDP2, teacher-forced eval (multi-GPU)**

`/data/home/amartyavarma/ppm/probes-1028-mot/probe_light.py` is the harness that reproduced the logged eval.
1. It builds the model with `MOT.build(c14, c4, c06, dev, partial=False, proj_r=2048, tower_vocab=151682, init=False, dec_mode="replicate", rope_pos_div=3, backbone_root=M.BACKBONES["qwen3-14b"][0], read_pos="v2")`.
2. It loads the export with `dcp.load(sd, checkpoint_id=EXPORT, planner=DefaultLoadPlanner(allow_partial_load=True))`. Here `sd` holds every non-`dec_s.*` parameter, because the text eval does not run the decoder.
3. It runs `model(batch, rot, rot)` with `rot = M.Rot(c14, dev)`.

It is launched through `ppm/motvox/asr_eval/task.sh` (srun, one task per GPU) with `PPM_CD=2 PPM_DEC_ATTN=attn32`.

## Provenance

- **Run:** `ppm-mot` (MoT 14B, `MOT_INIT=base`), step 14,810. The env is `/data/home/amartyavarma/ppm/mot/mot.env`; the design and test write-up is in `/data/home/amartyavarma/ppm/mot/STATUS.md`.
- **Source on disk:** `/data/home/amartyavarma/ppm/mot/ckpt-weka/ppm-mot/w-14810/`, exported 2026-10-27 18:16 UTC (11:16 PT).
- **Upload:** uploaded 2026-10-28 PT (2026-10-29 UTC) with `hfb_up.py`, unchanged.
- **Verification:**
  - The HF file list, sizes and LFS sha256 of every weight file match the local files.
  - 3 randomly chosen shards were streamed back from HF and their sha256 matched the local files.
