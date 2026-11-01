# PPM — pose + podcast + text-duplex model (design spec, 2026-10-11)

Project page: https://claude.ai/artifact/Hvbf3MUbpSHXokBbrxX3oh (task / decision codes T*, D* below refer to it).

## Architecture
- **Backbone**: base Qwen3-32B (D13), plain RoPE on the frame index (no mix-RoPE), no LoRA; full fine-tune with an aggressive LR (forgetting accepted).
- **Input** (per frame, summed like stage 3): for each channel c ∈ {A=user, B=assistant}: emb_c(text lane) + aud_c(Mimi codes of frame − CD);
  pose rows add pose_emb(pose codes of frame − POSE_DELAY) on channel B. emb_A / emb_B init = copies of Qwen3-32B's embedding; aud_c and
  pose_emb random. CD = POSE_DELAY = 3.
- **Text head**: Qwen3-32B lm_head + the speech control rows (PAD / NEW_WORD / EOB), per-channel heads as in stage 3 (projector family TBD in build).
- **Speech pair** (D12): Qwen3-4B tower (random init, D14) + 0.6B decoder (random init, D15).
- **Pose pair** (D12): a second Qwen3-4B tower (random) + a second 0.6B decoder (random).
- **Tower conditioning**: ONLY the backbone's final (normed) last hidden state at T − 1, projected 5120 → 2560 (no intermediate layers).
  Tower input at T = tower_text_emb(lane[T]) + tower_code_emb(codes of frame T − CD − 1) + proj(h_last[T − 1]).
  Tower embeddings sized for our inputs (D14): text vocab 151,680 (Qwen ids + our control ids 151670–151679), codes 32 × 2048.
- **0.6B decoder**: 32 output heads (D14) — see D16 (whether cb0 is predicted by the 4B or by the 0.6B).

## Data (per optimizer step = many task micros)
- **Pose micro**: packed pose rows (2.21M train; /scratch/amartyavarma/pose-s3-samdata/rows), 600 s, separator-packed clips, channel B only.
- **Speech micro** (2 per pose micro, D1): the 4.45M FULL 600 s podcast rows (D2/D3) of the stage-3 main store (s3-rows; slinky-14 slice
  rebuilt by T3; manifest T9). System-like prompt (D4): persona + voice A clip (3–10 s) + voice B clip (3–10 s), clips from later in the
  podcast, in its second half (D11); persona source D10.
- **Text micro**: two-track text-duplex rows, tracks summed like speech, pure CE (D7) — store T4 (all 3.28M Gemma WLT conversations,
  /scratch/amartyavarma/ppm/txtdup-full on slinky-13). Amount per step: D9.
- **Nanostepping**: many 0.6B updates per micro (as the pose trainer's chunked light updates).

## Build plan (T7)
Start from the pose trainer (stage3/train_pose_s3q.ga2.py, model_s3q.py, pose_s3q.py, txtdup_s3q.py, sysprompt.py):
1. model_ppm.py: plain-RoPE Qwen3-32B trunk (no adapters), S3-style 2-channel input + pose input, text head, 2 × (4B tower + 0.6B decoder).
2. data: reuse data_s3q (speech rows + sysprompt with D4/D11 config), pose_s3q (pose rows), txtdup_s3q (text CE packs).
3. train_ppm.py: GA micro schedule (pose : speech : text = 1 : 2 : D9), nanostepped 0.6B, per-module LR groups, FSDP as stage 3.
4. Step-0 checks: trunk == HF Qwen3-32B logits on a text row; towers/decoders produce uniform-ish losses (ln 2048 ≈ 7.62) at init.
