---
tags: [sam, pose, tts, private]
---
# SAM 8B pose fine-tune on Aloe v2pure, PPM-26089-comparable

bf16 weights (`model.safetensors`, SAM PoseModel layout = `InternalCan/sam-speech-pose-stage2-fast` export format), init SAM stage-2 step 61258. 200 steps = 1 epoch of the Aloe pose release v2 active-speaker clips (367,495), with PPM job 26089's per-step clip assignment. Recipe and every difference vs SAM's own v2pure fine-tune: `run/code/train/train_pose_cmp.py` docstring and `run/code/train/sam_cmp.env`; metrics: `run/logs/metrics.jsonl`.
