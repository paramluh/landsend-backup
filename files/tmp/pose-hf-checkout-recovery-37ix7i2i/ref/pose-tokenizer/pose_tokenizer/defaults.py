"""Project-wide defaults for agent-friendly tokenizer usage."""

from __future__ import annotations

AGENT_DEFAULT_TOKENIZER_REPO = (
    "InternalCan/base-causal-2x-wide-32cb-qd0.5-lr1e-4-lossshift1-root-frame-0-cb_size_2048"
)

AGENT_DEFAULT_LOCAL_CHECKPOINT = (
    "/mnt/som-fs/james-checkpoints/pose-tokenizer-james/32cb_size_2048/"
    "base-causal-2x-wide-32cb-qd0.5-lr1e-4-lossshift1-root-frame-0-cb_size_2048/"
    "step_200000"
)

