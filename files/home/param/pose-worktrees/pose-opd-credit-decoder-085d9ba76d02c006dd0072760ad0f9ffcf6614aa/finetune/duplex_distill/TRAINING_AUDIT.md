# Pose duplex distillation: implementation audit

This audit describes the current trainer and resolved seven-node recipe. Production selects microbatch two: 112 examples per update, 284 updates, and nine warmup updates.

## Models and execution

The student starts from `InternalCan/ppm-aloe-poseduplex-v5-final`, export 498. Frozen speech/text supervision comes from `InternalCan/ppm-aloe-samsep-vox17127-final`, export 249; frozen pose supervision comes from `InternalCan/ppm-aloe-v2pure-main-17850-final`, export 200. The latter’s historical local directory is `transcript-pose/xabi-original`; it does not identify the training dataset.

All student parameters remain trainable, including both depth decoders, conditioning adapters, gain, and backbone-to-pose-decoder link. Teachers use evaluation mode, disabled parameter gradients, and `no_grad`; their distributions are detached. A separate frozen Voxtral encoder supplies audio features. Every update checks finite losses, absent teacher gradients, and nonzero student gradients.

Execution uses landsend-0,1,2,3,4,5,12, eight H100s each: 56 ranks. Microbatch one means 56 examples per update; microbatch two means 112. Neither uses gradient accumulation. Initialization reshards production checkpoints to this world size and restores exact FP32 gain sidecars.

## Data and conditioning

The only dataset is `/home/param/datasets/aloe-duplex`. The pinned production loader defines splits, masks, delays, prompts, and deterministic ordering. Each epoch contains 15,898 oriented examples. Both batch sizes cover exactly 31,796 real examples over two epochs, with six zero-weight dummy slots per epoch. Batch one takes 568 updates; batch two takes 284. The 316 held-out orientations are never evaluated.

Each row contains one conversation’s first maximum 7,500 frames, without random cropping or filling short rows with another conversation. Prefix and delay extend the timeline: `T = prefix_length + conversation_frames + 2`. Text predicts `n-1` next-token positions, including words, padding, NEW_WORD, and EOB markers.

With microbatch two, two independently prepared conversations are packed into one forward. Segment boundaries isolate attention and reset positions; each row retains its own prefix, masks, and audio features. Waveforms are encoded separately. The normalization denominator includes both rows across all ranks. Dummy rows contribute neither selected depth frames nor loss/counts.

The B voice-conditioning draw has probability 0.30 and is deterministic by conversation, orientation, and seed. A positive draw can lack an available clip; available clips contain 38–125 frames. There is no persona conditioning or attention dropout. The same draw persists across epochs.

Student and teachers share supported B conditioning and dataset histories. Student and audio teacher share real A audio and one frozen Voxtral encoding per row; the audio teacher excludes pose. The monoplex teacher receives its native empty A lane: PAD text, absent audio codes, zero presence, and no Voxtral features. Thus A conditioning differs across architectures.

All predictions use teacher forcing on dataset histories. Depth decoders receive dataset codebook prefixes, not generated rollouts. Missing pose masks the entire affected side while preserving speech/text supervision.

## Objective and normalization

Every optimized distribution uses forward KL at temperature one over the complete vocabulary:

```text
KL = sum(p_teacher * (log(p_teacher) - log(p_student)))
speech = 0.25 * speech_cb0_KL + 0.75 * mean(speech_cb1..31_KL)
pose   = 0.25 * pose_cb0_KL   + 0.75 * mean(pose_cb1..31_KL)
loss   = (text_KL + speech + pose) / 3
```

Each output uses globally summed valid-position counts; local losses multiply by 56 before distributed gradient averaging. Higher-codebook means divide by 31. CB0 uses every valid frame. Both depth decoders use the same deterministic 10% conversation-frame sample, aligned by their delays, and reuse that sample across epochs. Missing modalities do not redistribute their weights.

Text KL’s lexical weight is one, giving full-vocabulary KL. Separate dataset diagnostics retain production text weighting: category CE plus five times conditional lexical CE. Dataset labels determine histories, validity, diagnostic categories, and CE targets; they never replace teacher distributions or contribute CE gradients.

## Backward and optimization

Frozen teacher forwards precede the student. The text head chunks positions and precomputes hidden-state/output-row gradients while FSDP parameters are gathered. Chunk size and direct-addmm implementation are explicit run configuration. CB0 KL remains connected to the student graph.

Higher-codebook decoding uses chunks capped at 470 frames. Detached student tower/link outputs become gradient roots. Each chunk runs teacher forward, student forward, KL, and immediate backward. The 0.75/3 coefficient is applied once. Replicated decoder gradients are reduced, averaged, clipped, and updated once; accumulated root gradients then enter one backbone backward alongside text and CB0. Main gradients are clipped before main AdamW updates.

Student decoder layers checkpoint activations; backbone/tower checkpointing comes from production. Computation uses BF16, with FP32 master parameters, optimizer state, and parameter-gradient reductions. Detached roots accumulate gradients in activation dtype; count/statistic reductions use FP64. FA3, fused kernels, and fused AdamW remain enabled; depth decoding is eager.

AdamW uses betas (0.9,0.95), epsilon 1e-8, and zero weight decay. Peak learning rates are trunk 3e-6; special output rows and audio/pose embeddings 6e-5; projection 3e-5; towers 1.5e-5; voice adapter/gain 5e-4; pose link 1e-5; speech decoder 1.5e-5; pose decoder 2e-5. Trunk clipping is 0.5; other groups use 1.0. Batch one warms up for 19 updates and decays toward step 568; batch two uses nine and 284. Update index zero has zero learning rate but updates Adam moments.

## Logging and persistence

Separate metric namespaces distinguish aggregate KL, text-category KL, individual codebooks, aggregate dataset CE, text-category CE, and dataset codebooks. Dataset CE is detached. There is no evaluation loop or evaluation scheduling.

After a completed update, rank zero requests a save when 7,200 seconds have elapsed since the last completed save. Training also saves at its final update and on a graceful stop. Every node writes its local FP32 model/optimizer/RNG resume shards and BF16 export shards under `/home/param/models/trained-weights/<run-name>/checkpoints`. Logs, metrics, and control files remain under `/home/param/runs/pose-runs/<run-name>` (or `/home/param/runs/smokes` for a bounded check). The run-local `checkpoints` path is a compatibility link; validation and retention operate on the canonical model directory. All ranks verify the new resume files before every node atomically updates `checkpoints/LATEST_RESUME.json`; only then can older resume directories be deleted. Older BF16 exports remain.

Synchronized save failures preserve previous checkpoints. Periodic failures continue training and retry after 60 seconds; final/stop saves try three distinct attempt paths, five seconds apart, then fail visibly. Transport errors propagate. Marker/pruning failures warn and retain old resumes.

Resume requires the same host/rank mapping, microbatch, and schedule. W&B uses the recorded ID with `resume="must"`; missing or inconsistent metadata is rejected. Portable exports require gathering all 56 rank shards plus matching replicated state and metadata. A node’s completion marker covers only its own files. The CPU suite verifies exact resume behavior, retention safety, migrated model paths, and launcher source synchronization. The separate local two-rank CPU proof checks DTensor save/resume and exact subsequent AdamW updates. Historical GPU runs and fault-injection reports are recorded with their run outputs; no such run is required by the CPU unit suite.
