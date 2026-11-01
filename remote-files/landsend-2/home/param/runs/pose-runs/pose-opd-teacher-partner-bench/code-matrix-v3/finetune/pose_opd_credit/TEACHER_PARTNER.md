# Frozen conversational partner with silence replay

`recipe.teacher_partner.json` starts from the original two-epoch pose-duplex
student (`export-424`). The frozen raw audio-duplex scoring teacher
(`raw-w14503/export-253`) also generates the conversational partner. The pose
scoring teacher remains monoplex bandfix355 with CFG strength 3. Partner sampling
uses the same natural temperature-one categorical policy as the student.

The learner, modality/codebook weights, eight-frame dense score credit,
optimizer, LR schedule, three epochs, checkpoint interval and four saved samples
per update remain those of `recipe.raw.json`. Batch 320 still means320 student
trajectories; partner trajectories do not add optimizer examples. The initial
deployment groups 40 GPUs into ten four-GPU teams, with 32 concurrent student rows
and 32 corresponding partner rows per team. Student and partner each use two
GPUs. The eight-row learner microbatch per GPU is unchanged.

Every example retains its first 1500 physical frames (120 seconds). Its real
conversation prefill is independently sampled uniformly from 38..62 frames
(3.04..4.96 seconds, respecting the 80 ms frame grid). The draw is reproducible from
run seed, absolute update, rank, slot and source item, so epochs use different
draws. The same boundary applies to both speakers. Real text/audio/pose after
that boundary are removed from the student seed; the partner requires only its
own text/audio, not its pose. Available audio-only voice clips remain optional.

## Waveform clock and committed speech

There is no artificial timestamp shift in either waveform. At student physical
frame `i`, the student's audio tower needs partner PCM through
`(i+2)*1920+72` samples, exclusively. It therefore needs partner frame `i+1` in
full and the first 72 samples (3ms) of frame `i+2`. The partner produces those frames
using actual student audio where available and digital silence in its unknown
future. The student never consumes placeholder audio.

Once any part of a partner frame has been heard, that entire Mimi token frame
and its decoded waveform are committed. They cannot be redrawn. On receiving
the next student chunk, the partner rewinds only its conditioning state:

1. Replace the corresponding future-silence chunk with actual student audio.
2. Restore the hidden state and frontend convolution state immediately before
   the first affected frame. The 3 ms frontend lookahead makes this the preceding
   frontend frame, not the frame containing the new chunk itself.
3. Recompute the six affected backbone/frontend positions. Keep existing text
   and audio fixed, refreshing their tower KV states where necessary.
4. Sample and decode exactly one new teacher audio frame, append it to the
   student's incoming waveform, and advance the student.

The six-position replay is not six full generations: five tower positions use
committed tokens and skip text-head/light sampling. Only the new position
samples its 32 audio codebooks. Position-indexed random uniforms keep sampling
independent of how many times committed history was replayed.

Temporal KV arrays remain allocated; explicit cache positions overwrite the
invalid suffix and exclude stale future slots. Only small hidden/convolution
states are saved in a bounded rewind ring. Mimi state is never rewound because
the published output waveform is append-only. Bootstrap generates the partner's
necessary future frames first, including the first 3 ms needed by the student's
last prefill frontend frame.

The existing two terminal model positions still flush delayed student media
with forced PAD text and zero input-feature vectors. They are not new physical
conversation frames. Partner generation beyond 120 seconds is limited to the
lookahead needed by the last real input feature; it is not scored as student
output.

## Audio conventions and replay

Both prefills use their source raw 24 kHz mono PCM. Generated Mimi waveforms undo
the source speaker's recorded original raw-code loudness gain and apply the
live input peak bound 0.995 before entering the other model's streaming frontend.
The gain comes from original `RAWGAIN.json`, not SAM-separated metadata; the
recorded Mimi-decoded fallback convention is handled explicitly. No per-frame
loudness normalization is performed.

`streaming/` vendors the selected checkpoint's causal Mimi decoder and fused
Voxtral streaming frontend; source hashes and adaptation notes are in
`streaming/SOURCES.json`. The student's exact frozen Voxtral outputs are saved
for learner replay and audio-teacher scoring. The audio teacher applies its own
trained adapter to those same features. Reconstructing the dialogue or replacing
streaming features with a separately encoded waveform would change the context
and is not used for training replay.

The frozen partner is an environment transition conditional on sampled student
history. It has no trainable parameter derivative at fixed history. Existing
student score factors account for earlier sampled actions' effects on the
partner within the chosen eight-frame credit window. Partner samples do not
receive student log-probability terms. Full student network backpropagation
through its deterministic replay history remains enabled.

## Entrypoints and validation

Use the existing distributed entrypoint with the new recipe:

```sh
torchrun ... -m finetune.pose_opd_credit.train \
  --config finetune/pose_opd_credit/recipe.teacher_partner.json \
  --run-dir /path/to/new/run
```

The standalone four-GPU rollout benchmark accepts short prefixes while keeping
the actors' two-minute cache capacity:

```sh
python -m finetune.pose_opd_credit.teacher_partner_benchmark \
  --config finetune/pose_opd_credit/recipe.teacher_partner.json \
  --output /path/to/new/benchmark --instances 32 --frames 125 --repeats 2
```

The standalone benchmark does not include learner, KL-teacher or
Adam residency; full training capacity and speed require an integrated update.
CPU tests establish the clock, forcing, data and replay contracts. GPU numerical
parity, memory measurements and benchmark evidence must be recorded before
calling the new training setup validated.
