# Checkpoints and native conditioning

All three models and the frame-0 data filter are selected. A run JSON explicitly
records the checkpoint paths, raw-code data variant and audio gain policy.

| Role | Selected export | Pinned Hugging Face revision |
|---|---|---|
| Student | `/home/param/models/imported-weights/ppm-aloe-poseduplex-raw-final/export-424` | [`InternalCan/ppm-aloe-poseduplex-raw-final`](https://huggingface.co/InternalCan/ppm-aloe-poseduplex-raw-final/tree/506aebef47895ffcc9b83194e37abff2e6e97188), `506aebef47895ffcc9b83194e37abff2e6e97188` |
| Pose teacher | `/home/param/models/imported-weights/ppm-aloe-monoplex-pose-raw-bandfix-355/weights/export-355` | [`InternalCan/ppm-aloe-monoplex-pose-raw-bandfix-355`](https://huggingface.co/InternalCan/ppm-aloe-monoplex-pose-raw-bandfix-355/tree/3a10893132063793791072e12f0a21d15e9efc25), `3a10893132063793791072e12f0a21d15e9efc25` |
| Audio teacher | `/home/param/models/imported-weights/ppm-aloe-v2vox-raw-w14503/export-253` | [`InternalCan/ppm-aloe-v2vox-raw-w14503`](https://huggingface.co/InternalCan/ppm-aloe-v2vox-raw-w14503/tree/40869a83b5052b081236e060484f08511cec9f8c), `40869a83b5052b081236e060484f08511cec9f8c` |

Student step 424 is the final two-epoch export: 212 steps per epoch. Step 211
is the older hosted checkpoint. The local rank-shard export is selected because
its metadata preserves exact fp32 `vgain_theta_fp32=-5.571630477905273`; the
same-step DCP's `train_cfg.json` does not preserve that value. Both contain the
same step-424 model weights. [Plan](https://huggingface.co/InternalCan/ppm-aloe-poseduplex-raw-final/blob/506aebef47895ffcc9b83194e37abff2e6e97188/config/plans/PLAN_7da6116e9f61e36b.json#L31),
[gain metadata](https://huggingface.co/InternalCan/ppm-aloe-poseduplex-raw-final/blob/506aebef47895ffcc9b83194e37abff2e6e97188/export-424/meta.json#L75).

## Conditioning for this run

The recipe supplies a B **audio-only reference whenever one is available**,
identically to student and both teachers. A row without a supported clip uses
the native empty five-token system prefix in all three models. There is no
reference transcript, persona or reference pose; native boundary markers remain.
The first 25 physical conversation frames (two seconds) prefill all three B
streams and receive no KL or sampled-action credit.

Native student training drew a B voice reference on 30% of items, without
persona. Native mono training used independent 80% voice and 20% persona draws;
only 28.18% of its training items actually had an available reference. Both
used audio-only references from the same speaker's nearest eligible later
chunk in the recording's second half, with 38–125 frames and a blank text lane.
The training loader has no same-row fallback. OPD always uses available clips,
without the original random conditioning dropout, and uses the native empty
prefix when none is available. [Student prefix](https://huggingface.co/InternalCan/ppm-aloe-poseduplex-raw-final/blob/506aebef47895ffcc9b83194e37abff2e6e97188/code/train/code/data_aloe_v2.py#L81),
[mono prompt recipe](https://huggingface.co/InternalCan/ppm-aloe-monoplex-pose-raw-bandfix-355/blob/3a10893132063793791072e12f0a21d15e9efc25/README.md#L71),
[clip selection](https://huggingface.co/InternalCan/ppm-aloe-monoplex-pose-raw-bandfix-355/blob/3a10893132063793791072e12f0a21d15e9efc25/code/train/data_ppm.py#L93).

The monoplex teacher's A lane stays empty, matching its training. Student and
audio teacher share the A waveform/frozen Voxtral latents, then apply their own
trained adapters. A contributes no transcript or Mimi input tokens. Neither
selected pose checkpoint requires framing, G0 or a same-frame audio-to-pose
conditioning module; both use the D33 backbone link at alpha 1.

**Raw B codes and native A waveform are separate choices.** Both selected
checkpoints trained with raw Mimi codes, including the reference. The student's
raw-code wrapper replaces code values from mapped raw stores while preserving
the selected rows, pose, words and A waveform. Native A uses its indexed FLAC
and raw gain: 0 dB for WLT FLAC, inverse recorded normalization for decoded-Mimi
fallbacks. SAM-separated store names alone do not imply that A's waveform is
SAM-separated. No random A masking or augmentation appears in that data path.
[Raw-code audit](https://huggingface.co/InternalCan/ppm-aloe-poseduplex-raw-final/blob/506aebef47895ffcc9b83194e37abff2e6e97188/config/RAW_CODES.md#L10),
[A waveform/gain implementation](https://huggingface.co/InternalCan/ppm-aloe-poseduplex-raw-final/blob/506aebef47895ffcc9b83194e37abff2e6e97188/code/train/code/data_aloe_v2vox.py#L68).

The installed provenance audit compared all 11,777 mapped rows: **23,554 A
entries, zero mismatches** between original, source remapped and installed audio
indices. FLAC identity, generation, byte count, offsets, lengths and gains match;
all retained entries are WLT inputs at 0 dB raw gain. The original/new gain-index
hash prefixes (`2acb4c7e4fce2c9e` / `e2e679c638ea8f3a`) match the audio/student
checkpoint configs. The current row cache, waveform decoder and Voxtral front
source files are byte-identical to the student's packaged training versions.
This verifies provenance and preprocessing; it does not claim a new exhaustive
waveform decode or GPU latent comparison.

The original training excluded reference-prefix losses, but did not exclude
the first two seconds of conversation. That exclusion is specific to this OPD
recipe; two-second prefill also appears in the published inference settings.
[Training targets](https://huggingface.co/InternalCan/ppm-aloe-poseduplex-raw-final/blob/506aebef47895ffcc9b83194e37abff2e6e97188/code/train/code/lib/data_s3q.py#L309).

## Bandfix and data selection

Both source runs masked ground-truth pose losses for black bands/frozen runs
while retaining those pose inputs. They also dropped lanes whose initial
skeleton made frame-0 normalization invalid. Those ground-truth timestamp loss
masks are **not applied to newly sampled OPD pose**. Among 15,898 candidate
orientations, 2,320 lack a source pose track that passes dataset validation and
12 have invalid frame-zero normalization. The selected recipe retains
**13,566 orientations: 6,410 with a reference and 7,156 with an empty prefix**.
The source pose is required for the first two seconds of conversation prefill,
not as supervision for the subsequent generated rollout. Skipped rows and
admitted clip counts are audited.
[Source bandfix rules](https://huggingface.co/InternalCan/ppm-aloe-poseduplex-raw-final/blob/506aebef47895ffcc9b83194e37abff2e6e97188/README.md#L55).

## Audio teacher and gain precision

The selected `ppm-aloe-v2vox-raw-w14503/export-253` is native PPM without pose,
initialized from `w-14503` and trained as job 25963. Its raw Mimi, voice-only
reference and native raw-gain A distributions match the selected data path.
[Pinned configuration](https://huggingface.co/InternalCan/ppm-aloe-v2vox-raw-w14503/blob/40869a83b5052b081236e060484f08511cec9f8c/export-253/meta.json).

This export stores learned `vgain.theta` in bf16 and lacks exact fp32 gain
metadata. The explicit run setting `audio_teacher_gain_policy="exported"`
retains that saved scalar. The original fp32 training gain is not recovered or
replaced by the initial/recalibrated gain from configuration. The student uses
its own exact fp32 gain under the default `require_fp32` policy. Each model
keeps its own trained adapter and gain.

The downloaded audio export stores `theta=-5.71875` (scale approximately
`0.003283813`). Both selected rank exports omit `vadapter.target_rms`, an unused
buffer when their adapter normalization is `none`. The loader reconstructs its
native NaN sentinel and records this in the load receipt; learned weights and
normalization modes that consume this buffer still require strict loading.
