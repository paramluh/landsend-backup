# stage2-handover-20261008-conversations-v1  (PRIVATE, Hugging Face bucket `InternalCan/aloe-conversationa-data`)

**12,681 two-speaker conversations / 2,196.949 conversation-hours / 4,393.898 speaker-track-hours**
of Aloe studio/app calls, cut to the context-free conversation only, with per speaker: raw FLAC audio, word-level timestamps, and
`mimi-stream-v5` Mimi tokens. 1:1 mirror of `gs://aloe-processed-datasets/stage2-handover-20261008/` (same file names and bytes),
made 2026-10-27. The same conversations were later passed through SAM separation (`InternalCan/stage2-sam-sep-v1-mimi`,
`sam-sep-v1/` in GCS) — none of that is in here; this is the audio that went INTO SAM.

## Start here
    conversations.csv   one row per conversation, both speakers side by side, every label and every file path (see CSV_GUIDE.txt)
    hours.json          hours by crossbleed side / language / duration / opening / ending
    verification.json   the checks behind the "tokens == this audio" claim (below)

    pip install -U "huggingface_hub>=2.2"      # buckets API; log in with an InternalCan token (hf auth login)
    hf buckets cp  hf://buckets/InternalCan/aloe-conversationa-data/stage2-handover-20261008-conversations-v1/conversations.csv .
    hf buckets sync hf://buckets/InternalCan/aloe-conversationa-data/stage2-handover-20261008-conversations-v1 ./stage2-conversations        # full mirror, resumable (395 GB, 88,767 files)
    hf buckets sync hf://buckets/InternalCan/aloe-conversationa-data/stage2-handover-20261008-conversations-v1/mimi-stream-v5 ./tokens        # tokens only (12.7 GB)

## Layout (keyed by session / segment, exactly like GCS)
    wlt/<session>/<seg>/SPEAKER_00.flac        speaker 1 raw audio for the window: 16-bit FLAC, NO gain / normalisation; 48 kHz stereo for
                                               12,448 conversations, 48 kHz mono for 225, 44.1 kHz mono for 8 (columns lane_sample_rate / lane_channels)
    wlt/<session>/<seg>/SPEAKER_01.flac        speaker 2 (identical sample count; one known exception, see CSV_GUIDE lane_length_mismatch)
    wlt/<session>/<seg>/words.json             word-level timestamps of both speakers (`words[]`: text, start, end, _separated_speaker), seconds from the FLAC start
    wlt/<session>/<seg>/vad.json               VAD intervals per speaker (seconds from the FLAC start)
    mimi-stream-v5/<session>/<seg>/w00.SPEAKER_00.codes.npy   int16 [32, T], 12.5 Hz, T = ceil(samples_48k / 3840); frame k = samples [k*3840, (k+1)*3840)
    mimi-stream-v5/<session>/<seg>/w00.SPEAKER_01.codes.npy   same T
    mimi-stream-v5/<session>/<seg>/w00.mimi.json              encoder provenance (model revision, recipe, LUFS gain used, codes sha256)
    _meta/                                      copies of the GCS dataset metadata (manifest, cut decisions, crossbleed labels, purge manifests, FLAC conversion report)
    excluded_presence_gate.jsonl                conversations that exist in GCS but were NOT mirrored (one side has almost no speech after the cut)

## What the conversations are
Two strangers on a moderated Aloe call (booth phones in the London studio, and home phones), one aligned track per speaker. The
colleagues' stage-2 judge (`moonshotai/kimi-k3`, rules-v2) labelled every recording turn by turn; we kept ONE continuous window per
recording (`handover-cut-v3-conversation`): start at the judged greeting/introduction (or the restart point after booth/setup talk),
end at the goodbye, the natural end, or — strictly, at any confidence — just before the first turn where outside context leaks in
(prompts, monitor instructions, a third voice, tech talk, language switch). No cuts inside the window. Then: misalignment-risk
sessions removed, windows that had to start more than 15 s into the recording removed, windows under 60 s or over 25 min removed,
recordings with a no-face speaker removed, and a post-cut two-speaker presence gate (each side >= 2 s speech, >= 10 % of the speech,
>= 2 turns) — 0 conversation(s) failed it and are listed in `excluded_presence_gate.jsonl`.
All of this is visible per row in `conversations.csv` (start_rule, end_rule, opening_kind, ending, presence/reciprocal/developed, ...).

Crossbleed (the partner audible in a speaker's own audio) is LABELLED, not removed: `crossbleed_side` = none | SPEAKER_00 | SPEAKER_01 |
both (confirmed = acoustic `audible_bleed` or transcript tier `bleed`). 8,885 conversations /
1,413.874 h are clean on both sides.

## Tokens == this audio (verification.json)
The Mimi codes were produced on 2026-10-08 from the PCM16 clip WAVs of this dataset (one stateful mimi-stream-v5 pass per speaker over
the whole window, `kyutai/mimi` rev 89091b3e, BS.1770 gain to -23 LUFS applied inside the encoder only). The FLACs are lossless
copies of those WAVs. For EVERY conversation mirrored here:
- the encoder sentinel's source WAV (uri, bytes), decoded seconds and T agree with the WAV object still in GCS and with the FLAC's
  pinned source (md5, generation, size) and sample count: T == ceil(samples / sample_rate * 12.5), both speakers equal (for the 682
  conversations whose sentinel names the EU-bucket copy of the clip, that copy still exists and has the same md5 and size);
- every FLAC was parsed while streaming (STREAMINFO: the lane's sample rate / channels, 16 bit, total_samples == pinned sample count), crc32c == GCS;
- every codes.npy was parsed (int16 [32, T]) and its sha256 equals the sentinel's `codes_sha256`;
- every file's size was verified on HF after upload; a random read-back sample was byte-compared (sha256).
Re-encode spot check: 24 random conversations (48 speaker tracks, 4.176 h) re-encoded from the FLAC with the pinned mimi-stream-v5 encoder on an L4: 48 / 48 token streams byte-identical to the shipped codes.

## Not in this folder
The PCM16 WAVs (identical PCM to the FLACs), the colleagues' `system_prompts/` and `turn_taking*/` side folders, the SAM separation
stems and their tokens (`sam-sep-v1/`, `mimi-stream-v5-samsep/` in GCS; tokens on HF in `InternalCan/stage2-sam-sep-v1-mimi`), pose.

## Provenance
GCS dataset `gs://aloe-processed-datasets/stage2-handover-20261008/` (run `aloe-hc1008`, 2026-10-08; FLAC 2026-10-09; purges 2026-10-08/16).
Mirror run: `~/aloe-audio-pipeline/runs/2026-10-26-stage2-handover-hf-mirror/` (tool `tools/mirror_stage2_handover_to_hf.py`).
This README is written last: when it exists, the folder is complete and verified.
