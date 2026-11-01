# With/Without Objects

Utilities for creating object-conditioned splits and tokenizing pose data per split.

## Quick start for the agent (32cb default)

Before running the split pipeline, verify the recommended tokenizer is available:

```bash
uv run python scripts/smoke_test_agent_tokenizer.py
```

Recommended pose tokenizer for this flow:
- `InternalCan/base-causal-2x-wide-32cb-qd0.5-lr1e-4-lossshift1-root-frame-0-cb_size_2048`
- local fallback: `/mnt/som-fs/james-checkpoints/pose-tokenizer-james/32cb_size_2048/base-causal-2x-wide-32cb-qd0.5-lr1e-4-lossshift1-root-frame-0-cb_size_2048/step_200000`

## Files

- `prepare_with_without_resources.py`
  - Builds split resources under `resources/`:
    - `<bucket>_all.csv`
    - `<bucket>_with_objects.csv`
    - `<bucket>_without_objects.csv`
    - `with_objects_clip_ids.txt`
    - `without_objects_clip_ids.txt`
    - `with_objects_pose_manifest.txt`
    - `without_objects_pose_manifest.txt`
    - `with_objects_questions.txt`
    - `without_objects_questions.txt`
    - `with_objects_question_answer.tsv`
    - `without_objects_question_answer.tsv`
- `tokenize_with_without_objects.py`
  - Runs `scripts/tokenize_chunks.py` for:
    - `with_objects`
    - `without_objects`
- `collate_with_mimi_questions_answers.py`
  - Filters Mimi speech rows by split
  - Builds chunk audio+pose token rows
  - Attaches question/answer text
  - Collates interleaved `input_ids` with `question` + `answer` metadata

## Usage

From repository root:

```bash
cd /home/james/pose-tokenizer
```

### 1) Prepare split resources

Use an existing CSV:

```bash
uv run python with_without_objects/prepare_with_without_resources.py \
  --bucket hands \
  --source-csv /home/james/pose-tokenizer/hands_all.csv
```

Or query DB directly (requires `TIKTOK_DB_DSN`):

```bash
uv run python with_without_objects/prepare_with_without_resources.py --bucket hands
```

### 2) Tokenize both splits

```bash
uv run python with_without_objects/tokenize_with_without_objects.py \
  --checkpoint /path/to/checkpoint.pt \
  --pose-dir /path/to/flat_pose_npy_dir \
  --dims-csv /path/to/dims.csv \
  --extra-args "--num-workers 8 --batch-size 128"
```

Outputs go to:

- `with_without_objects/tokenized_pose_splits/with_objects/`
- `with_without_objects/tokenized_pose_splits/without_objects/`

### 3) Collate with Mimi + questions + answers

```bash
uv run python with_without_objects/collate_with_mimi_questions_answers.py \
  --split both \
  --pose-dir /mnt/som-fs/pose_cond/pose_cond_npy \
  --dims-csv /mnt/som-fs/pose_cond/dims_upperbody_ideal_fresh.csv \
  --pose-tokenizer InternalCan/base-causal-2x-wide-32cb-qd0.5-lr1e-4-lossshift1-root-frame-0-cb_size_2048 \
  --collate-script /home/james/pose-LLM-trainer/collation_32.py \
  --questions-source resource_tsv
```

Outputs go to:

- `with_without_objects/collated_with_mimi_questions_answers/with_objects/`
- `with_without_objects/collated_with_mimi_questions_answers/without_objects/`

## Notes

- Manifests in `resources/` are clip-based (`<clip_id>_poses.npy`), so `--pose-dir` should contain matching files.
- For clip-level question mapping, the script uses `/mnt/som-fs/pose_cond/tiktok_questions/tiktok_clip_questions.parquet` by default.
- For portable runs (no shared FS), use `--questions-source resource_tsv` so it reads split QA TSV files from `resources/`.
- For 32cb/2048 tokenizers, use `--collate-script /home/james/pose-LLM-trainer/collation_32.py`.
- For 12cb/1024 tokenizers, use `--collate-script /home/james/pose-LLM-trainer/collate.py`.
