# Aloe sam-sep FT with a persona text block (prep, 10-24)

A copy of job 29345 (run `ppm-aloe-samsep-vox17127`) with one change: on half of the train items, channel B's system prompt holds its speaker's persona prose.

## What changes

- **Data.** `code/data_aloe_persona.py` wraps job 29345's `data_aloe_v2vox.py` without changing it.
  - The trainer asserts the wrapped module's sha256 (`0b06343f..`).
  - Every other module in `code/` is a byte copy of 29345's code dir (`code/SNAPSHOT_md5.txt`).
- **Trainer.** `code/train_samsep_persona.py` is `train_samsep_vox.py` plus:
  - the persona module, and the `persona` cfg key allowed in the 26232 diff;
  - a report-only cfg diff vs 29345;
  - a startup plan report: realized share, persona x clip 2x2 table, and lengths. Any block that would not fit aborts the start;
  - `DUMP_start.txt`, written by rank 0 at start.
- **Slot.** The block goes in PPM's persona slot: `<|im_start|> system \n [B voice clip] <sop> PROSE <eop> <|im_end|> \n`.
  - `<sop>` / `<eop>` are ids 151680 / 151681. The tokenizer is Qwen3-32B.
  - The block sits on channel B only. Channel A carries nothing in the prefix.
  - PPM's own block text puts a label line before each section: `Persona`, `Starting context`, `Earlier conversation memory` (`persona_text_s3q.block_text`). Per the user, none of these labels is emitted here and no starting context is included: the block is the persona prose exactly as written.
- **No loss on the block.** No target, cb0 label or light frame sits in the prefix (asserted).
- **The conversation is never cropped.** A block is never truncated. Any block that would not fit `sp_cap` 2,000 makes the start abort.
- **Eval micros are unchanged:** there is no persona on eval, so the step-0 eval should equal 29345's 1.8484.

## Knobs (`run_persona.env`, final)

| knob | value | meaning |
|---|---|---|
| `ALOE_PERSONA_P` | 0.5 | Target share of ALL train items. An eligible item (B side has non-empty persona prose) draws at p_eff = 0.5 / eligible share. 0 = job 29345 exactly. |
| `ALOE_PERSONA_SIDE` | B | The trained side. A is optional. |
| `ALOE_PERSONA_SECTIONS` | persona | Prose sections, joined with no labels. |
| `ALOE_PERSONA_TEXT` | (empty) | Optional fixed string, or a file path, used instead of the sidecar prose. |
| `ALOE_PERSONA_SEED` | 20261011 | The draw is `h64('aloe-persona\|seed\|row_key\|swap') % 1e6 < p_eff x 1e6`. It is independent of the 30 % voice draw, which is unchanged. |
| `ALOE_PERSONA_EVAL` | 0 | Eval micros carry no persona. |

The hosts are slinky-6, 7, 13, 14, 15, 16, 17 and 19, at 8 x 8 (world 64). The init is `/scratch/amartyavarma/aloevox-init/w-17127` with `ALOE_INIT_VOX=1`.

New names, so nothing collides with 29345:
- run name: `QQA_RUN=ppm-aloe-samsep-vox17127-persona`
- control dir: `persona/control`
- export dirs: `persona/export-vox17127-persona-249` and `/scratch/.../exports-vox17127-persona`
- saves root: `saves-persona`
- private HF repos: `InternalCan/ppm-aloe-samsep-vox17127-persona` and `-persona-final`

## Launch (not run)

`bash /data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona/launch_persona.sh`

This runs `sbatch -N8 -w <hosts> --ntasks-per-node=8 --cpus-per-task=6 --mem=520G persona/run_persona.sbatch` with `QQA_ENV=persona/run_persona.env`.

The init must already be staged on all 8 hosts.

## Tests (CPU, slinky-18 / slinky-4; `tests/`)

- **p = 0 identity: PASS.** 4,992 digests compared, 0 mismatches.
  - Compared: every array and field of the step-0 train element of all 64 logical ranks (micro, the `singles` loss-check micro, the A waveform) and of each rank's first eval micro.
  - Both sides were built by `AccDS` / `eval_micros` as at world 64: job 29345's code path vs the persona module.
  - Plan digest d1a3e3721328da0a: 15,898 / 316 micros, 249 steps.
  - Identical micros imply the GPU packed-loss check would also match; it was not run, since it cannot run on CPU.
- **p = 0.5:**
  - The eval micros and the non-persona step-0 train micros are bit-identical to 29345's (3,561 digests, 0 mismatches).
  - 27 of 64 step-0 items carry a block. Each of these passes the shift check: it equals the no-persona micro shifted by the block, its prefix is header + clip + `<sop>..<eop>` + footer, and there are no labels or targets in the prefix.
- **Plan report** (all 15,898 train items):
  - Eligible: 13,039 = 0.8202. p_eff is 0.6096.
  - Drawn: 8,036 = 0.5055 of all items.
  - 2x2 table:

    | persona + clip draw | persona only | clip only | neither |
    |---|---|---|---|
    | 0.1474 | 0.3581 | 0.1461 | 0.3485 |

  - Block length: mean 159 tokens, median 116, max 452.
  - Overflow: 0. Special-token skips: 0.
  - Max row length T is 8,089 (no-persona bound 7,637; budget 9,502). 29345's peak memory is 45 / 80 GiB.
- **Dump.** `DUMP_p0.5.txt` holds 5 decoded rows. A case-sensitive grep for `Persona`, `Starting context` and `Earlier conversation memory` finds 0 hits; a lowercase grep for `persona` / `starting` also finds 0.
- **Dry config check.** The cfg diff vs 26232 gives VERDICT PASS. Against 29345 only dirs, hosts, the saves root and the `persona.*` keys differ (`CFG_DIFF_persona.txt`).
