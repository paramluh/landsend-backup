# Playground GPU pool: fixed minimums (10-26, aloedemo)

The user's order was a fixed minimum per role, which a reshuffle can never take to 0. Every change goes in an **empty-queue window**, make-before-break and drain-safe (`roll_flexpair.sh`).

| Role | Job (10-26 14:00 UTC) | Host:GPUs | Minimum | May lend |
|---|---|---|---|---|
| P: live duplex call backend | posedup-lb-P | slinky-18:2-3 | 1 | nothing (fixed; not ours to touch) |
| P's avatar + playground stream backend (hybrid) | genweb-avhybrid 30664 | slinky-4:2-3 | 1 | renders Param jobs while no stream is open (fixed) |
| Hot monoplex worker A | genweb-u-hot 30675 (`hot_unit.sh`, LANE_ROLE=none) | slinky-18:4-5 | 1 | nothing |
| Duplex "both sides" worker | genweb-dup 30677 (`unit.sbatch dup`, DUP_PREFER_MONO=1) | slinky-18:0-1 | 1 (on a duplex arrival) | is a hot **monoplex** worker while no duplex run waits; hands back after its current run when a duplex run arrives (duplex load ~100 s) |
| Hot monoplex worker B (guest) | srun overlap step in videort's job 29805 (`mono_guest.sh`, worker id mono29805g36, tmux `monoguest`) | slinky-10:3,6 (lent by videort 10-26 14:00) | 1 | nothing |
| Param render pair | genweb-u-param 30679 (`unit.sbatch paramonly`) | slinky-4:0-1 | 2 lanes | nothing |
| Vansh overflow (throttled GPUs) | genweb-u-vanshonly 30678 (`unit.sbatch vanshonly`) | slinky-4:6-7 | 0 | Vansh 20/50 renders only; never Param or generation |
| Live demo instance 2 | posedup-lb-B | slinky-18:6-7 | 1 | nothing (not playground) |
| MoT live (motlive) | motlive-gpu | slinky-4:4-5 | n/a | not ours |

## Generation slots

There are 3 monoplex slots at once: hot A, hot B (slinky-10 guest), plus the duplex pair while duplex is idle. Measured 14:01 UTC: a 3-sample batch started all 3 within 0.2 s on 3 workers. There is 1 duplex slot, the duplex pair.

A single-GPU mono worker is not possible:

- The worker holds 52.2 GB + 41.5 GB = 93.7 GB, measured with nvidia-smi on a hot worker. That is more than one 80 GB H100.
- The export is 83 GB on disk.
- Fitting it would need a different precision or placement, which is a maths change, so it was not tried.

## Co-location (measured 13:54–13:56 UTC, rejected)

The test put a hot mono worker and a Param render lane on the worker's 41 GB GPU.

| Measure | Solo | Co-located |
|---|---|---|
| Peak memory, worker's 41 GB GPU | ~41.5 GB | 76.8 GB (76759 MiB) |
| Peak memory, worker's 52 GB GPU | 52.2 GB | 52.2 GB |
| Generation speed | 58.7 ms/frame | 95.6 ms/frame (+63%) |
| 40 s render time | 73.9 s | 68.7 s |

There was no OOM. It failed the "generation within +5%" rule, so the fallback layout above is used.

## How the minimums are enforced

- **Hot A and the Param pair:** neither has a flex controller. Their worker or lanes only restart in their own loop.
- **Duplex pair:** it never turns into Vansh or Param lanes while it is a mono worker (mono comes first). It returns to duplex only after its current run, drained. Then it stays a duplex worker until no duplex run is queued and its worker is idle.
- **Renderer pairs** (`flex_lib.sh render_pair`, if one is ever run again): the last warm Param lane never leaves (the warm registry is in `flex/warm`). A pair converts to generation only while another warm Param lane exists. When Param jobs wait and no lane is warm, bringing one up comes first.
- **pool.sh:**
  - It backfills only free GPU pairs.
  - It cancels only its own `genweb-pool-*` backfills, and only after draining them: no run of theirs is queued or running and their GPUs are idle.
  - It never touches the fixed units above.
- **Every stop of a worker drains first.** A drain flag in `flex/drain/<wid>` means "claim nothing new". The stop then waits for the worker's runs to finish and only then kills it.
