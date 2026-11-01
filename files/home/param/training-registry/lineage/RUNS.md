# Run registry

As of 2026-10-27 20:14 PT. 161 training runs (+4 note rows), 22 deployment slots. Machine-readable copy: [runs.json](runs.json). Graph: [LINEAGE.md](LINEAGE.md).

How to read: headline eval = held-out **normalised word CE** (lower is better); pose runs also show pose cb0 / pose light CE. Every value is cited in the run's detail block below (file:line). `unknown` = could not be verified from files on disk. HF links marked *name match* were matched by repo name only (metadata listing), not by a file in the run dir.

## Contents

- [Early Qwen / Mistral stage runs (pre-PPM) and reference runs](#early-qwen--mistral-stage-runs-pre-ppm-and-reference-runs) (73 rows)
- [PPM 32B pretraining, 8,1,1 continuation, pose5, other pretraining](#ppm-32b-pretraining-811-continuation-pose5-other-pretraining) (12 rows)
- [14B family: INTL 14B (ppm-next), ppm14b, MoT, motvox, 14B FTs and ASR adapters](#14b-family-intl-14b-ppm-next-ppm14b-mot-motvox-14b-fts-and-asr-adapters) (9 rows)
- [32B speech FTs, PPMVOX, 32B ASR adapters, si20k / xfer](#32b-speech-fts-ppmvox-32b-asr-adapters-si20k--xfer) (38 rows)
- [32B pose FTs (Xabi v2pure, monoplex, pose-duplex, Aloe pose) and SAM 8B pose FTs](#32b-pose-fts-xabi-v2pure-monoplex-pose-duplex-aloe-pose-and-sam-8b-pose-fts) (33 rows)
- [Deployments](#deployments)
- [HF model repos not mapped to a run](#hf-model-repos-not-mapped-to-a-run)


## Early Qwen / Mistral stage runs (pre-PPM) and reference runs

| run | status | jobs | dates | init (parent node@step) | steps | headline evals | outputs |
|---|---|---|---|---|---|---|---|
| [`early-to-ppm-link`](#early-to-ppm-link) | finished (note) | 18873 | 2026-10-11 17:41 PT (first PPM job, fresh start) | Qwen/Qwen3-32B | 0 (fresh start; '[resume] auto ... none (fresh start)') | unknown | unknown |
| [`qwen-e25b`](#qwen-e25b) | unknown | unknown | unknown (before 2026-09-30) | ext:qwen-e4@unknown | unknown | unknown | unknown |
| [`qwen-e28`](#qwen-e28) | finished | 11140..11411 (prior cluster) | unknown (before 2026-09-30) | ext:qwen-e27@1350 | -> 4550 (dcp-4550) | KL 0.0126 lane-B word KL on klpre's 112 E28 text rows | j11411 dcp-4550; j11411-dcp-4550-bf16 / HF: qwen-duplex-e28, klpre-tts128m-ckpts |
| [`klpre-bpad2`](#klpre-bpad2) | finished | 21059 (prior cluster) | unknown (before 2026-09-30) | qwen-e28@4550 | -> 532 | KL 0.0126 @0 -> 0.0205 @530 (112 held-out E28 rows) | dcp-532 / HF: klpre-tts128m-ckpts |
| [`klpre-recov`](#klpre-recov) | unknown | 21228 (prior cluster) | unknown | ext:klpre-kl10@2363 | unknown -> 600 (recovery dcp-600 mentioned as SKIP) | unknown | dcp-600 |
| [`klpre-imend`](#klpre-imend) | finished | 2552, 2592, 2601, 2656, 2712 (+3) | 2026-10-01 03:16 .. 2026-10-01 12:31 PT | qwen-e28@4550 | 0 -> 1000 | word CE(norm) 2.7499 @1000 (speech normalized_word_b); GB none recorded; KL 0.01617 @1000 (kl_word_invariant); kl_im_end_b 0.01826 | dcp-1000; export-1000-hf / HF: klpre-tts128m-ckpts |
| [`qwen1ch-s1-plain-4165`](#qwen1ch-s1-plain-4165) | abandoned | 4139, 4142, 4165 | 2026-10-01 13:58 .. 2026-10-01 ~14:54 PT (4165 cancelled) | base:qwen3-32b | 0 -> ~666 (eval 650) | GB 60-game: plain w-606 45; KL 0.02712 @650 (0.02761 @600) | w-606 (HF note) / HF: qwen-1ch-s1-ckpts |
| [`qwen1ch-s1`](#qwen1ch-s1) | finished | 4332, 4335, 4362, 4365, 4866 | 2026-10-01 15:12 .. 2026-10-01 20:34 PT | base:qwen3-32b | 0 -> 4000 (horizon 4553) | GB w-4000: 15/20 (v3), 12/20 (v5); 60-game 46 (w-2000 47); KL 0.01237 @4000; 0.01422 @2000 | w-2000; dcp-2000; w-4000 / HF: qwen-1ch-s1-ckpts |
| [`qwen1ch-s1pt-plain`](#qwen1ch-s1pt-plain) | finished | 9682, 9684 | 2026-10-08 01:22 .. 2026-10-08 02:10 PT | base:qwen3-32b | 0 -> 700 | GB 42/80 (w-700); 120 games 74 vs plain p606 83; KL 0.02534 @700 (from 0.5584) | w-700; dcp-700 |
| [`qwen1ch-s2-dev`](#qwen1ch-s2-dev) | finished | 5528, 5542, 5578, 5596, 5615 (+17) | 2026-10-01 22:08 .. 2026-10-03 07:51 PT | qwen1ch-s1@4000 | 0 -> 2557 (max_steps 6597) | word CE(norm) 2.7158 @2500 (j7226); GB w-248 12/20, w-641 15/20 (v3); KL 0.01427 @2500 | dcp-248; w-500; w-641 / HF: qwen-1ch-s2-ckpts |
| [`qwen1ch-s2-promA`](#qwen1ch-s2-proma) | abandoned | 7530, 7558, 7560, 7566 | 2026-10-03 13:34 .. 2026-10-03 16:42 PT | base:qwen3-32b | 0 -> 500 (j7530), resume 250 -> 276 (j7566) | word CE(norm) 2.6543 @500; KL 1.66140 @500 (text intelligence collapses) | w-250; w-276 / HF: qwen-1ch-s2-ckpts |
| [`qwen1ch-s2-promB`](#qwen1ch-s2-promb) | abandoned | 7569, 7582, 7583 | 2026-10-03 16:42 .. 2026-10-03 19:46 PT | base:qwen3-32b | 7569: 0 -> 50; 7583: 0 -> 307 | word CE(norm) 2.8110 @300 (7583); 3.1052 @50 (7569); KL 0.00108 @300 (7583, chat-template KL) | w-307 (7583) |
| [`qwen1ch-s2-promB2`](#qwen1ch-s2-promb2) | finished | 7604 | 2026-10-03 19:46 .. 2026-10-03 22:18 PT | base:qwen3-32b | 0 -> 345 | word CE(norm) 2.7475 @300; KL 0.00277 @300 (chat view) | dcp-345; w-345 / HF: qwen-1ch-s2-ckpts |
| [`qwen1ch-s2-promB2m`](#qwen1ch-s2-promb2m) | finished | 7614 | 2026-10-03 22:19 .. 2026-10-04 01:02 PT | qwen1ch-s1@4000 | 0 -> 337 | word CE(norm) 2.7320 @300; KL 0.01626 @300 | dcp-337; w-337 / HF: qwen-1ch-s2-ckpts |
| [`samft-qwen-lane`](#samft-qwen-lane) | finished | 7616 | 2026-10-03..04 (exact unknown) | base:sam-51960 | 0 -> 1000 | unknown | step-1000 / HF: sam-graft-ckpts, sam-b700k-cd3-ad0-lead1-51960 |
| [`qwen1ch-s2-final`](#qwen1ch-s2-final) | finished | 7643, 7645, 7652, 7671, 7755 (+1) | 2026-10-04 01:34 .. 2026-10-05 12:44 PT | qwen1ch-s1@4000, samft-qwen-lane@1000 | 0 -> 4289 (max_steps 4396) | word CE(norm) 2.4239 @4250 (final); 2.4851 @3300; 2.5553 @2500; GB j7671 w-2500 19/20 (v5), 16/20 (v4); 60-game: 2500 42, 3000 46, 3301 41, 4000 33, 4250 40; K… | dcp-996; w-2250; dcp-2250 / HF: qwen-1ch-s2-final-ckpts |
| [`qwen1ch-s2-plainpt`](#qwen1ch-s2-plainpt) | finished | 9716, 9810, 9876, 9920, 9949 (+6) | 2026-10-08 02:29 .. 2026-10-09 05:21 PT | qwen1ch-s1pt-plain@700 | 0 -> 2348 (max_steps 4644) | word CE(norm) 2.6036 @2325 (final); 2.7242 @500; GB w-1150 48/80; LD1190 45/80 (v7); KL 0.08790 @2325; 0.04744 @500 | w-500; w-1190 (branch); w-2348 |
| [`qwen1ch-s2-plainpt-lowlr`](#qwen1ch-s2-plainpt-lowlr) | finished | 9722 | 2026-10-08 05:46 .. 2026-10-08 06:29 PT | qwen1ch-s2-plainpt@500 | 500 -> 600 | word CE(norm) 2.7117 @600; GB submitted (result not found); KL 0.03759 @600 (from 0.04744 @500) | w-600 |
| [`opd-proto`](#opd-proto) | abandoned | 6210, 6217, 6228, 6229, 6239 (+5) | 2026-10-02 15:03 .. 2026-10-02 20:28 PT | qwen1ch-s2-dev@641, qwen1ch-s2-dev@500 | A: -> ck-00024; A2: -> 100 | GB A2 @100: 25/40 (first 20: 12/20) | runs/A-w641-lr1e4/ck-00024.pt |
| [`qwen1ch-s3-exp1`](#qwen1ch-s3-exp1) | finished | 8176, 8185, 8197, 8246, 8324 (+2) | 2026-10-05 13:33 .. 2026-10-05 19:26 PT | qwen1ch-s2-final@4289 | 0 -> 750 | word CE(norm) B 2.1200 @750; KL 0.0264 @750 | KEEP_dcp-750; w-750 |
| [`qwen1ch-s3-lr03`](#qwen1ch-s3-lr03) | paused | 8424, 8724, 8796, 8815 | 2026-10-05 22:25 .. 2026-10-07 13:33 PT | qwen1ch-s3-exp1@750 | 750 -> 5900 (planned end 16,994) | word CE(norm) B 2.0018 @5900 (A 2.0138); 2.0498 @1500; KL 0.01213 @5900 | keep-lr0.3-dcp-1250; dcp-1500; w-5750 / HF: qwen-1ch-s3-exp1-lr0.3-from750-ckpts |
| [`qwen1ch-s3-exp1b`](#qwen1ch-s3-exp1b) | abandoned | 8347 | 2026-10-05 19:27 .. 20:09 PT | qwen1ch-s3-exp1@750 | 750 -> 850 | word CE(norm) B 2.1433 @850; KL 0.1329 @850 | unknown |
| [`qwen1ch-s3-exp1c`](#qwen1ch-s3-exp1c) | finished | 8354 | 2026-10-05 20:09 .. 20:58 PT | qwen1ch-s3-exp1@750 | 750 -> 850 | word CE(norm) B 2.0871 @850; KL 0.0205 @850 | unknown |
| [`qwen1ch-s3-exp1d`](#qwen1ch-s3-exp1d) | finished | 8403 | 2026-10-05 21:00 .. 22:24 PT | qwen1ch-s3-exp1@750 | 750 -> 950 | word CE(norm) B 2.1514 @950; KL 0.0681 @950 | unknown |
| [`qwen1ch-s3-exp1e`](#qwen1ch-s3-exp1e) | abandoned | 8495, 8574, 8582, 8597 | 2026-10-06 04:00 .. ~09:50 PT | qwen1ch-s3-lr03@1250 | 1250 -> 1350 (e) / 1425 (e2) / 1400 (e3) | word CE(norm) B e 2.0597 @1350; e2 2.0535 @1425; e3 2.0551 @1400; KL e 0.0165, e2 0.0161, e3 0.0162 | unknown |
| [`qwen1ch-s3-exp1f`](#qwen1ch-s3-exp1f) | abandoned | 8462 | 2026-10-06 03:03 .. 03:59 PT | qwen1ch-s3-lr03@1250 | 1250 -> 1400 | word CE(norm) B 2.0469 @1400; KL 0.0152 @1400 | unknown |
| [`qwen1ch-s3-exp1r`](#qwen1ch-s3-exp1r) | abandoned | 8590, 8595 | 2026-10-06 07:18 .. 07:44 PT | qwen1ch-s3-lr03@1250 | 1250 -> 1300 | word CE(norm) B 2.0561 @1300; KL 0.0165 @1300 | unknown |
| [`qwen1ch-s3-aloe-ft`](#qwen1ch-s3-aloe-ft) | finished | 9474, 9478, 9482, 9485, 9494 | 2026-10-07 23:40 .. 2026-10-08 00:56 PT | qwen1ch-s3-lr03@5900 | 0 -> 93 | word CE(norm) B 1.9529 @93 (own eval set) | final export (dir q32-s3-aloe-ft-2n-r1-nokl-from5900-j9494) |
| [`qwen1ch-s3-aloe-ft-8n`](#qwen1ch-s3-aloe-ft-8n) | finished | 21926 | 2026-10-16 02:16 .. 02:28 PT | qwen1ch-s3-lr03@5900 | 0 -> 23 | word CE(norm) B 1.9609 @23 | unknown |
| [`qwen1ch-s3-m1`](#qwen1ch-s3-m1) | finished | 9939, 9995, 9999 | 2026-10-08 19:06 .. 21:03 PT | qwen1ch-s3-lr03@5900 | 0 -> 365 (9999 resumed dcp-100) | unknown | unknown |
| [`qwen1ch-s3-m1fix`](#qwen1ch-s3-m1fix) | finished | 10036 | 2026-10-08 21:05 .. 22:46 PT | qwen1ch-s3-lr03@5900 | 0 -> 365 | word CE(norm) B 1.9257 @365 | unknown |
| [`qwen1ch-s3-tt-script`](#qwen1ch-s3-tt-script) | abandoned | 10773 | 2026-10-09 05:39 .. 07:13 PT | qwen1ch-s3-lr03@5900 | 0 -> 360 | word CE(norm) B 2.0298 @360 (tt eval set) | unknown |
| [`qwen1ch-s3-tt-scriptvoice`](#qwen1ch-s3-tt-scriptvoice) | finished | 10886 | 2026-10-09 07:17 .. 17:03 PT | qwen1ch-s3-lr03@5900 | 0 -> 2306 (final step per preflight; export w-2096) | word CE(norm) B 0.0849 @2040 (scripted rows: transcript given, not comparable) | w-2096 |
| [`qwen1ch-s3-tt3`](#qwen1ch-s3-tt3) | stopped-with-save | 11460, 11717 | 2026-10-09 19:41 .. 2026-10-10 00:38 PT | qwen1ch-s3-tt-scriptvoice@2096 | 0 -> 792 (11717 resumed dcp-480) | word CE(norm) B 0.1567 @720 (scripted) | w-792 / HF: tt-v3-base-w792 |
| [`qwen1ch-s3-ttaloe`](#qwen1ch-s3-ttaloe) | finished | 11938, 11953 | 2026-10-10 00:47 .. 01:40 PT | qwen1ch-s3-tt3@792 | 0 -> 145 | word CE(norm) B 0.2143 @145 (scripted) | unknown / HF: tt-v3-aloe-w145 |
| [`qwen1ch-s3-ttpw`](#qwen1ch-s3-ttpw) | finished | 20336 | 2026-10-13 18:50 .. 22:51 PT | qwen1ch-s3-tt3@792 | 0 -> 996 | word CE(norm) B 0.1573 @996 (scripted) | w-996 / HF: tt-s3-ttpw-j20336 |
| [`qwen1ch-s3-alopw`](#qwen1ch-s3-alopw) | finished | 20465, 20471, 20474, 20477, 20480 (+1) | 2026-10-14 00:10 .. 02:39 PT | qwen1ch-s3-ttpw@996 | alopw 0 -> 120; alopw8 0 -> 129 | word CE(norm) B alopw 0.2214 @120; alopw8 0.2189 @129 (scripted) | unknown |
| [`qwen1ch-s3-gen-usertext`](#qwen1ch-s3-gen-usertext) | finished | 14550, 14709, 15813 | 2026-10-10 22:55 .. 2026-10-11 03:29 PT | qwen1ch-s3-lr03@5900 | 0 -> 306 | word CE(norm) B 1.1672 @306 (no KL); 1.1644 @306 (with KL); KL 0.0142 @306 (-kl run) | unknown |
| [`qwen1ch-pose-s3`](#qwen1ch-pose-s3) | finished | 10854, 11354, 11379 | 2026-10-09 17:04 .. 19:05 PT | qwen1ch-s3-lr03@5900 | 0 -> 392 (11379 resumed dcp-150) | word CE(norm) conv 0.0923 @300 (scripted tt rows); pose cb0 2.9222 @300; pose light 7.5741 @300; KL 0.01517 @300 | w-392 |
| [`qwen1ch-pose-s3pv`](#qwen1ch-pose-s3pv) | finished | 11421, 12207 | 2026-10-09 19:05 .. 2026-10-10 04:12 PT | qwen1ch-s3-lr03@5900 | 0 -> 470 (12207 resumed dcp-102) | word CE(norm) conv 2.0299 @450; pose cb0 2.8045 @450; pose light 7.1536 @450; KL 0.01510 @450 | w-470 |
| [`qwen1ch-pose-s3sam`](#qwen1ch-pose-s3sam) | abandoned | 12433, 12486, 12530, 12546, 12547 | 2026-10-10 04:53 .. 07:01 PT | qwen1ch-s3-lr03@5900 | s3sam 0 -> 27; s3sam311 0 -> 50 | word CE(norm) conv 2.0447 @25; pose cb0 7.6229 @25; pose light 7.2205 @25; KL 0.01223 @25 | w-27 |
| [`qwen1ch-pose-s3sam311b`](#qwen1ch-pose-s3sam311b) | stopped-with-save | 12582, 12822, 12990 | 2026-10-10 07:01 .. 18:06 PT | qwen1ch-s3-lr03@5900 | 0 -> 696 (planned 34,422) | word CE(norm) conv 2.0111 @650 (B 2.0156); pose cb0 2.1625 @650; pose light 5.6205 @650; KL 0.01228 @650 | w-696; dcp-696 |
| [`qwen1ch-s4-adapter-asr`](#qwen1ch-s4-adapter-asr) | finished | 9231, 9235, 9239, 9247, 9310 (+1) | 2026-10-07 13:35 .. 18:29 PT | qwen1ch-s3-lr03@5900, base:voxtral-mini-4b-realtime-2602 | -> 935 | unknown | w-935 |
| [`qwen1ch-s4-adapter-kl`](#qwen1ch-s4-adapter-kl) | finished | 9233, 9237, 9241, 9244, 9249 (+2) | 2026-10-07 13:46 .. 18:29 PT | qwen1ch-s3-lr03@5900, base:voxtral-mini-4b-realtime-2602 | -> 207 (j9249), -> 526 (j9347) | unknown | w-207; w-526 |
| [`qwen1ch-s4-val`](#qwen1ch-s4-val) | finished | 9267, 9401, 9406, 9409, 9411 | 2026-10-07 15:22 .. 21:08 PT | qwen1ch-s3-lr03@5900, qwen1ch-s4-adapter-kl@207 | 5900 -> 6339 | word CE(norm) B 2.0172 @6325; 2.0214 @6150; KL 0.0140 (cited in s4-qqa STATUS) | w-6339; dcp-6339 |
| [`qwen1ch-s4-val-kl10`](#qwen1ch-s4-val-kl10) | finished | 9414, 9453 | 2026-10-07 21:09 .. 23:08 PT | qwen1ch-s3-lr03@5900, qwen1ch-s4-adapter-kl@207 | kl10 -> 6250; bs4x -> 6204 | word CE(norm) kl10 B 2.0205 @6250; bs4x B 2.0074 @6204 | unknown |
| [`qwen1ch-s4-ft-hovfix`](#qwen1ch-s4-ft-hovfix) | finished | 9977, 10006, 10139, 10242 | 2026-10-08 19:48 .. 23:52 PT | qwen1ch-s4-val@6339 | 0 -> 731 | word CE(norm) B 1.8969 @731 (own eval); hov 1.9559 @125; KL 0.0204 at its w-731 (s4-qqa step-0 eval) | w-731 |
| [`qwen1ch-s4-ft-gen`](#qwen1ch-s4-ft-gen) | finished | 13783, 13858 | 2026-10-10 20:35 .. 21:43 PT | qwen1ch-s4-ft-hovfix@731 | 0 -> 306 | word CE(norm) B 1.2864 @306 | w-306 |
| [`qqa-run-a`](#qqa-run-a) | finished | 21103 | 2026-10-14 22:42 .. 23:30 PT | qwen1ch-s4-ft-hovfix@731 | 0 -> 223 | word CE(norm) B 1.5865 @223 (96 held-out quick-qa conversations) | w-223 / HF: q32-s4-ft-qqa-j21103 |
| [`qqa-run-b`](#qqa-run-b) | abandoned | 21205 | 2026-10-15 01:37 .. (cancelled) PT | qwen1ch-s4-ft-hovfix@731, qwen1ch-s4-asr2-ga1@3695 | 0 -> 75 | word CE(norm) B 1.7595 @75 | unknown |
| [`qqa-run-c`](#qqa-run-c) | finished | 21169 | 2026-10-15 00:55 .. 01:35 PT | qwen1ch-s4-ft-hovfix@731 | 0 -> 223 | word CE(norm) B 1.3687 @223 (2.2205 @0); KL 0.0251 at w-223 (forgetting check) | w-223 |
| [`qqa-run-d`](#qqa-run-d) | finished | 21276 | 2026-10-15 03:59 .. 04:31 PT | qwen1ch-s4-ft-hovfix@731 | 0 -> 223 | word CE(norm) B 1.3634 @223 (5.9937 @0) | w-223 |
| [`qqa-run-e`](#qqa-run-e) | finished | 21302 | 2026-10-15 05:10 .. 05:49 PT | qwen1ch-s4-ft-hovfix@731 | 0 -> 223 | word CE(norm) B 1.5366 @223 (2.4235 @0) | w-223 |
| [`qqa-run-f`](#qqa-run-f) | finished | 21455 | 2026-10-15 09:51 .. 10:31 PT | qwen1ch-s4-ft-hovfix@731, qwen1ch-s4-asr2-ga1@3695 | 0 -> 223 | word CE(norm) B 1.4770 @223 (2.8002 @0); KL 0.0255 at w-223 | w-223 |
| [`qqa-run-g`](#qqa-run-g) | finished | 21443 | 2026-10-15 08:55 .. 09:48 PT | qwen1ch-s4-ft-hovfix@731 | 0 -> 223 | word CE(norm) B 1.3729 @223; KL 0.0204 -> 0.0135 @223 | w-223 |
| [`qqa-run-h`](#qqa-run-h) | finished | 21493 | 2026-10-15 11:06 .. 11:58 PT | qwen1ch-s4-ft-hovfix@731, qwen1ch-s4-asr2-ga1@3695 | 0 -> 223 | word CE(norm) B 1.4908 @223; KL 0.0204 -> 0.0136 | w-223 |
| [`qqa-scaling`](#qqa-scaling) | finished | 21335, 21373, 21504, 21734 | 2026-10-15 05:52 .. 20:56 PT | qwen1ch-s4-ft-hovfix@731 | 0 -> 223 each | word CE(norm) B @223: noamimi 1.5573, s3text 1.3633, s3textnc 1.3655, asrinit 1.4793 | w-14; w-28; w-59 |
| [`qwen1ch-s4-asr2-ga1`](#qwen1ch-s4-asr2-ga1) | finished | 19161, 19163, 19169, 19174, 19253 | 2026-10-12 06:41 .. 16:31 PT | qwen1ch-s3-lr03@5900, base:voxtral-mini-4b-realtime-2602 | 0 -> 3696 | KL B-KL vs teacher 0.20057 @3695 | w-3600; w-3695/w-3696 |
| [`qwen1ch-s4-asr2-gain`](#qwen1ch-s4-asr2-gain) | finished | 19379, 19399, 19404 | 2026-10-12 17:31 .. 18:50 PT | qwen1ch-s4-asr2-ga1@3695 | -> 3995 (nonorm) | KL B-KL 5.29895 @3995 (nonorm) | w-3995 |
| [`qwen1ch-s4-adonly`](#qwen1ch-s4-adonly) | abandoned | 19262, 19263, 19267, 19287, 19277 (+3) | 2026-10-12 16:38 .. 2026-10-13 02:36 PT | qwen1ch-s3-lr03@5900, qwen1ch-s4-asr2-ga1@3695 | adonly 5900 -> 5945; noamimi 5900 -> 6000; full killed at ~5900 | unknown | w-5945 (adonly); w-6000 (noamimi) |
| [`hera2-ft`](#hera2-ft) | finished | 10219, 10252, 10289, 10306, 10471 (+4) | 2026-10-08 23:09 .. 2026-10-10 20:16 PT (launch times) | ext:hera2@6974 | sys75 -> 176; speech8 -> 365; vp8 -> 365 | unknown | export-hf/hera2-6974-ft-hov-j10306-step176-bf16; export-hf/hera2-6974-ft-hov-speech8-j10492-step365-bf16; export-hf/hera2-6974-ft-hov-vp8-j… |
| [`mistral-s1-2lane`](#mistral-s1-2lane) | abandoned | 46, 60, 77, 91, 105 (+15) | 2026-09-30 12:34 .. 2026-10-01 05:13 PT | base:mistral-medium-3.5-128b | j519 -> 500 (eval 490); j111 -> 393 | KL j519 0.04085 @490; j111 0.07303 @392 | j519 w-500; j111 w-393 / HF: mistral-1ch-s1-ckpts |
| [`mistral-t19-handoff`](#mistral-t19-handoff) | unknown | 21222, 21278, 21329, 21338 (prior cluster) | before 2026-09-30 (prior cluster) | base:mistral-medium-3.5-128b | 21278 -> 100; 21338 -> 45; 21329 (fp8) -> 58 | unknown | w-100; dcp-100; dcp-45 |
| [`mistral-s1-1ch`](#mistral-s1-1ch) | stopped-with-save | 709 | 2026-09-30 22:21 .. 2026-10-01 08:07 PT (cancelled after sa… | base:mistral-medium-3.5-128b | 0 -> 531 (horizon 6865) | GB 20/20, median 11 questions (v4 oracle); KL 0.04095 @490 | w-531; dcp-531 / HF: mistral-1ch-s1-ckpts |
| [`mistral-s2-1ch-early`](#mistral-s2-1ch-early) | finished | 2835, 2933, 2959 | 2026-10-01 08:44 .. 10:17 PT | mistral-s1-1ch@531 | 0 -> 100 | KL 0.03740 @100 | w-100 |
| [`mistral-s2-kl20`](#mistral-s2-kl20) | abandoned | 7391, 7393, 7406, 7421, 7425 (+1) | 2026-10-03 02:57 .. 05:26 PT | mistral-s1-1ch@531, base:ministral-3-8b-base-2512 | 0 -> 101 | word CE(norm) 2.7342 @100; KL 0.09480 @100 | w-101 |
| [`mistral-s2-lr5e6`](#mistral-s2-lr5e6) | finished | 7454 | 2026-10-03 05:36 .. 08:30 PT | mistral-s1-1ch@531 | 0 -> 271 | word CE(norm) 2.6924 @250; GB w-250 19/20, median 11 (v5); KL 0.07509 @250 | w-271 |
| [`mistral-s2-baseline`](#mistral-s2-baseline) | stopped-with-save | 7466, 7489 | 2026-10-03 08:31 .. 13:26 PT | mistral-s1-1ch@531, base:ministral-3-8b-base-2512 | 0 -> 427 | word CE(norm) 2.6731 @400; 2.6976 @250; KL 0.06941 @400 | w-250; dcp-250; w-427 / HF: mistral-1ch-s2-ckpts |
| [`samft-mistral-lane`](#samft-mistral-lane) | finished | 7603 | ~2026-10-03 (unknown) | base:sam-51960 | 0 -> 1000 | unknown | step-1000 / HF: sam-graft-ckpts |
| [`prometheus`](#prometheus) | finished | unknown | checkpoints 2026-08-05 (step 2001) and 2026-08-09 (step 260… | base:qwen3-32b | -> 26013 | unknown | checkpoint-2001-2026-08-05; checkpoint-26013-2026-08-09 |
| [`demeter-hestia`](#demeter-hestia) | finished | unknown | unknown | prometheus@unknown | -> 20505 | unknown | hestia-ctx-19500 / HF: demeter-32b-ckpts |
| [`demeter-causalsam`](#demeter-causalsam) | unknown | unknown | downloaded 2026-10-13 11:25 PT | demeter-hestia@unknown | -> 3960 | unknown | checkpoint-3960 / HF: demeter-32b-causalsam |
| [`gemma-not-trained`](#gemma-not-trained) | unknown (note) | unknown | 2026-10-11 (txtdup_*.sbatch written 15:34-16:12 PT) | unknown | n/a | unknown | unknown / HF: gemma-synth-2m-wlts-high-overlap, gemma-synth-r2-wlts |


### early-to-ppm-link

**PPM 32B init link (answer: base Qwen3-32B, not a stage run)**  
group `ppm32b` · status **finished** · dir `/data/home/amartyavarma/ppm/code`

- **Jobs:** 18873
- **Dates:** 2026-10-11 17:41 PT (first PPM job, fresh start)
- **Base model:** Qwen/Qwen3-32B
- **Init:** /data/home/amartyavarma/models/Qwen3-32B (stock safetensors, 708 tensors); towers and decoders random init; no stage-1/2/3/4, Prometheus or E28 weights
- **Parents (resolved):** Qwen/Qwen3-32B
- **Produces:** none recorded
- **Data:** n/a (link row)
- **Recipe:** model_ppm.py QWEN32 constant + build/init; prod_ppm.env has no init knob (PPM_RESUME_FROM=auto only)
- **Steps:** 0 (fresh start; '[resume] auto ... none (fresh start)')
- **Evals:** none recorded
- **Checkpoints / exports:** unknown
- **Local paths:** /data/home/amartyavarma/models/Qwen3-32B
- **HF repos:** none named in run files
- **Notes:** PPM 32B (ppm-v1) initialised from BASE Qwen3-32B, not from any stage-1/2/3/4 Qwen run, Prometheus or E28. memory:pose-podcast-model-plan.md:13 says 'start from our 32B backbone', but line 19 and the job-18873 init log settle it as stock Qwen3-32B. The stage lineages below are therefore separate from PPM: no checkpoint edge from them into ppm32b. Only the new control-token input rows (SOA/EOA/SOB, NW) reuse the data_1ch.INIT_FROM rule from the stage-1 code (model_ppm.py:12).
- **Citations:**
  - `/data/home/amartyavarma/ppm/logs/ppm-18873.log:4 -> '[init] backbone <- /data/home/amartyavarma/models/Qwen3-32B: 708 tensors (emb copied to A and B, bit-equal); 1440 random tensors (seed 20261011)'`
  - `/data/home/amartyavarma/ppm/logs/ppm-18873.log:7 -> resume auto: none (fresh start)`
  - `/data/home/amartyavarma/ppm/code/model_ppm.py:3 -> 'BACKBONE base Qwen3-32B (models/Qwen3-32B), PLAIN RoPE ... NO adapters / LoRA'`
  - `/data/home/amartyavarma/ppm/code/model_ppm.py:18 -> towers Qwen3-4B-architecture, random init`
  - `/data/home/amartyavarma/ppm/code/model_ppm.py:47 -> 'INIT backbone tensors <- Qwen3-32B safetensors'`
  - `/data/home/amartyavarma/ppm/code/model_ppm.py:70 -> QWEN32 = /data/home/amartyavarma/models/Qwen3-32B`
  - `/data/home/amartyavarma/ppm-repo/README.md:54 -> Backbone = Qwen3-32B (models/Qwen3-32B), plain RoPE, no mix-RoPE`
  - `memory:pose-podcast-model-plan.md:19 -> 'Both 4B and 0.6B random init. Base Qwen3-32B backbone.'`
  - `memory:pose-podcast-model-plan.md:16 -> NO LoRA and NO mix-RoPE for this model`

### qwen-e25b

**E25b (mix-RoPE) - prior-cluster Qwen two-lane experiment**  
group `base-stage` · status **unknown** · dir `unknown (prior cluster; code referenced in /data/home/amartyavarma/qwen-s2-imend/qwendup)`

- **Jobs:** unknown
- **Dates:** unknown (before 2026-09-30)
- **Base model:** unknown
- **Init:** E4 (j10773, prior-cluster job id) per lineage line
- **Parents (resolved):** ext:qwen-e4@unknown
- **Produces:** qwen-e25b@unknown
- **Data:** unknown
- **Recipe:** introduced mix-RoPE: per-head router g = sigmoid(xW+b) mixing word-position and frame-position RoPE
- **Steps:** unknown
- **Evals:** none recorded
- **Checkpoints / exports:** unknown
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Notes:** Inherited history from a handoff repo; job ids are from a previous cluster (they collide with slinky job numbers). No local run dir or logs found. The original 'stage1..4' runs before E4 are also unknown.
- **Citations:**
  - `/data/home/amartyavarma/qwen-s2-imend/HANDOFF.md:15 -> 'Qwen3-32B -> stage1..4 -> E4 (j10773) -> E25b (mix-RoPE) -> E27 (dcp-1350)'`
  - `memory:qwen-mixrope-mistral-stock.md:11 -> E25b per-head router mixing word/frame RoPE`

### qwen-e28

**E28 (two-lane text-KL duplex trunk, mix-RoPE + L_PAD)**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-s2-imend/qwendup/stage4/run-mixrope-padce-14n-fast (code copy only)`

- **Jobs:** 11140..11411 (prior cluster)
- **Dates:** unknown (before 2026-09-30)
- **Base model:** unknown
- **Init:** E27 dcp-1350, weights-only, + L_PAD
- **Parents (resolved):** ext:qwen-e27@1350
- **Produces:** qwen-e28@4550
- **Data:** text-KL distillation of frozen stock Qwen3-32B in canonical ChatML, assistant tokens only (corpus15 klpre rows)
- **Recipe:** two-lane duplex student; mix-RoPE router per layer/head; L_PAD BCE alpha 0.1, pad_head lr 1e-4 (folded into lm_head[151670] at export)
- **Steps:** -> 4550 (dcp-4550)
- **Evals:** kl = 0.0126 lane-B word KL on klpre's 112 E28 text rows
- **Checkpoints / exports:** j11411 dcp-4550; j11411-dcp-4550-bf16
- **Local paths:** /scratch/amartyavarma/e28-export/j11411-dcp-4550-fp32 (fp32 export used by klpre-imend, node-local)
- **HF repos:** [InternalCan/qwen-duplex-e28](https://huggingface.co/InternalCan/qwen-duplex-e28), [InternalCan/klpre-tts128m-ckpts](https://huggingface.co/InternalCan/klpre-tts128m-ckpts)
- **Notes:** Prior-cluster run (handoff); no local logs. The user abandoned the E28/klpre stack on 10-01 for the clean 1ch redo (qwen1ch-s1). PPM does NOT descend from E28.
- **Citations:**
  - `/data/home/amartyavarma/qwen-s2-imend/HANDOFF.md:8 -> Stage 1 = E28, text-KL distillation, mix-RoPE + L_PAD`
  - `/data/home/amartyavarma/qwen-s2-imend/HANDOFF.md:16 -> 'E28 = E27 weights-only + L_PAD: j11140..11411 -> j11411 dcp-4550'`
  - `/data/home/amartyavarma/qwen-s2-imend/HANDOFF.md:26 -> InternalCan/qwen-duplex-e28 holds j11411-dcp-4550-bf16 and -dcp`
  - `/data/home/amartyavarma/qwen-s2-imend/HANDOFF.md:34 -> E28 scores 0.0126 on the klpre text rows`
  - `/data/home/amartyavarma/qwen-s2-imend/hera3s2/klpre/code_imend/logs/s2-train-3020.log:3 -> context_export /scratch/amartyavarma/e28-export/j11411-dcp-4550-fp32`
  - `memory:qwen-clean-1ch-plan.md:18 -> user: E28/klpre stack 'botched in pieces'; redo as clean 1ch`

### klpre-bpad2

**klpre-bpad2-tts128m (E28 + speech KL pretrain, stage 2 two-lane)**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-s2-imend/hera3s2/klpre/code_bpad (code copy)`

- **Jobs:** 21059 (prior cluster)
- **Dates:** unknown (before 2026-09-30)
- **Base model:** unknown
- **Init:** E28 fp32 master j11411 dcp-4550
- **Parents (resolved):** qwen-e28@4550
- **Produces:** klpre-bpad2@532
- **Data:** tts128m speech + E28 text-KL micro (W_kl 10 + L_PAD)
- **Recipe:** 32B context tower + 8B voice + 0.6B light; Mimi 32 cb, cd 3, 7,500-frame rows; text CE/cb0/light 5:1:1
- **Steps:** -> 532
- **Evals:** kl = 0.0126 @0 -> 0.0205 @530 (112 held-out E28 rows)
- **Checkpoints / exports:** dcp-532
- **Local paths:** unknown
- **HF repos:** [InternalCan/klpre-tts128m-ckpts](https://huggingface.co/InternalCan/klpre-tts128m-ckpts)
- **Notes:** Prior-cluster run; normalised word CE not recorded in the handoff. Variant code_recov j21228 is a separate row.
- **Citations:**
  - `/data/home/amartyavarma/qwen-s2-imend/README.md:9 -> klpre on the E28 trunk, init j11411 dcp-4550`
  - `/data/home/amartyavarma/qwen-s2-imend/HANDOFF.md:19 -> code_bpad j21059 klpre-bpad2-tts128m -> dcp-532`
  - `/data/home/amartyavarma/qwen-s2-imend/HANDOFF.md:27 -> HF InternalCan/klpre-tts128m-ckpts folder klpre-bpad2-tts128m-j21059/dcp-532`
  - `/data/home/amartyavarma/qwen-s2-imend/HANDOFF.md:34 -> bpad2 text KL 0.0126 at step 0 -> 0.0205 at step 530`

### klpre-recov

**klpre code_recov (T10 text-only KL recovery)**  
group `base-stage` · status **unknown** · dir `/data/home/amartyavarma/qwen-s2-imend/hera3s2/klpre/code_recov (code copy)`

- **Jobs:** 21228 (prior cluster)
- **Dates:** unknown
- **Base model:** unknown
- **Init:** KL-10 dcp-2363 (weights only)
- **Parents (resolved):** ext:klpre-kl10@2363
- **Produces:** klpre-recov@600
- **Data:** text-only KL, continuing the data cursor
- **Recipe:** trunk finetune of the context group only (configs/klrecov.yaml, recov.init_from)
- **Steps:** unknown -> 600 (recovery dcp-600 mentioned as SKIP)
- **Evals:** none recorded
- **Checkpoints / exports:** dcp-600
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Notes:** Prior-cluster; user marked its checkpoints SKIP. The KL-10 parent run id is a guess (klpre-kl10).
- **Citations:**
  - `/data/home/amartyavarma/qwen-s2-imend/HANDOFF.md:20 -> variant code_recov j21228 (text-only KL recovery from KL-10)`
  - `/data/home/amartyavarma/qwen-s2-imend/HANDOFF.md:29 -> recovery dcp-600 and KL-10 dcp-2363 are SKIP per user`
  - `/data/home/amartyavarma/qwen-s2-imend/HANDOFF.md:61 -> T10 recovery: text-only trunk finetune from KL-10 dcp-2363`

### klpre-imend

**klpre-imend-kl10 (im_end text-KL variant), ends j3020 dcp-1000 ('klpre step-1000')**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-s2-imend/hera3s2/klpre/code_imend`

- **Jobs:** 2552, 2592, 2601, 2656, 2712, 2754, 2787, 3020
- **Dates:** 2026-10-01 03:16 .. 2026-10-01 12:31 PT
- **Base model:** unknown
- **Init:** E28 j11411 dcp-4550 fp32 export (context tower); Qwen3-8B voice, Qwen3-0.6B light
- **Parents (resolved):** qwen-e28@4550
- **Produces:** klpre-imend@1000
- **Data:** tts-pretrain-128m speech + text-KL (W_KL 10, text_kl.im_end true)
- **Recipe:** code_imend train_s2.py, configs/klpre.yaml; lineage lr1e-5 j2592 (dcp-200) -> lr5e-6 ext j2787 (dcp-530) -> ext2 j3020 (-> 1000)
- **Steps:** 0 -> 1000
- **Evals:** norm_word_ce = 2.7499 @1000 (speech normalized_word_b); genghisbench = none recorded; kl = 0.01617 @1000 (kl_word_invariant); kl_im_end_b 0.01826
- **Checkpoints / exports:** dcp-1000; export-1000-hf
- **Local paths:** /scratch/amartyavarma/klpre-imend-sweep/ (node-local, per README)
- **HF repos:** [InternalCan/klpre-tts128m-ckpts](https://huggingface.co/InternalCan/klpre-tts128m-ckpts)
- **Notes:** Last two-lane E28-lineage run, on this cluster. After it the user switched to the clean single-channel recipe (qwen1ch-*). Exact per-job step ranges for 2552/2601/2656/2712/2754 not checked (smokes/sweeps).
- **Citations:**
  - `/data/home/amartyavarma/qwen-s2-imend/hera3s2/klpre/code_imend/logs/s2-train-3020.log:18220 -> EVAL step 1000 speech normalized_word_b 2.7499`
  - `/data/home/amartyavarma/qwen-s2-imend/hera3s2/klpre/code_imend/logs/s2-train-3020.log:18219 -> EVAL step 1000 kl_word_invariant 0.01617, kl_im_end_b 0.01826`
  - `/data/home/amartyavarma/qwen-s2-imend/hera3s2/klpre/code_imend/logs/s2-train-2592.log:1 -> start 2026-10-01 03:16:43`
  - `/data/home/amartyavarma/hf-backup-staging/readme/klpre-tts128m-ckpts/README_j3020.md:3 -> trunk = E28 j11411-dcp-4550; Qwen3-8B voice; W_KL 10, im_end`
  - `/data/home/amartyavarma/hf-backup-staging/readme/klpre-tts128m-ckpts/README_j3020.md:5 -> lineage j2592 (dcp-200) -> j2787 (dcp-530) -> j3020 (to 1000)`
  - `/data/home/amartyavarma/hf-backup-staging/readme/klpre-tts128m-ckpts/README_j3020.md:13 -> 'No GenghisBench / probe result was found for j3020'`
  - `memory:qwen-clean-1ch-plan.md:11 -> user tested inference on the klpre step-1000 checkpoint, then asked for the clean 1ch redo`

### qwen1ch-s1-plain-4165

**q32-1ch-s0 plain-RoPE stage 1 (killed for missing mix-RoPE)**  
group `base-stage` · status **abandoned** · dir `/data/home/amartyavarma/qwen-1ch/train`

- **Jobs:** 4139, 4142, 4165
- **Dates:** 2026-10-01 13:58 .. 2026-10-01 ~14:54 PT (4165 cancelled)
- **Base model:** Qwen/Qwen3-32B
- **Init:** stock Qwen3-32B; SOA/EOA/SOB rows = mean of data_1ch.INIT_FROM rows
- **Parents (resolved):** base:qwen3-32b
- **Produces:** none recorded
- **Data:** corpus15 split-0 text-KL, teacher cache 524396be8b724954 (canonical teacher view)
- **Recipe:** qwen-1ch/train train_q.py (port of Mistral train1ch), plain RoPE, LR 2e-6, warmup 100, cosine to 0.1x over 4553 steps, 288 rows/step, 18 nodes / 144 GPUs
- **Steps:** 0 -> ~666 (eval 650)
- **Evals:** genghisbench = 60-game: plain w-606 45; kl = 0.02712 @650 (0.02761 @600)
- **Checkpoints / exports:** w-606 (HF note)
- **Local paths:** /data/home/amartyavarma/qwen-1ch/train/ckpt-weka/q32-1ch-s0-lr2e-6-r288-w144-j4165
- **HF repos:** [InternalCan/qwen-1ch-s1-ckpts](https://huggingface.co/InternalCan/qwen-1ch-s1-ckpts)
- **Notes:** 4139 = TC_CHECK parity run (crashed in teacher-free check), 4142 = first production launch (killed after step-0 eval), 4165 = relaunch killed by user.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/train/READY.md:24 -> Init: stock weights, SOA/EOA/SOB = mean of INIT_FROM rows`
  - `/data/home/amartyavarma/qwen-1ch/train/READY.md:25 -> LR peak 2e-6, warmup 100, cosine to 0.1x over 4553 steps, 288 rows/step`
  - `/data/home/amartyavarma/qwen-1ch/train/logs/q1ch-4142.log:19 -> EVAL step 0 kl_word_invariant 0.53802`
  - `/data/home/amartyavarma/qwen-1ch/train/logs/q1ch-4165.log:1237 -> EVAL step 600 kl 0.02761`
  - `/data/home/amartyavarma/qwen-1ch/train/logs/q1ch-4165.log:1339 -> EVAL step 650 kl 0.02712`
  - `memory:qwen-mixrope-mistral-stock.md:13 -> q1ch-s0 job 4165 killed at ~666 because Qwen must use mix-RoPE`
  - `/data/home/amartyavarma/qwen-1ch/HANDOVER_night_1008.md:36 -> GenghisBench 60 games: plain w-606 45`
  - `/data/home/amartyavarma/hf-backup-staging/readme/qwen-1ch-s1-ckpts/README.md:12 -> j4165 earlier plain-RoPE run (stopped at 606)`

### qwen1ch-s1

**Qwen clean 1ch stage 1, mix-RoPE asst (q1ch-s0-mixrope-asst j4365 -> j4866)**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/train`

- **Jobs:** 4332, 4335, 4362, 4365, 4866
- **Dates:** 2026-10-01 15:12 .. 2026-10-01 20:34 PT
- **Base model:** Qwen/Qwen3-32B
- **Init:** stock Qwen3-32B (+ mix-RoPE routers, b init -4.0)
- **Parents (resolved):** base:qwen3-32b
- **Produces:** qwen1ch-s1@2000, qwen1ch-s1@4000
- **Data:** corpus15 split-0 text-KL vs frozen Qwen3-32B (teacher cache 524396be8b724954, canonical think view), Mistral 709 single-channel format
- **Recipe:** qwen-1ch/train train_q.py, launch_q.sh; mix-RoPE=1; LR 2e-6 (spec_out 1e-4, router W 1e-3, b 1e-2), warmup 100, cosine over horizon 4553, 288 rows/step, 18 nodes; part 1 = 0-2000 (j4365), part 2 = full resume 2000-4000 (j4866); user ended stage 1 at 4000
- **Steps:** 0 -> 4000 (horizon 4553)
- **Evals:** genghisbench = w-4000: 15/20 (v3), 12/20 (v5); 60-game 46 (w-2000 47); kl = 0.01237 @4000; 0.01422 @2000
- **Checkpoints / exports:** w-2000; dcp-2000; w-4000; dcp-4000
- **Local paths:** /data/home/amartyavarma/qwen-1ch/train/ckpt-weka/q1ch-s0-mixrope-asst-j4866/w-4000; /data/home/amartyavarma/qwen-1ch/stage1-final/q1ch-s0-mixrope-asst-j4866-w-4000; /data/home/amartyavarma/qwen-1ch/train/ckpt-weka/q1ch-s0-mixrope-asst-j4365/w-2000
- **HF repos:** [InternalCan/qwen-1ch-s1-ckpts](https://huggingface.co/InternalCan/qwen-1ch-s1-ckpts)
- **Notes:** 4332/4335/4362 = mix-RoPE smokes / first attempt (runs q1ch-s0-mixrope-*); not checked individually. Stage-1 is text-only (no speech, no NEW_WORD).
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/train/logs/q1ch-4365.log:5 -> mix-RoPE optimizer groups, lr 2e-06, router b init -4.0`
  - `/data/home/amartyavarma/qwen-1ch/train/runs/cfg-4365.json:27 -> max_steps 2000 (part 1); :28 horizon 4553; :36 lr 2e-06; :109 mixrope 1`
  - `/data/home/amartyavarma/qwen-1ch/train/logs/q1ch-4365.log:4080 -> EVAL step 2000 kl_word_invariant 0.01422`
  - `/data/home/amartyavarma/qwen-1ch/train/logs/q1ch-4866.log:6 -> full resume from j4365 dcp-2000`
  - `/data/home/amartyavarma/qwen-1ch/train/logs/q1ch-4866.log:4050 -> EVAL step 4000 kl_word_invariant 0.01237`
  - `/data/home/amartyavarma/qwen-1ch/train/logs/q1ch-4866.log:4051 -> saved dcp-4000 + Weka export w-4000`
  - `memory:qwen-clean-1ch-plan.md:14 -> init plain Qwen3-32B, data corpus15 split-0`
  - `memory:qwen-clean-1ch-plan.md:21 -> Stage 1 ENDS at step 4000`
  - `memory:genghisbench.md:14 -> v3: stage-1 end w-4000 15/20`
  - `memory:genghisbench.md:21 -> v5: stage-1 w-4000 12/20`
  - `/data/home/amartyavarma/qwen-1ch/HANDOVER_night_1008.md:36 -> 60 games: mix w-2000 47, mix w-4000 46`
  - `/data/home/amartyavarma/hf-backup-staging/readme/qwen-1ch-s1-ckpts/README.md:11 -> j4866-w-4000 = stage-1 end and INIT_FROM of every mix-RoPE stage-2 run`
  - `memory:qwen-kl-teacher-think-template.md:11 -> teacher render 'canonical' (think block only before last assistant turn)`

### qwen1ch-s1pt-plain

**q32-1ch-s1pt-plain (plain-RoPE stage 1, per_turn teacher, 700 steps)**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/train`

- **Jobs:** 9682, 9684
- **Dates:** 2026-10-08 01:22 .. 2026-10-08 02:10 PT
- **Base model:** Qwen/Qwen3-32B
- **Init:** stock Qwen3-32B
- **Parents (resolved):** base:qwen3-32b
- **Produces:** qwen1ch-s1pt-plain@700
- **Data:** corpus15 text-KL, NEW per_turn teacher cache (multi-turn teacher fix)
- **Recipe:** qwen-1ch/train/launch_s1_plain_pt.sh, MIXROPE=0, LR 2e-6, 288 rows/step, 18 nodes, 700 steps
- **Steps:** 0 -> 700
- **Evals:** genghisbench = 42/80 (w-700); 120 games 74 vs plain p606 83; kl = 0.02534 @700 (from 0.5584)
- **Checkpoints / exports:** w-700; dcp-700
- **Local paths:** /data/home/amartyavarma/qwen-1ch/train/ckpt-weka/q32-1ch-s1pt-plain-lr2e-6-r288-w144-s700-j9684/w-700
- **HF repos:** none named in run files
- **Notes:** Plain-RoPE redo (10-08 night plan) feeding qwen1ch-s2-plainpt. 9682 = smoke.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/HANDOVER_night_1008.md:20 -> Run 1 stage 1: launch_s1_plain_pt.sh, 700 steps, 18 nodes, from base Qwen3-32B, per_turn teacher cache`
  - `/data/home/amartyavarma/qwen-1ch/train/logs/q1ch-9684.log:1440 -> EVAL step 700 kl_word_invariant 0.02534`
  - `/data/home/amartyavarma/qwen-1ch/train/logs/q1ch-9684.log:1441 -> saved dcp-700 + Weka w-700`
  - `/data/home/amartyavarma/qwen-1ch/logs/night-1008.log:24 -> eval KL 0.5584 -> 0.0253 @700; GB 42/80`
  - `/data/home/amartyavarma/qwen-1ch/logs/night-1008.log:25 -> 120 games s1pt700 74 vs p606 83 (noise)`

### qwen1ch-s2-dev

**Qwen 1ch stage 2 development chain (q32-s2-1ch-lr5e-06 / lr4e-06), agreed baseline j7226 dcp-2557**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage2`

- **Jobs:** 5528, 5542, 5578, 5596, 5615, 5647, 5764, 5795, 5812, 5860, 5892, 5901, 6015, 6091, 6108, 6709, 7218, 7219, 7221, 7223, 7226, 7229
- **Dates:** 2026-10-01 22:08 .. 2026-10-03 07:51 PT
- **Base model:** unknown
- **Init:** stage-1 end /data/home/amartyavarma/qwen-1ch/stage1-final/q1ch-s0-mixrope-asst-j4866-w-4000
- **Parents (resolved):** qwen1ch-s1@4000
- **Produces:** qwen1ch-s2-dev@248, qwen1ch-s2-dev@500, qwen1ch-s2-dev@641, qwen1ch-s2-dev@2557
- **Data:** tts-pretrain-128m speech rows (7,500 frames, NEW_WORD) + text-KL on klpre-split1
- **Recipe:** stage2/train_s2q.py launch_s2q.sh; mix-RoPE; trunk LR 5e-6 (j5812..6015) then 4e-6 from j6091 (PHASE-2b, W_KL 20); 19 nodes / 152 GPUs; j7229 = lr6.5e-6 branch from keep-j7223 dcp-2357
- **Steps:** 0 -> 2557 (max_steps 6597)
- **Evals:** norm_word_ce = 2.7158 @2500 (j7226); genghisbench = w-248 12/20, w-641 15/20 (v3); kl = 0.01427 @2500; other = cb0 1.4254, light 5.1387 @2500
- **Checkpoints / exports:** dcp-248; w-500; w-641; dcp-2557; w-2557
- **Local paths:** /data/home/amartyavarma/hf-backup-staging/q32-s2-j7226-w2557
- **HF repos:** [InternalCan/qwen-1ch-s2-ckpts](https://huggingface.co/InternalCan/qwen-1ch-s2-ckpts)
- **Notes:** 5528..5795 are aborted/smoke launches at step 0-100 (lr5e-6, one lr2.5e-6 j5647). Text KL plateau ~0.014-0.017 matched the user's klpre target curve.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7226.log:179 -> EVAL step 2500 kl 0.01427, nword 2.7158, cb0 1.4254, light 5.1387`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7226.log:238 -> saved dcp-2557 + Weka w-2557`
  - `/data/home/amartyavarma/qwen-1ch/stage2/runs/cfg-6091.json -> lr_trunk 4e-06, max_steps 6597 (grep)`
  - `/data/home/amartyavarma/hf-backup-staging/readme/qwen-1ch-s2-ckpts/README.md:11 -> j7226-dcp-2557 = agreed Qwen stage-2 baseline`
  - `/data/home/amartyavarma/hf-backup-staging/readme/qwen-1ch-s2-ckpts/README.md:12 -> chain j5812 -> j5860 (dcp-248) -> j6091 -> j6108 -> j6709 -> j7218 -> j7219 -> j7223 -> j7226`
  - `memory:genghisbench.md:14 -> stage-2 w-248 12/20, w-641 15/20`
  - `memory:qwen-clean-1ch-plan.md:21 -> stage 2 ROW_FRAMES 7500, 19 nodes`

### qwen1ch-s2-promA

**Prometheus-replication Test A (q32-s2-1ch-promA-base-lr5e-05)**  
group `base-stage` · status **abandoned** · dir `/data/home/amartyavarma/qwen-1ch/stage2`

- **Jobs:** 7530, 7558, 7560, 7566
- **Dates:** 2026-10-03 13:34 .. 2026-10-03 16:42 PT
- **Base model:** Qwen/Qwen3-32B
- **Init:** /data/home/amartyavarma/models/Qwen3-32B (base), stock RoPE
- **Parents (resolved):** base:qwen3-32b
- **Produces:** none recorded
- **Data:** stage-2 speech only (no text micro); teacher cache only for eval KL
- **Recipe:** prod_promA.env: Prometheus trunk budget peak LR 5e-5, warmup 500, cosine -> 0.1x, clip 0.5, MIXROPE=0, NO text-KL anchor
- **Steps:** 0 -> 500 (j7530), resume 250 -> 276 (j7566)
- **Evals:** norm_word_ce = 2.6543 @500; kl = 1.66140 @500 (text intelligence collapses)
- **Checkpoints / exports:** w-250; w-276
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage2/ckpt-weka (j7530 w-250, j7566 w-276; dirs pruned from listing)
- **HF repos:** [InternalCan/qwen-1ch-s2-ckpts](https://huggingface.co/InternalCan/qwen-1ch-s2-ckpts)
- **Notes:** Shows text KL blows up without the anchor at Prometheus LR.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage2/prod_promA.env:1 -> TEST A: fresh from BASE Qwen3-32B with stock RoPE`
  - `/data/home/amartyavarma/qwen-1ch/stage2/prod_promA.env:3 -> peak LR 5e-5, warmup 500, cosine 0.1x, clip 0.5, no text-KL anchor`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7530.log:541 -> EVAL step 500 kl 1.66140, nword 2.6543`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7530.log:286 -> saved dcp-250 + w-250`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7566.log:60 -> saved dcp-276 + w-276`
  - `memory:qwen-mixrope-mistral-stock.md:16 -> exception: Prometheus-replication Test A from base with stock RoPE theta 1e6`
  - `/data/home/amartyavarma/hf-backup-staging/readme/qwen-1ch-s2-ckpts/README.md:13 -> Prometheus-budget tests j7530/j7566 (Test A), j7604 (B2), j7614 (B2m)`

### qwen1ch-s2-promB

**Prometheus test B / B-chat (promB-base / promB-chat-base adapt256 lrt4e-06)**  
group `base-stage` · status **abandoned** · dir `/data/home/amartyavarma/qwen-1ch/stage2`

- **Jobs:** 7569, 7582, 7583
- **Dates:** 2026-10-03 16:42 .. 2026-10-03 19:46 PT
- **Base model:** Qwen/Qwen3-32B
- **Init:** /data/home/amartyavarma/models/Qwen3-32B (base)
- **Parents (resolved):** base:qwen3-32b
- **Produces:** none recorded
- **Data:** stage-2 speech + chat-template text KL (B-chat)
- **Recipe:** rank-256 speech adapters, trunk LR 4e-6, adapter LR 5e-5 (prod_promB.env); 7582 = chat-KL probe
- **Steps:** 7569: 0 -> 50; 7583: 0 -> 307
- **Evals:** norm_word_ce = 2.8110 @300 (7583); 3.1052 @50 (7569); kl = 0.00108 @300 (7583, chat-template KL)
- **Checkpoints / exports:** w-307 (7583)
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Notes:** KL here is on the chat-template view (not comparable to the frame-layout KL).
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage2/runs/cfg-7583.json -> run_tag q32-s2-1ch-promB-chat-base-adapt256-lrt4e-06-lra5e-05 (grep)`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7583.log:341 -> EVAL step 300 kl 0.00108 nword 2.8110`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7583.log:349 -> saved dcp-307 + Weka export`

### qwen1ch-s2-promB2

**Prometheus test B2 (promB2-chat-base adapt256 lrt1e-05 lra5e-04)**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage2`

- **Jobs:** 7604
- **Dates:** 2026-10-03 19:46 .. 2026-10-03 22:18 PT
- **Base model:** Qwen/Qwen3-32B
- **Init:** /data/home/amartyavarma/models/Qwen3-32B (base)
- **Parents (resolved):** base:qwen3-32b
- **Produces:** none recorded
- **Data:** stage-2 speech + chat-template text KL
- **Recipe:** prod_promB2.env: adapt256, trunk LR 1e-5, adapter LR 5e-4
- **Steps:** 0 -> 345
- **Evals:** norm_word_ce = 2.7475 @300; kl = 0.00277 @300 (chat view)
- **Checkpoints / exports:** dcp-345; w-345
- **Local paths:** unknown
- **HF repos:** [InternalCan/qwen-1ch-s2-ckpts](https://huggingface.co/InternalCan/qwen-1ch-s2-ckpts)
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7604.log:341 -> EVAL step 300 kl 0.00277 nword 2.7475`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7604.log:387 -> saved dcp-345 + Weka export`

### qwen1ch-s2-promB2m

**Prometheus test B2m (promB2m-s1mix adapt256 lrt1e-05) = recipe of the stage-2 final run**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage2`

- **Jobs:** 7614
- **Dates:** 2026-10-03 22:19 .. 2026-10-04 01:02 PT
- **Base model:** unknown
- **Init:** stage-1 j4866 w-4000
- **Parents (resolved):** qwen1ch-s1@4000
- **Produces:** none recorded
- **Data:** stage-2 speech + frame-layout text-KL anchor
- **Recipe:** prod_promB2m.env: B2 recipe + MIXROPE=1, TEXT_CHAT=0, INIT_FROM stage-1 w-4000
- **Steps:** 0 -> 337
- **Evals:** norm_word_ce = 2.7320 @300; kl = 0.01626 @300
- **Checkpoints / exports:** dcp-337; w-337
- **Local paths:** unknown
- **HF repos:** [InternalCan/qwen-1ch-s2-ckpts](https://huggingface.co/InternalCan/qwen-1ch-s2-ckpts)
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage2/prod_promB2m.env:1 -> B2's recipe on stage-1 trunk + frame-layout text-KL anchor`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7614.log:341 -> EVAL step 300 kl 0.01626 nword 2.7320`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7614.log:379 -> saved dcp-337 + Weka export`
  - `/data/home/amartyavarma/hf-backup-staging/readme/qwen-1ch-s2-ckpts/README.md:13 -> 'B2m's recipe became the final run'`
  - `/data/home/amartyavarma/hf-backup-staging/readme/qwen-1ch-s2-ckpts/README.md:21 -> j7614-w-337-bf16 row`

### samft-qwen-lane

**samft-qwen-lane-1k (SAM 8B tower fine-tuned to the Qwen lane vocab)**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/ref/sam-graft`

- **Jobs:** 7616
- **Dates:** 2026-10-03..04 (exact unknown)
- **Base model:** unknown
- **Init:** SAM 8B InternalCan/sam-b700k-cd3-ad0-lead1-51960 (local /data/home/amartyavarma/models/sam-51960)
- **Parents (resolved):** base:sam-51960
- **Produces:** samft-qwen-lane@1000
- **Data:** Qwen stage-2 speech rows (304 eval rows for its eval)
- **Recipe:** ft_tower_q.py, text embedding table swapped for the Qwen lane vocab
- **Steps:** 0 -> 1000
- **Evals:** other = cb0 CE 1.2864 (top-1 0.601), light 4.3825 @1000 (from cb0 2.2931 @0)
- **Checkpoints / exports:** step-1000
- **Local paths:** /data/home/amartyavarma/ref/sam-graft/runs/samft-qwen-lane-1k/step-1000
- **HF repos:** [InternalCan/sam-graft-ckpts](https://huggingface.co/InternalCan/sam-graft-ckpts), [InternalCan/sam-b700k-cd3-ad0-lead1-51960](https://huggingface.co/InternalCan/sam-b700k-cd3-ad0-lead1-51960)
- **Notes:** SAM base node id 'base:sam-51960' is not in the canonical list (external SAM 8B speech model).
- **Citations:**
  - `/data/home/amartyavarma/hf-backup-staging/readme/sam-graft-ckpts/README.md:11 -> samft-qwen-lane-1k job 7616, step-1000 = SAM tower init of the Qwen stage-2 final run`
  - `/data/home/amartyavarma/hf-backup-staging/readme/sam-graft-ckpts/README.md:20 -> step-1000 cb0 CE 1.2864, light 4.3825; step 0 cb0 2.2931`
  - `/data/home/amartyavarma/hf-backup-staging/readme/sam-graft-ckpts/README.md:36 -> SAM base InternalCan/sam-b700k-cd3-ad0-lead1-51960`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7652.log:9 -> [sam] voice + light <- .../samft-qwen-lane-1k/step-1000`

### qwen1ch-s2-final

**Qwen stage-2 FINAL run (q32-s2-1ch-final-s1mix-adapt256-lra1e-3-sam, j7652 -> j7671 -> j7755 -> j7862)**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage2`

- **Jobs:** 7643, 7645, 7652, 7671, 7755, 7862
- **Dates:** 2026-10-04 01:34 .. 2026-10-05 12:44 PT
- **Base model:** unknown
- **Init:** stage-1 j4866 w-4000 trunk + SAM tower samft-qwen-lane-1k step-1000
- **Parents (resolved):** qwen1ch-s1@4000, samft-qwen-lane@1000
- **Produces:** qwen1ch-s2-final@2500, qwen1ch-s2-final@2750, qwen1ch-s2-final@3301, qwen1ch-s2-final@4289
- **Data:** tts-pretrain-128m speech (256 shards, 7,500-frame rows, NEW_WORD) + text-KL on split-1 pool (1,309,490 rows; 304 eval rows)
- **Recipe:** prod_final.env (B2m recipe + SAM graft): mix-RoPE, rank-256 speech-only adapters, LR_TRUNK 1e-5, W_KL 20 (30 from j7671), KL_ROWS 2, K=16 light sub-steps, flat schedule; 19 nodes / 152 GPUs; j7755 resumed from j7671 dcp-2750 (pre-spike), j7862 from j7755 dcp-3301
- **Steps:** 0 -> 4289 (max_steps 4396)
- **Evals:** norm_word_ce = 2.4239 @4250 (final); 2.4851 @3300; 2.5553 @2500; genghisbench = j7671 w-2500 19/20 (v5), 16/20 (v4); 60-game: 2500 42, 3000 46, 3301 41, 4000 33, 4250 40; kl = 0.01732 @4250; 0.02104 @2500; other = cb0 1.2038, light 4.2279 @4250
- **Checkpoints / exports:** dcp-996; w-2250; dcp-2250; w-2500; dcp-2750; w-3000; w-3301; w-4289; dcp-4289
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage2/keep/j7862-w-4289; /data/home/amartyavarma/qwen-1ch/stage2/keep/j7755-w-3000; /data/home/amartyavarma/qwen-1ch/stage2/ckpt-weka/q32-s2-1ch-final-s1mix-adapt256-lra1e-3-sam-j7862/w-4289
- **HF repos:** [InternalCan/qwen-1ch-s2-final-ckpts](https://huggingface.co/InternalCan/qwen-1ch-s2-final-ckpts)
- **Notes:** Its w-4289 export (stage2/keep/j7862-w-4289) is the init of every stage-3 run. j7643/j7645 are step-0 smokes. GB best mid-run (w-3000); w-4289 GB not found.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage2/prod_final.env:1 -> FINAL Qwen stage-2 run recipe (LR_TRUNK 1e-5, W_KL 20, K=16, SAM graft, flat schedule)`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7652.log:9 -> SAM voice+light tower init`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7652.log:1005 -> EVAL step 950 nword 2.6168, KL 0.02042`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7671.log:6 -> [data] split-1 pool 1,309,490 rows; speech 256 shards, 7500 frames`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7671.log:7 -> [init] trunk <- stage1-final j4866 w-4000`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7671.log:1576 -> EVAL step 2500 KL 0.02104 nword 2.5553`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7671.log:1832 -> EVAL step 2750 nword 2.5495`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7671.log:1986 -> EVAL step 2900 word CE 4.58 / nword 2.7487 (spike)`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7755.log:599 -> EVAL step 3300 KL 0.02047 nword 2.4851`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7862.log:1006 -> EVAL step 4250 KL 0.01732, nword 2.4239, cb0 1.2038, light 4.2279`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-7862.log:1047 -> saved dcp-4289 + Weka w-4289`
  - `/data/home/amartyavarma/hf-backup-staging/readme/qwen-1ch-s2-final-ckpts/README.md:15 -> leg 1 j7652 0-995 fresh from j4866 w-4000 + SAM tower`
  - `/data/home/amartyavarma/hf-backup-staging/readme/qwen-1ch-s2-final-ckpts/README.md:18 -> 'j7671 is the Qwen stage-2 FINAL run'`
  - `memory:genghisbench.md:21 -> v5: j7671 w-2500 19/20`
  - `memory:genghisbench.md:19 -> j7671 w-2500 16/20 (v4)`
  - `memory:genghisbench.md:23 -> 60-game GB 2500 42, 2750 42, 3000 46, 3301 41, 3750 43, 4000 33, 4250 40`

### qwen1ch-s2-plainpt

**q32-s2-1ch-plainpt-noadapt-hiLR (plain-RoPE stage 2, phase A + continuation)**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage2`

- **Jobs:** 9716, 9810, 9876, 9920, 9949, 9990, 10021, 10042, 10080, 10336, 10569
- **Dates:** 2026-10-08 02:29 .. 2026-10-09 05:21 PT
- **Base model:** unknown
- **Init:** /data/home/amartyavarma/qwen-1ch/train/ckpt-weka/q32-1ch-s1pt-plain-lr2e-6-r288-w144-s700-j9684/w-700
- **Parents (resolved):** qwen1ch-s1pt-plain@700
- **Produces:** qwen1ch-s2-plainpt@500, qwen1ch-s2-plainpt@1150, qwen1ch-s2-plainpt@2348
- **Data:** stage-2 speech + text KL (W_KL 10 -> 30 -> 15 over the chain)
- **Recipe:** prod_s2_plain_pt.env (= final stage-2 recipe of j7862 except CHANGED block): plain RoPE, no adapter, full FT, trunk LR 3e-5 (6e-5 from j9876, 3.3e-5 from j10021, 6e-5 in j10569); resharded 18 -> 10 nodes at 1150; 9920/9949 = LR-down branch br1150 (w-1190)
- **Steps:** 0 -> 2348 (max_steps 4644)
- **Evals:** norm_word_ce = 2.6036 @2325 (final); 2.7242 @500; genghisbench = w-1150 48/80; LD1190 45/80 (v7); kl = 0.08790 @2325; 0.04744 @500
- **Checkpoints / exports:** w-500; w-1190 (branch); w-2348
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage2/ckpt-weka/q32-s2-1ch-plainpt-noadapt-hiLR-lrt3e-5-j10569/w-2348; /data/home/amartyavarma/qwen-1ch/stage2/ckpt-weka/q32-s2-1ch-plainpt-noadapt-hiLR-lrt3e-5-br1150-lrdown-j9920/w-1190
- **HF repos:** none named in run files
- **Notes:** Ended 10-09 05:21 at 2348 with no later resume found; not used by PPM (PPM started fresh from base 10-11).
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage2/prod_s2_plain_pt.env:1 -> RUN 2 recipe = final stage-2 recipe (j7862) except CHANGED block`
  - `/data/home/amartyavarma/qwen-1ch/HANDOVER_night_1008.md:21 -> phase A launch_s2_plain_hiLR.sh 500 steps higher trunk LR, no adapter`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-9716.log:541 -> EVAL step 500 KL 0.04744 nword 2.7242`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-10569.log:371 -> EVAL step 2325 KL 0.08790 nword 2.6036`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-10569.log:397 -> saved dcp-2348 + Weka w-2348`
  - `/data/home/amartyavarma/qwen-1ch/logs/night-1008.log:70 -> main line resumed on 10 nodes (job 9990), W_KL 30, GB v7 w-1150 48/80, LD1190 45/80`
  - `/data/home/amartyavarma/qwen-1ch/stage2/runs/cfg-10569.json -> lr_trunk 6e-05, max_steps 4644 (grep)`

### qwen1ch-s2-plainpt-lowlr

**q32-s2-1ch-plainpt-noadapt-lowLRcheck-from500 (phase B)**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage2`

- **Jobs:** 9722
- **Dates:** 2026-10-08 05:46 .. 2026-10-08 06:29 PT
- **Base model:** unknown
- **Init:** full resume of j9716 dcp-500
- **Parents (resolved):** qwen1ch-s2-plainpt@500
- **Produces:** qwen1ch-s2-plainpt-lowlr@600
- **Data:** same as phase A
- **Recipe:** launch_s2_plain_lowLR_check.sh: short low-LR run to see whether KL comes back
- **Steps:** 500 -> 600
- **Evals:** norm_word_ce = 2.7117 @600; genghisbench = submitted (result not found); kl = 0.03759 @600 (from 0.04744 @500)
- **Checkpoints / exports:** w-600
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage2/ckpt-weka/q32-s2-1ch-plainpt-noadapt-lowLRcheck-from500-j9722/w-600
- **HF repos:** none named in run files
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-9722.log:135 -> EVAL step 600 KL 0.03759 nword 2.7117`
  - `/data/home/amartyavarma/qwen-1ch/stage2/logs/q1ch-s2-9722.log:136 -> saved dcp-600 + Weka w-600`
  - `/data/home/amartyavarma/qwen-1ch/logs/night-1008.log:26 -> phase B finished; GenghisBench submitted`

### opd-proto

**OPD on-policy distillation prototype (LoRA on stage-2 exports)**  
group `other` · status **abandoned** · dir `/data/home/amartyavarma/qwen-1ch/opd`

- **Jobs:** 6210, 6217, 6228, 6229, 6239, 6250, 6304, 6305, 6384, 6385
- **Dates:** 2026-10-02 15:03 .. 2026-10-02 20:28 PT
- **Base model:** unknown
- **Init:** stage-2 dev exports j5901 w-641 (runs A/A2) and j6108 w-500 (runs B/B2)
- **Parents (resolved):** qwen1ch-s2-dev@641, qwen1ch-s2-dev@500
- **Produces:** none recorded
- **Data:** user turns of held-out corpus split-0 rows; student rollouts scored by stock Qwen3-32B teacher
- **Recipe:** opd_q.py: frozen bf16 student + LoRA on q/k/v/o/gate/up/down, per-token reverse KL; runs A-w641-lr1e4, A2-w641-perturn, B-j6108w500-lr1e4, B2-j6108w500-perturn
- **Steps:** A: -> ck-00024; A2: -> 100
- **Evals:** genghisbench = A2 @100: 25/40 (first 20: 12/20)
- **Checkpoints / exports:** runs/A-w641-lr1e4/ck-00024.pt
- **Local paths:** /data/home/amartyavarma/qwen-1ch/opd/runs
- **HF repos:** none named in run files
- **Notes:** GENGHISBENCH @100 25/40 is from the tail of opd-A2-6304.log (last lines). Main value of this work = the teacher think-template finding.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/opd/opd_q.py:2 -> OPD prototype (quick test, not production)`
  - `/data/home/amartyavarma/qwen-1ch/opd/logs/opd-A2-6304.log:1 -> CK=opd/init/j5901/w-641`
  - `/data/home/amartyavarma/qwen-1ch/opd/logs/opd-B2-6305.log:1 -> CK=opd/init/j6108/w-500`
  - `memory:qwen-kl-teacher-think-template.md:13 -> the OPD agent found the canonical-teacher think-template bug`

### qwen1ch-s3-exp1

**Qwen stage 3 duplex experiment 1 (q32-s3-exp1-duplex-sp-split2-glu-r2), steps 0-750**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 8176, 8185, 8197, 8246, 8324, 8327, 8333
- **Dates:** 2026-10-05 13:33 .. 2026-10-05 19:26 PT
- **Base model:** unknown
- **Init:** /data/home/amartyavarma/qwen-1ch/stage2/keep/j7862-w-4289 (weights only, fresh Adam)
- **Parents (resolved):** qwen1ch-s2-final@4289
- **Produces:** qwen1ch-s3-exp1@750
- **Data:** SAM duplex podcast sets (InternalCan/podcast-raw-audio-data-mass: 700k + intelligent-podcasts), 10-min 7,500-frame rows (slinky-0 node: 272,970 rows), swap prob 0.5 + system prompt + two-track text KL on split-2 cache, 2:1 conv:text rows
- **Recipe:** stage3/train_s3q.py, prod_s3_exp1.env; plain-sum input, per-channel tables, GLU projectors; trunk LR 1e-5; cosine -> 20% over planned 17,050 steps, WSD from 425; 19 nodes / 152 GPUs
- **Steps:** 0 -> 750
- **Evals:** norm_word_ce = B 2.1200 @750; kl = 0.0264 @750
- **Checkpoints / exports:** KEEP_dcp-750; w-750
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-s3-exp1-duplex-sp-split2-glu-r2-j8333
- **HF repos:** none named in run files
- **Notes:** Normalised word CE is from metrics.jsonl (the log EVAL line prints raw word CE only). Judged vs Hestia (memory:qwen-s3-success-criteria.md:9). 8176/8185 cancelled at step 0.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/logs/q1ch-s3-8197.log:4 -> [init] model <- stage-2 export stage2/keep/j7862-w-4289`
  - `/data/home/amartyavarma/qwen-1ch/stage3/logs/q1ch-s3-8197.log:3 -> duplex node slinky-0: 272,970 rows`
  - `/data/home/amartyavarma/qwen-1ch/stage3/prod_s3_exp1.env:1 -> EXPERIMENT 1: duplex + system prompt + two-track text KL on split-2, 2:1 rows, prob 0.5 swap, plain sum`
  - `/data/home/amartyavarma/qwen-1ch/stage3/prod_s3_exp1.env:2 -> cosine -> 20% over 17,050 steps for 0..425; WSD from 425`
  - `/data/home/amartyavarma/qwen-1ch/stage3/README_S3.md:195 -> HORIZON_S3 17,050 WSD decision`
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-exp1-duplex-sp-split2-glu-r2-j8333/metrics.jsonl:288 -> step 750 eval_speech_B/word_ce_norm 2.1200, KL 0.0264`
  - `memory:qwen-stage3-plan.md:19 -> stage-3 data = SAM duplex sets from InternalCan/podcast-raw-audio-data-mass`
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/cfg-8197.json -> lr_trunk 1e-05 (grep)`

### qwen1ch-s3-lr03

**Qwen stage-3 MAIN run q32-s3-exp1-lr0.3-from750 (LRs x0.3), paused at 5900**  
group `base-stage` · status **paused** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 8424, 8724, 8796, 8815
- **Dates:** 2026-10-05 22:25 .. 2026-10-07 13:33 PT
- **Base model:** unknown
- **Init:** full resume of j8333 dcp-750 with all peak LRs x0.3
- **Parents (resolved):** qwen1ch-s3-exp1@750
- **Produces:** qwen1ch-s3-lr03@1250, qwen1ch-s3-lr03@1500, qwen1ch-s3-lr03@5900
- **Data:** as exp1 (SAM duplex + split-2 text KL); row exclusion from 2979
- **Recipe:** prod_s3_exp1.env + relaunch-main-lr0.3-750.sh / relaunch-main-rowx-*.sh; trunk LR 3e-6; WSD planned decay 13640 -> 16994; ~20 s/step
- **Steps:** 750 -> 5900 (planned end 16,994)
- **Evals:** norm_word_ce = B 2.0018 @5900 (A 2.0138); 2.0498 @1500; kl = 0.01213 @5900; other = B cb0 0.6830, light 2.1559 @5900
- **Checkpoints / exports:** keep-lr0.3-dcp-1250; dcp-1500; w-5750; dcp-5900; w-5900
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-s3-exp1-lr0.3-from750-j8815/w-5900; /data/home/amartyavarma/qwen-1ch/stage4/keep/s3-init-q32-s3-exp1-lr0.3-from750-j8815-w-5900
- **HF repos:** [InternalCan/qwen-1ch-s3-exp1-lr0.3-from750-ckpts](https://huggingface.co/InternalCan/qwen-1ch-s3-exp1-lr0.3-from750-ckpts)
- **Notes:** Never resumed past 5900 (no later q32-s3-exp1-lr0.3 logs found); w-5900 became the init of every stage-3 FT, pose-s3 run and every stage-4 run. Its relation to Hestia (wandb demeter-32b-ga3) is comparison only.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-exp1-lr0.3-from750-j8815/metrics.jsonl:3007 -> step 5900 A 2.0138, B 2.0018, KL 0.01213`
  - `/data/home/amartyavarma/qwen-1ch/stage3/logs/q1ch-s3-8815.log:3863 -> EVAL step 5900 KL 0.01213, B cb0 0.6830, light 2.1559`
  - `/data/home/amartyavarma/qwen-1ch/stage3/logs/q1ch-s3-8815.log:3865 -> saved dcp-5900 + Weka w-5900`
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-exp1-lr0.3-from750-j8424/metrics.jsonl:783 -> step 1500 B 2.0498, KL 0.0159`
  - `/data/home/amartyavarma/qwen-1ch/stage3/logs/pause-main-5900.log:1 -> pause touched for 5900`
  - `/data/home/amartyavarma/qwen-1ch/stage3/BOUNDARY_8628.md:7 -> run ends at 16,994 (end of duplex speech data)`
  - `/data/home/amartyavarma/hf-backup-staging/s3-lr0.3/s3bak.py:3 -> Repo InternalCan/qwen-1ch-s3-exp1-lr0.3-from750-ckpts`
  - `/data/home/amartyavarma/qwen-1ch/stage4/RUNS.json:21 -> stage-4 init_from keep/s3-init-...-j8815-w-5900`
  - `memory:qwen-stage4-voxtral-plan.md:11 -> stage 4 planned to continue from end of stage 3 (17,050)`

### qwen1ch-s3-exp1b

**q32-s3-exp1b-trunk1e-4-kl10 (branch from 750)**  
group `base-stage` · status **abandoned** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 8347
- **Dates:** 2026-10-05 19:27 .. 20:09 PT
- **Base model:** unknown
- **Init:** full resume j8333 dcp-750
- **Parents (resolved):** qwen1ch-s3-exp1@750
- **Produces:** none recorded
- **Data:** as exp1
- **Recipe:** trunk LR 1e-4, W_KL 10
- **Steps:** 750 -> 850
- **Evals:** norm_word_ce = B 2.1433 @850; kl = 0.1329 @850
- **Checkpoints / exports:** unknown
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Notes:** KL blew up; killed.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-exp1b-trunk1e-4-kl10-j8347/metrics.jsonl:105 -> step 850 B 2.1433, KL 0.1329`
  - `/data/home/amartyavarma/qwen-1ch/stage3/logs/q1ch-s3-8347.log -> exit 143 20:09 PT (tail)`

### qwen1ch-s3-exp1c

**q32-s3-exp1c-decay100-from750**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 8354
- **Dates:** 2026-10-05 20:09 .. 20:58 PT
- **Base model:** unknown
- **Init:** full resume j8333 dcp-750
- **Parents (resolved):** qwen1ch-s3-exp1@750
- **Produces:** none recorded
- **Data:** as exp1
- **Recipe:** 100-step LR decay probe
- **Steps:** 750 -> 850
- **Evals:** norm_word_ce = B 2.0871 @850; kl = 0.0205 @850
- **Checkpoints / exports:** unknown
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-exp1c-decay100-from750-j8354/metrics.jsonl:121 -> step 850 B 2.0871, KL 0.0205`

### qwen1ch-s3-exp1d

**q32-s3-exp1d-txtdup-from750 (Gemma text-duplex CE instead of text KL)**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 8403
- **Dates:** 2026-10-05 21:00 .. 22:24 PT
- **Base model:** unknown
- **Init:** full resume j8333 dcp-750
- **Parents (resolved):** qwen1ch-s3-exp1@750
- **Produces:** none recorded
- **Data:** SAM duplex + Demeter's Gemma text-duplex store ~/data/gemma-txtdup-v1 (S3_TXT_MODE=duplex_ce)
- **Recipe:** txtdup_s3q.py replaces the text-KL micro
- **Steps:** 750 -> 950
- **Evals:** norm_word_ce = B 2.1514 @950; kl = 0.0681 @950
- **Checkpoints / exports:** unknown
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-exp1d-txtdup-from750-j8403/metrics.jsonl:209 -> step 950 B 2.1514, KL 0.0681`
  - `/data/home/amartyavarma/qwen-1ch/stage3/README_S3.md:179 -> exp1d: Gemma text-duplex conversations replace the text-KL micro`

### qwen1ch-s3-exp1e

**q32-s3-exp1e / e2 / e3 rank-1024 adapter branches from 1250**  
group `base-stage` · status **abandoned** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 8495, 8574, 8582, 8597
- **Dates:** 2026-10-06 04:00 .. ~09:50 PT
- **Base model:** unknown
- **Init:** full resume keep-lr0.3-dcp-1250 (8597 resumes 8582 dcp-1303)
- **Parents (resolved):** qwen1ch-s3-lr03@1250
- **Produces:** none recorded
- **Data:** as main
- **Recipe:** exp1e rank1024; exp1e2 rank1024warm; exp1e3 rank1024 lr3.75e-5 adapter
- **Steps:** 1250 -> 1350 (e) / 1425 (e2) / 1400 (e3)
- **Evals:** norm_word_ce = B e 2.0597 @1350; e2 2.0535 @1425; e3 2.0551 @1400; kl = e 0.0165, e2 0.0161, e3 0.0162
- **Checkpoints / exports:** unknown
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Notes:** Grouped (same branch point, same question). Main lr0.3 at 1500 = 2.0498.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-exp1e-rank1024-from1250-j8495/metrics.jsonl:105 -> step 1350 B 2.0597`
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-exp1e2-rank1024warm-from1250-j8574/metrics.jsonl:183 -> step 1425 B 2.0535`
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-exp1e3-rank1024-lr3.75e-5-from1250-j8597/metrics.jsonl:102 -> step 1400 B 2.0551`

### qwen1ch-s3-exp1f

**q32-s3-exp1f-lr0.1-from1250**  
group `base-stage` · status **abandoned** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 8462
- **Dates:** 2026-10-06 03:03 .. 03:59 PT
- **Base model:** unknown
- **Init:** full resume j8424 dcp-1250
- **Parents (resolved):** qwen1ch-s3-lr03@1250
- **Produces:** none recorded
- **Data:** as main
- **Recipe:** LRs x0.1 (trunk 1e-6)
- **Steps:** 1250 -> 1400
- **Evals:** norm_word_ce = B 2.0469 @1400; kl = 0.0152 @1400
- **Checkpoints / exports:** unknown
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-exp1f-lr0.1-from1250-j8462/metrics.jsonl:157 -> step 1400 B 2.0469, KL 0.0152`

### qwen1ch-s3-exp1r

**q32-s3-exp1r-repeat-from1250 (repeat/control)**  
group `base-stage` · status **abandoned** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 8590, 8595
- **Dates:** 2026-10-06 07:18 .. 07:44 PT
- **Base model:** unknown
- **Init:** full resume keep-lr0.3-dcp-1250
- **Parents (resolved):** qwen1ch-s3-lr03@1250
- **Produces:** none recorded
- **Data:** as main
- **Recipe:** repeat of main from 1250 (noise check)
- **Steps:** 1250 -> 1300
- **Evals:** norm_word_ce = B 2.0561 @1300; kl = 0.0165 @1300
- **Checkpoints / exports:** unknown
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-exp1r-repeat-from1250-j8595/metrics.jsonl:53 -> step 1300 B 2.0561`

### qwen1ch-s3-aloe-ft

**q32-s3-aloe-ft-2n (Aloe sysprompt duplex FT on stage-3 w-5900)**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 9474, 9478, 9482, 9485, 9494
- **Dates:** 2026-10-07 23:40 .. 2026-10-08 00:56 PT
- **Base model:** unknown
- **Init:** FT_FROM stage3 j8815 w-5900 (fresh step 0, fresh Adam)
- **Parents (resolved):** qwen1ch-s3-lr03@5900
- **Produces:** qwen1ch-s3-aloe-ft@93
- **Data:** akv13/aloe-duplex-ft-sysprompts (1,661 rows, 189.5 h, 409 conversations)
- **Recipe:** prod_s3_aloe_ft.env (j8815 recipe + [ft] overrides), 2 nodes, 1 epoch; final = r1-nokl j9494
- **Steps:** 0 -> 93
- **Evals:** norm_word_ce = B 1.9529 @93 (own eval set)
- **Checkpoints / exports:** final export (dir q32-s3-aloe-ft-2n-r1-nokl-from5900-j9494)
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-s3-aloe-ft-2n-r1-nokl-from5900-j9494
- **HF repos:** none named in run files
- **Notes:** 9474-9485 = 'e1' launch attempts. Pre-PPM Aloe FT on the Qwen stage-3 model.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/prod_s3_aloe_ft.env:1 -> Aloe FT data akv13/aloe-duplex-ft-sysprompts 1,661 rows`
  - `/data/home/amartyavarma/qwen-1ch/stage3/logs/q1ch-s3-9494.log -> '[ft] FINETUNE INIT ... <- j8815/w-5900' (grep)`
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-aloe-ft-2n-r1-nokl-from5900-j9494/metrics.jsonl:104 -> step 93 B 1.9529`

### qwen1ch-s3-aloe-ft-8n

**q32-s3-aloe-ft-8n-r1-nokl-from5900**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 21926
- **Dates:** 2026-10-16 02:16 .. 02:28 PT
- **Base model:** unknown
- **Init:** FT_FROM j8815 w-5900
- **Parents (resolved):** qwen1ch-s3-lr03@5900
- **Produces:** none recorded
- **Data:** as aloe-ft-2n
- **Recipe:** prod_s3_aloe_ft_8n.env: 8-node copy (steps /4)
- **Steps:** 0 -> 23
- **Evals:** norm_word_ce = B 1.9609 @23
- **Checkpoints / exports:** unknown
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-s3-aloe-ft-8n-r1-nokl-from5900-j21926
- **HF repos:** none named in run files
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/prod_s3_aloe_ft_8n.env:1 -> 8-node copy of prod_s3_aloe_ft.env`
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-aloe-ft-8n-r1-nokl-from5900-j21926/metrics.jsonl:32 -> step 23 B 1.9609`

### qwen1ch-s3-m1

**q32-s3-m1-s2hnone-from5900 (Model 1: two-sided duplex FT on stage-2 handover none bucket)**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 9939, 9995, 9999
- **Dates:** 2026-10-08 19:06 .. 21:03 PT
- **Base model:** unknown
- **Init:** FT_FROM j8815 w-5900
- **Parents (resolved):** qwen1ch-s3-lr03@5900
- **Produces:** none recorded
- **Data:** stage-2 handover 'none' crossbleed bucket (9,012 conversations, 1,433 h)
- **Recipe:** prod_s3_m1.env
- **Steps:** 0 -> 365 (9999 resumed dcp-100)
- **Evals:** none recorded
- **Checkpoints / exports:** unknown
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-s3-m1-s2hnone-from5900-j9999
- **HF repos:** none named in run files
- **Notes:** Superseded by m1fix (j10036). Final norm word CE not extracted (unknown).
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/prod_s3_m1.env:1 -> MODEL 1 two-sided duplex FT on handover none bucket`
  - `/data/home/amartyavarma/qwen-1ch/stage3/logs/q1ch-s3-9939.log -> FINETUNE INIT <- j8815 w-5900 (grep)`

### qwen1ch-s3-m1fix

**q32-s3-m1fix-s2hnone-from5900**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 10036
- **Dates:** 2026-10-08 21:05 .. 22:46 PT
- **Base model:** unknown
- **Init:** FT_FROM j8815 w-5900
- **Parents (resolved):** qwen1ch-s3-lr03@5900
- **Produces:** qwen1ch-s3-m1fix@365
- **Data:** as m1 (voice-clip sibling-rule fix)
- **Recipe:** prod_s3_m1.env + fix
- **Steps:** 0 -> 365
- **Evals:** norm_word_ce = B 1.9257 @365
- **Checkpoints / exports:** unknown
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-s3-m1fix-s2hnone-from5900-j10036
- **HF repos:** none named in run files
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-m1fix-s2hnone-from5900-j10036/metrics.jsonl:381 -> step 365 B 1.9257`

### qwen1ch-s3-tt-script

**q32-s3-tt-script-from5900 (turn-taking script FT v1)**  
group `other` · status **abandoned** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 10773
- **Dates:** 2026-10-09 05:39 .. 07:13 PT
- **Base model:** unknown
- **Init:** FT_FROM j8815 w-5900
- **Parents (resolved):** qwen1ch-s3-lr03@5900
- **Produces:** none recorded
- **Data:** turn-taking-annotated SAM conversations (tt-s3), script prefix on 50% of rows
- **Recipe:** prod_s3_tt.env
- **Steps:** 0 -> 360
- **Evals:** norm_word_ce = B 2.0298 @360 (tt eval set)
- **Checkpoints / exports:** unknown
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-s3-tt-script-from5900-j10773
- **HF repos:** none named in run files
- **Notes:** Restarted as tt v2 (scriptvoice) per user 10-09.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/prod_s3_tt.env:1 -> TT finetune of j8815 w-5900, script prefix`
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-tt-script-from5900-j10773/metrics.jsonl:367 -> step 360 B 2.0298`

### qwen1ch-s3-tt-scriptvoice

**q32-s3-tt-scriptvoice-from5900 (TT v2: script every row + voice clips)**  
group `other` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 10886
- **Dates:** 2026-10-09 07:17 .. 17:03 PT
- **Base model:** unknown
- **Init:** FT_FROM j8815 w-5900
- **Parents (resolved):** qwen1ch-s3-lr03@5900
- **Produces:** qwen1ch-s3-tt-scriptvoice@2096
- **Data:** tt-s3 rows, script on every row, voice-clip prompts, no persona, no text KL
- **Recipe:** prod_s3_ttv.env
- **Steps:** 0 -> 2306 (final step per preflight; export w-2096)
- **Evals:** norm_word_ce = B 0.0849 @2040 (scripted rows: transcript given, not comparable)
- **Checkpoints / exports:** w-2096
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-s3-tt-scriptvoice-from5900-j10886/w-2096
- **HF repos:** none named in run files
- **Notes:** Very low word CE because the script prefix carries the transcript.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/prod_s3_ttv.env:1 -> TT v2 restart from j8815 w-5900`
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-tt-scriptvoice-from5900-j10886/metrics.jsonl:2075 -> step 2040 B 0.0849`
  - `/data/home/amartyavarma/qwen-1ch/stage3/logs/q1ch-s3-12582.log:2 -> latest job 10886, final step 2306; exports w-2096`

### qwen1ch-s3-tt3

**q32-s3-tt3-from2096 (TT v3, turn_taking_v3 conversations)**  
group `other` · status **stopped-with-save** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 11460, 11717
- **Dates:** 2026-10-09 19:41 .. 2026-10-10 00:38 PT
- **Base model:** unknown
- **Init:** FT_FROM j10886 w-2096
- **Parents (resolved):** qwen1ch-s3-tt-scriptvoice@2096
- **Produces:** qwen1ch-s3-tt3@792
- **Data:** turn_taking_v3 (28.7k new episodes)
- **Recipe:** prod_s3_tt3.env, one epoch plain continued training
- **Steps:** 0 -> 792 (11717 resumed dcp-480)
- **Evals:** norm_word_ce = B 0.1567 @720 (scripted)
- **Checkpoints / exports:** w-792
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-s3-tt3-from2096-j11717/w-792
- **HF repos:** [InternalCan/tt-v3-base-w792](https://huggingface.co/InternalCan/tt-v3-base-w792)
- **Notes:** w-792 called tt-v3's 'stop export' (prod_s3_ttaloe.env:3).
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/prod_s3_tt3.env:1 -> TT v3 one epoch of j10886 w-2096 on turn_taking_v3`
  - `/data/home/amartyavarma/qwen-1ch/stage3/logs/q1ch-s3-11460.log -> FINETUNE INIT <- j10886 w-2096 (grep)`
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-tt3-from2096-j11717/metrics.jsonl:244 -> step 720 B 0.1567`

### qwen1ch-s3-ttaloe

**q32-s3-ttaloe-from792 (Aloe TT FT)**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 11938, 11953
- **Dates:** 2026-10-10 00:47 .. 01:40 PT
- **Base model:** unknown
- **Init:** FT_FROM j11717 w-792
- **Parents (resolved):** qwen1ch-s3-tt3@792
- **Produces:** none recorded
- **Data:** Aloe app calls ('none' crossbleed bucket, 11,925 rows, one orientation)
- **Recipe:** prod_s3_ttaloe.env, 10 nodes
- **Steps:** 0 -> 145
- **Evals:** norm_word_ce = B 0.2143 @145 (scripted)
- **Checkpoints / exports:** unknown
- **Local paths:** unknown
- **HF repos:** [InternalCan/tt-v3-aloe-w145](https://huggingface.co/InternalCan/tt-v3-aloe-w145)
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/prod_s3_ttaloe.env:1 -> Aloe TT FT, FT from tt-v3 w-792`
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-ttaloe-from792-j11953/metrics.jsonl:151 -> step 145 B 0.2143`

### qwen1ch-s3-ttpw

**q32-s3-ttpw-from792 (TT piecewise on podcast TT rows)**  
group `other` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 20336
- **Dates:** 2026-10-13 18:50 .. 22:51 PT
- **Base model:** unknown
- **Init:** FT_FROM j11717 w-792
- **Parents (resolved):** qwen1ch-s3-tt3@792
- **Produces:** qwen1ch-s3-ttpw@996
- **Data:** unconsumed podcast turn-taking rows, piecewise script + EOS_PIECE
- **Recipe:** prod_s3_ttpw.env, 16 nodes (world 128)
- **Steps:** 0 -> 996
- **Evals:** norm_word_ce = B 0.1573 @996 (scripted)
- **Checkpoints / exports:** w-996
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-s3-ttpw-from792-j20336
- **HF repos:** [InternalCan/tt-s3-ttpw-j20336](https://huggingface.co/InternalCan/tt-s3-ttpw-j20336)
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/prod_s3_ttpw.env:1 -> TT piecewise fine-tune of j11717 w-792`
  - `/data/home/amartyavarma/qwen-1ch/stage3/logs/q1ch-s3-20336.log -> FINETUNE INIT <- j11717 w-792 (grep)`
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-ttpw-from792-j20336/metrics.jsonl:1014 -> step 996 B 0.1573`

### qwen1ch-s3-alopw

**q32-s3-alopw-from996 / alopw8 (Aloe piecewise FT)**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 20465, 20471, 20474, 20477, 20480, 20490
- **Dates:** 2026-10-14 00:10 .. 02:39 PT
- **Base model:** unknown
- **Init:** FT_FROM j20336 w-996
- **Parents (resolved):** qwen1ch-s3-ttpw@996
- **Produces:** none recorded
- **Data:** Aloe calls, piecewise script
- **Recipe:** alopw chain 20465 -> 20474 -> 20477 -> 20480 (cancelled); alopw8 = 8-node variant 20490
- **Steps:** alopw 0 -> 120; alopw8 0 -> 129
- **Evals:** norm_word_ce = B alopw 0.2214 @120; alopw8 0.2189 @129 (scripted)
- **Checkpoints / exports:** unknown
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-s3-alopw8-from996-j20490
- **HF repos:** none named in run files
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-alopw-from996-j20480/metrics.jsonl:34 -> step 120 B 0.2214`
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-alopw8-from996-j20490/metrics.jsonl:133 -> step 129 B 0.2189`
  - `/data/home/amartyavarma/qwen-1ch/stage3/logs/q1ch-s3-20465.log -> FINETUNE INIT <- j20336 w-996 (grep)`

### qwen1ch-s3-gen-usertext

**q32-s3-gen-usertext(-kl)-from5900 (user-text comparison FT)**  
group `other` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 14550, 14709, 15813
- **Dates:** 2026-10-10 22:55 .. 2026-10-11 03:29 PT
- **Base model:** unknown
- **Init:** FT_FROM j8815 w-5900
- **Parents (resolved):** qwen1ch-s3-lr03@5900
- **Produces:** none recorded
- **Data:** stage-4 gen run's rows (qwen-1ch-data/s4ft-gen, QA, 24,494 train rows); A = user text + Mimi, loss on B only; -kl adds split-3 text KL
- **Recipe:** prod_s3_gen_usertext.env / prod_s3_gen_usertext_kl.env
- **Steps:** 0 -> 306
- **Evals:** norm_word_ce = B 1.1672 @306 (no KL); 1.1644 @306 (with KL); kl = 0.0142 @306 (-kl run)
- **Checkpoints / exports:** unknown
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-s3-gen-usertext-from5900-j14709; /data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-s3-gen-usertext-kl-from5900-j15813
- **HF repos:** none named in run files
- **Notes:** Compare with qwen1ch-s4-ft-gen (Voxtral on A) 1.2864 @306.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/prod_s3_gen_usertext.env:1 -> user-text comparison finetune on stage-4 gen rows`
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-gen-usertext-from5900-j14709/metrics.jsonl:320 -> step 306 B 1.1672`
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-gen-usertext-kl-from5900-j15813/metrics.jsonl:320 -> step 306 B 1.1644, KL 0.0142`

### qwen1ch-pose-s3

**q32-pose-s3 (first pose multi-task FT on stage 3)**  
group `pose` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 10854, 11354, 11379
- **Dates:** 2026-10-09 17:04 .. 19:05 PT
- **Base model:** unknown
- **Init:** FT_FROM j8815 w-5900 (fallback: no complete tt-s3 final export)
- **Parents (resolved):** qwen1ch-s3-lr03@5900
- **Produces:** none recorded
- **Data:** pose rows + text-KL rows + tt-s3 conversation rows on rotating nodes (kl,conv,pose,pose)
- **Recipe:** prod_pose_s3.env, train_pose_s3q.py (pose 8B / 0.6B towers)
- **Steps:** 0 -> 392 (11379 resumed dcp-150)
- **Evals:** norm_word_ce = conv 0.0923 @300 (scripted tt rows); pose_cb0_ce = 2.9222 @300; pose_light_ce = 7.5741 @300; kl = 0.01517 @300
- **Checkpoints / exports:** w-392
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-pose-s3-j11379/w-392
- **HF repos:** none named in run files
- **Notes:** Pre-PPM pose experiments on the Qwen stage-3 trunk; superseded by PPM.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/prod_pose_s3.env:1 -> POSE-S3 multi-task finetune (fallback j8815 w-5900)`
  - `/data/home/amartyavarma/qwen-1ch/stage3/logs/q1ch-s3-11379.log:184 -> EVAL step 300 pose cb0 2.9222 light 7.5741 KL 0.01517`
  - `/data/home/amartyavarma/qwen-1ch/stage3/logs/q1ch-s3-11354.log -> preflight INIT (FT_FROM) = j8815 w-5900 FALLBACK (grep)`

### qwen1ch-pose-s3pv

**q32-pose-s3pv**  
group `pose` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 11421, 12207
- **Dates:** 2026-10-09 19:05 .. 2026-10-10 04:12 PT
- **Base model:** unknown
- **Init:** FT_FROM j8815 w-5900
- **Parents (resolved):** qwen1ch-s3-lr03@5900
- **Produces:** none recorded
- **Data:** as pose-s3 (pv variant)
- **Recipe:** prod_pose_s3.env + pv overrides
- **Steps:** 0 -> 470 (12207 resumed dcp-102)
- **Evals:** norm_word_ce = conv 2.0299 @450; pose_cb0_ce = 2.8045 @450; pose_light_ce = 7.1536 @450; kl = 0.01510 @450
- **Checkpoints / exports:** w-470
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-pose-s3pv-j12207/w-470
- **HF repos:** none named in run files
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/logs/q1ch-s3-12207.log:407 -> EVAL step 450 nword 2.0299 pose cb0 2.8045 light 7.1536`

### qwen1ch-pose-s3sam

**q32-pose-s3sam / s3sam311 (pose on SAM data)**  
group `pose` · status **abandoned** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 12433, 12486, 12530, 12546, 12547
- **Dates:** 2026-10-10 04:53 .. 07:01 PT
- **Base model:** unknown
- **Init:** FT_FROM j8815 w-5900
- **Parents (resolved):** qwen1ch-s3-lr03@5900
- **Produces:** none recorded
- **Data:** SAM pose clip set (pose-s3-samdata)
- **Recipe:** overrides_pose_sam*.env
- **Steps:** s3sam 0 -> 27; s3sam311 0 -> 50
- **Evals:** norm_word_ce = conv 2.0447 @25; pose_cb0_ce = 7.6229 @25; pose_light_ce = 7.2205 @25; kl = 0.01223 @25
- **Checkpoints / exports:** w-27
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-pose-s3sam-j12530/w-27
- **HF repos:** none named in run files
- **Notes:** Short restarts before s3sam311b.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/logs/q1ch-s3-12530.log:133790 -> EVAL step 25 pose cb0 7.6229`

### qwen1ch-pose-s3sam311b

**q32-pose-s3sam311b (GA=7 pose x5/conv/kl, SAM data)**  
group `pose` · status **stopped-with-save** · dir `/data/home/amartyavarma/qwen-1ch/stage3`

- **Jobs:** 12582, 12822, 12990
- **Dates:** 2026-10-10 07:01 .. 18:06 PT
- **Base model:** unknown
- **Init:** FT_FROM j8815 w-5900
- **Parents (resolved):** qwen1ch-s3-lr03@5900
- **Produces:** qwen1ch-pose-s3sam311b@696
- **Data:** SAM pose data + conv + text KL; GA 7 micros (pose x5, conv x1, kl x1)
- **Recipe:** prod_pose_s3.env + overrides_pose_sam311b.env; MAX_STEPS_S3 34422
- **Steps:** 0 -> 696 (planned 34,422)
- **Evals:** norm_word_ce = conv 2.0111 @650 (B 2.0156); pose_cb0_ce = 2.1625 @650; pose_light_ce = 5.6205 @650; kl = 0.01228 @650
- **Checkpoints / exports:** w-696; dcp-696
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage3/ckpt-weka/q32-pose-s3sam311b-j12990/w-696
- **HF repos:** none named in run files
- **Notes:** Last pre-PPM pose run; nodes then went to PPM (10-11).
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage3/runs/overrides_pose_sam311b.env:1 -> pose SAM-data restart fresh from j8815 w-5900; GA=7`
  - `/data/home/amartyavarma/qwen-1ch/stage3/logs/q1ch-s3-12582.log:9 -> FT_FROM j8815 w-5900, MAX_STEPS_S3 34422`
  - `/data/home/amartyavarma/qwen-1ch/stage3/logs/q1ch-s3-12990.log:72 -> EVAL step 650 nword 2.0111 pose cb0 2.1625 light 5.6205 KL 0.01228`
  - `memory:pose-podcast-model-plan.md:37 -> user calls q32-pose-s3sam311b 'half arsed', never a reference`

### qwen1ch-s4-adapter-asr

**q32-s4-adapter-asr (Voxtral adapter, user's ASR method)**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage4`

- **Jobs:** 9231, 9235, 9239, 9247, 9310, 9345
- **Dates:** 2026-10-07 13:35 .. 18:29 PT
- **Base model:** unknown
- **Init:** stage4/keep/s3-init-q32-s3-exp1-lr0.3-from750-j8815-w-5900 (frozen) + Voxtral-Mini-4B-Realtime encoder (frozen)
- **Parents (resolved):** qwen1ch-s3-lr03@5900, base:voxtral-mini-4b-realtime-2602
- **Produces:** qwen1ch-s4-adapter-asr@935
- **Data:** short A clips -> A's full transcript on B (MLS etc.)
- **Recipe:** S4_PHASE=adapter, adapter only trains, 8 nodes, eval every 250
- **Steps:** -> 935
- **Evals:** other = WER unknown at end
- **Checkpoints / exports:** w-935
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage4/ckpt-weka/q32-s4-adapter-asr-j9345/w-935
- **HF repos:** none named in run files
- **Notes:** Superseded by stage4-asr2 (gain/nonorm adapters).
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage4/RUNS.json:7 -> q32-s4-adapter-asr (user's method)`
  - `/data/home/amartyavarma/qwen-1ch/stage4/RUNS.json:21 -> init_from keep/s3-init-...-w-5900`
  - `memory:qwen-stage4-voxtral-plan.md:14 -> frozen Voxtral Realtime encoder latents via adapter`

### qwen1ch-s4-adapter-kl

**q32-s4-adapter-kl (Voxtral adapter, text-KL-teacher method)**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage4`

- **Jobs:** 9233, 9237, 9241, 9244, 9249, 9312, 9347
- **Dates:** 2026-10-07 13:46 .. 18:29 PT
- **Base model:** unknown
- **Init:** stage-3 j8815 w-5900 (frozen teacher + student) + Voxtral encoder
- **Parents (resolved):** qwen1ch-s3-lr03@5900, base:voxtral-mini-4b-realtime-2602
- **Produces:** qwen1ch-s4-adapter-kl@207, qwen1ch-s4-adapter-kl@526
- **Data:** 40 conv rows; KL of student hearing A via adapter vs teacher reading A's words
- **Recipe:** S4_PHASE=adapter_kl, KL on B text + cb0
- **Steps:** -> 207 (j9249), -> 526 (j9347)
- **Evals:** other = kl_gap_closed 0.891 @130 (j9249)
- **Checkpoints / exports:** w-207; w-526
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage4/ckpt-weka/q32-s4-adapter-kl-j9249/w-207; /data/home/amartyavarma/qwen-1ch/stage4/ckpt-weka/q32-s4-adapter-kl-j9347/w-526
- **HF repos:** none named in run files
- **Notes:** w-207 is the adapter used by the stage-4 validation run (RUNS.json:94).
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage4/RUNS.json:49 -> q32-s4-adapter-kl text-KL teacher method`
  - `/data/home/amartyavarma/qwen-1ch/stage4/runs/q32-s4-adapter-kl-j9249/metrics.jsonl:182 -> step 130 kl_gap_closed 0.891`

### qwen1ch-s4-val

**q32-s4-val-s3x5900 (stage-4 validation: stage-3 5900 continued as stage 4)**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage4`

- **Jobs:** 9267, 9401, 9406, 9409, 9411
- **Dates:** 2026-10-07 15:22 .. 21:08 PT
- **Base model:** unknown
- **Init:** stage-3 j8815 w-5900 + KL adapter w-207
- **Parents (resolved):** qwen1ch-s3-lr03@5900, qwen1ch-s4-adapter-kl@207
- **Produces:** qwen1ch-s4-val@6339
- **Data:** stage-3 rows/swap/persona/text-KL; A heard via Voxtral adapter, A codes undelayed
- **Recipe:** stage4 train_s4q.py; 9267 = 5900 -> 6150; 9409/9411 continue to 6339
- **Steps:** 5900 -> 6339
- **Evals:** norm_word_ce = B 2.0172 @6325; 2.0214 @6150; kl = 0.0140 (cited in s4-qqa STATUS)
- **Checkpoints / exports:** w-6339; dcp-6339
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage4/keep/s4-ft-init-q32-s4-val-s3x5900-j9411-w-6339
- **HF repos:** none named in run files
- **Notes:** Stage-3 at 5900 = 2.0018 on its own eval (decomp RESULTS.md: stage-3 reproduced 2.0018).
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage4/RUNS.json:94 -> validation: stage-3 dcp-5900 continued as stage 4 with KL adapter w-207 for 250 steps 5900->6150`
  - `/data/home/amartyavarma/qwen-1ch/stage4/runs/q32-s4-val-s3x5900-j9267/metrics.jsonl:262 -> step 6150 B 2.0214`
  - `/data/home/amartyavarma/qwen-1ch/stage4/runs/q32-s4-val-s3x5900-j9411/metrics.jsonl:184 -> step 6325 B 2.0172`
  - `/data/home/amartyavarma/qwen-1ch/stage4/logs/q1ch-s4-9411.log:1040 -> saved dcp-6339 + w-6339`
  - `/data/home/amartyavarma/s4-qqa/STATUS.md:41 -> j9411 eval KL 0.0140`

### qwen1ch-s4-val-kl10

**q32-s4-val-kl10 / q32-s4-val-bs4x (stage-4 validation variants)**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage4`

- **Jobs:** 9414, 9453
- **Dates:** 2026-10-07 21:09 .. 23:08 PT
- **Base model:** unknown
- **Init:** stage-3 w-5900 + KL adapter (branch of validation)
- **Parents (resolved):** qwen1ch-s3-lr03@5900, qwen1ch-s4-adapter-kl@207
- **Produces:** none recorded
- **Data:** as validation
- **Recipe:** kl10 (W_KL 10) and bs4x (4x batch) variants
- **Steps:** kl10 -> 6250; bs4x -> 6204
- **Evals:** norm_word_ce = kl10 B 2.0205 @6250; bs4x B 2.0074 @6204
- **Checkpoints / exports:** unknown
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage4/ckpt-weka/q32-s4-val-bs4x-j9453
- **HF repos:** none named in run files
- **Notes:** Grouped short variants.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage4/runs/q32-s4-val-kl10-j9414/metrics.jsonl:105 -> step 6250 B 2.0205`
  - `/data/home/amartyavarma/qwen-1ch/stage4/runs/q32-s4-val-bs4x-j9453/metrics.jsonl:64 -> step 6204 B 2.0074`

### qwen1ch-s4-ft-hovfix

**q32-s4-ft-hov(fix)-from9411w6339 (Model 2: one-sided stage-4 FT on handover none bucket)**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage4`

- **Jobs:** 9977, 10006, 10139, 10242
- **Dates:** 2026-10-08 19:48 .. 23:52 PT
- **Base model:** unknown
- **Init:** FT_FROM stage4/keep/s4-ft-init-q32-s4-val-s3x5900-j9411-w-6339 (fresh Adam)
- **Parents (resolved):** qwen1ch-s4-val@6339
- **Produces:** qwen1ch-s4-ft-hovfix@731
- **Data:** stage-2 handover 'none' bleed bucket (9,012 conversations, 1,421.6 h), symmetric (each chunk twice), B only target, A via Voxtral
- **Recipe:** prod_s4_hov_ft.env; 9977 = first hov launch (-> 125); hovfix 10006 -> 10139 -> 10242 (resume dcp-600 -> 731)
- **Steps:** 0 -> 731
- **Evals:** norm_word_ce = B 1.8969 @731 (own eval); hov 1.9559 @125; kl = 0.0204 at its w-731 (s4-qqa step-0 eval)
- **Checkpoints / exports:** w-731
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage4/keep/s4-ft-init-q32-s4-ft-hovfix-from9411w6339-j10242-w-731
- **HF repos:** none named in run files
- **Notes:** w-731 = init of the stage-4 gen FT and every quick-QA (s4-qqa) run.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage4/prod_s4_hov_ft.env:1 -> MODEL 2: one-sided stage-4 FT on handover none bucket`
  - `/data/home/amartyavarma/qwen-1ch/stage4/logs/q1ch-s4-9977.log:9 -> FINETUNE INIT <- s4-ft-init-...-j9411-w-6339`
  - `/data/home/amartyavarma/qwen-1ch/stage4/logs/q1ch-s4-10242.log:12 -> full resume from j10139 dcp-600`
  - `/data/home/amartyavarma/qwen-1ch/stage4/runs/q32-s4-ft-hovfix-from9411w6339-j10242/metrics.jsonl:139 -> step 731 B 1.8969`
  - `/data/home/amartyavarma/s4-qqa/STATUS.md:29 -> init 10242 w-731 text KL 0.0204`

### qwen1ch-s4-ft-gen

**q32-s4-ft-gen-from10242w731 (stage-4 FT on generated QA speech)**  
group `other` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage4`

- **Jobs:** 13783, 13858
- **Dates:** 2026-10-10 20:35 .. 21:43 PT
- **Base model:** unknown
- **Init:** FT_FROM j10242 w-731
- **Parents (resolved):** qwen1ch-s4-ft-hovfix@731
- **Produces:** qwen1ch-s4-ft-gen@306
- **Data:** qa25k-speech generated calls (24,494 train rows), A via Voxtral, B target
- **Recipe:** prod_s4_gen_ft.env
- **Steps:** 0 -> 306
- **Evals:** norm_word_ce = B 1.2864 @306
- **Checkpoints / exports:** w-306
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage4/ckpt-weka/q32-s4-ft-gen-from10242w731-j13858/w-306
- **HF repos:** none named in run files
- **Notes:** 13783 cancelled at 25.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage4/prod_s4_gen_ft.env:1 -> QA 25k calls only`
  - `/data/home/amartyavarma/qwen-1ch/stage4/logs/q1ch-s4-13858.log:10 -> FINETUNE INIT <- j10242 w-731`
  - `/data/home/amartyavarma/qwen-1ch/stage4/runs/q32-s4-ft-gen-from10242w731-j13858/metrics.jsonl:320 -> step 306 B 1.2864`

### qqa-run-a

**s4-qqa run A (q32-s4-ft-qqa-from10242w731)**  
group `other` · status **finished** · dir `/data/home/amartyavarma/s4-qqa`

- **Jobs:** 21103
- **Dates:** 2026-10-14 22:42 .. 23:30 PT
- **Base model:** unknown
- **Init:** FT_FROM j10242 w-731
- **Parents (resolved):** qwen1ch-s4-ft-hovfix@731
- **Produces:** none recorded
- **Data:** quick-qa-100k-tts-ccby, corrupt-A track, packed (~17 conversations per micro), 1 epoch
- **Recipe:** stage4 prod_s4_qqa.env (stage-4 mode: A via KL adapter + A Mimi), 223 steps
- **Steps:** 0 -> 223
- **Evals:** norm_word_ce = B 1.5865 @223 (96 held-out quick-qa conversations)
- **Checkpoints / exports:** w-223
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage4/ckpt-weka/q32-s4-ft-qqa-from10242w731-j21103
- **HF repos:** [InternalCan/q32-s4-ft-qqa-j21103](https://huggingface.co/InternalCan/q32-s4-ft-qqa-j21103)
- **Notes:** Quick-QA family: runs A-H share data/init; results table at /data/home/amartyavarma/s4-qqa/STATUS.md:76.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage4/prod_s4_qqa.env:1 -> quick-qa-100k-tts-ccby, corrupt A, 1 epoch, packed`
  - `/data/home/amartyavarma/qwen-1ch/stage4/runs/q32-s4-ft-qqa-from10242w731-j21103/metrics.jsonl:233 -> step 223 B 1.5865`
  - `/data/home/amartyavarma/s4-qqa/STATUS.md:6 -> HF retry 21124 (run A, blocked by org storage)`

### qqa-run-b

**s4-qqa run B (q32-s4-ft-qqa-asrfrz)**  
group `other` · status **abandoned** · dir `/data/home/amartyavarma/s4-qqa`

- **Jobs:** 21205
- **Dates:** 2026-10-15 01:37 .. (cancelled) PT
- **Base model:** unknown
- **Init:** FT_FROM j10242 w-731 + ASR adapter (frozen)
- **Parents (resolved):** qwen1ch-s4-ft-hovfix@731, qwen1ch-s4-asr2-ga1@3695
- **Produces:** none recorded
- **Data:** as run A
- **Recipe:** ASR-adapter init, frozen
- **Steps:** 0 -> 75
- **Evals:** norm_word_ce = B 1.7595 @75
- **Checkpoints / exports:** unknown
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Notes:** Cancelled (log ends with CANCELLED).
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage4/runs/q32-s4-ft-qqa-asrfrz-from10242w731-j21205/metrics.jsonl:79 -> step 75 B 1.7595`

### qqa-run-c

**s4-qqa run C (q32-s4-ft-qqa-s3text: stage-3 style A text + A codes, B-only loss)**  
group `other` · status **finished** · dir `/data/home/amartyavarma/s4-qqa`

- **Jobs:** 21169
- **Dates:** 2026-10-15 00:55 .. 01:35 PT
- **Base model:** unknown
- **Init:** FT_FROM j10242 w-731
- **Parents (resolved):** qwen1ch-s4-ft-hovfix@731
- **Produces:** none recorded
- **Data:** as run A
- **Recipe:** S4_A_FEED stage-3 text mode (= 'speech duplex FT' per memory)
- **Steps:** 0 -> 223
- **Evals:** norm_word_ce = B 1.3687 @223 (2.2205 @0); kl = 0.0251 at w-223 (forgetting check)
- **Checkpoints / exports:** w-223
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage4/ckpt-weka/q32-s4-ft-qqa-s3text-from10242w731-j21169
- **HF repos:** none named in run files
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage4/runs/q32-s4-ft-qqa-s3text-from10242w731-j21169/metrics.jsonl:233 -> step 223 B 1.3687`
  - `/data/home/amartyavarma/s4-qqa/STATUS.md:99 -> run C curve 2.2205 -> 1.3688`
  - `/data/home/amartyavarma/s4-qqa/STATUS.md:30 -> run C w-223 text KL 0.0251`
  - `memory:speech-duplex-means-stage3.md:11 -> run C = stage-3 format`

### qqa-run-d

**s4-qqa run D (s3textnc: pure text duplex)**  
group `other` · status **finished** · dir `/data/home/amartyavarma/s4-qqa`

- **Jobs:** 21276
- **Dates:** 2026-10-15 03:59 .. 04:31 PT
- **Base model:** unknown
- **Init:** FT_FROM j10242 w-731
- **Parents (resolved):** qwen1ch-s4-ft-hovfix@731
- **Produces:** none recorded
- **Data:** run C rows with all Mimi codes removed
- **Recipe:** S4_A_FEED=s3text_nocodes, loss on B text only
- **Steps:** 0 -> 223
- **Evals:** norm_word_ce = B 1.3634 @223 (5.9937 @0)
- **Checkpoints / exports:** w-223
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage4/runs/q32-s4-ft-qqa-s3textnc-from10242w731-j21276/metrics.jsonl:233 -> step 223 B 1.3634`
  - `/data/home/amartyavarma/s4-qqa/STATUS.md:89 -> RUN D = run C as pure text duplex`

### qqa-run-e

**s4-qqa run E (noamimi: Voxtral only on A, KL adapter)**  
group `other` · status **finished** · dir `/data/home/amartyavarma/s4-qqa`

- **Jobs:** 21302
- **Dates:** 2026-10-15 05:10 .. 05:49 PT
- **Base model:** unknown
- **Init:** FT_FROM j10242 w-731
- **Parents (resolved):** qwen1ch-s4-ft-hovfix@731
- **Produces:** qqa-run-e@223
- **Data:** as run A
- **Recipe:** S4_A_MIMI=0
- **Steps:** 0 -> 223
- **Evals:** norm_word_ce = B 1.5366 @223 (2.4235 @0)
- **Checkpoints / exports:** w-223
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage4/ckpt-weka/q32-s4-ft-qqa-noamimi-from10242w731-j21302
- **HF repos:** none named in run files
- **Citations:**
  - `/data/home/amartyavarma/s4-qqa/STATUS.md:8 -> RUN E = run A with NO user Mimi codes`
  - `/data/home/amartyavarma/s4-qqa/STATUS.md:12 -> step-0 eval 2.4235`
  - `/data/home/amartyavarma/qwen-1ch/stage4/runs/q32-s4-ft-qqa-noamimi-from10242w731-j21302/metrics.jsonl:233 -> step 223 B 1.5366`

### qqa-run-f

**s4-qqa run F (noamimi-asrinit: ASR adapter init, trainable)**  
group `other` · status **finished** · dir `/data/home/amartyavarma/s4-qqa`

- **Jobs:** 21455
- **Dates:** 2026-10-15 09:51 .. 10:31 PT
- **Base model:** unknown
- **Init:** FT_FROM j10242 w-731; adapter <- stage4-asr2 j19174 w-3695
- **Parents (resolved):** qwen1ch-s4-ft-hovfix@731, qwen1ch-s4-asr2-ga1@3695
- **Produces:** none recorded
- **Data:** as run A
- **Recipe:** run E + ASR adapter init (trainable)
- **Steps:** 0 -> 223
- **Evals:** norm_word_ce = B 1.4770 @223 (2.8002 @0); kl = 0.0255 at w-223
- **Checkpoints / exports:** w-223
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Citations:**
  - `/data/home/amartyavarma/s4-qqa/STATUS.md:50 -> run F = run E + ASR adapter init j19174 w-3695`
  - `/data/home/amartyavarma/qwen-1ch/stage4/runs/q32-s4-ft-qqa-noamimi-asrinit-from10242w731-j21455/metrics.jsonl:233 -> step 223 B 1.4770`
  - `/data/home/amartyavarma/s4-qqa/STATUS.md:32 -> run F w-223 text KL 0.0255`

### qqa-run-g

**s4-qqa run G (s3text-kl: run C + text-KL co-training)**  
group `other` · status **finished** · dir `/data/home/amartyavarma/s4-qqa`

- **Jobs:** 21443
- **Dates:** 2026-10-15 08:55 .. 09:48 PT
- **Base model:** unknown
- **Init:** FT_FROM j10242 w-731
- **Parents (resolved):** qwen1ch-s4-ft-hovfix@731
- **Produces:** none recorded
- **Data:** run C rows + text-KL rows (keys 40255.., cache 17afc394d0bffd2c)
- **Recipe:** W_KL 30, S4_KL_NC=speech
- **Steps:** 0 -> 223
- **Evals:** norm_word_ce = B 1.3729 @223; kl = 0.0204 -> 0.0135 @223
- **Checkpoints / exports:** w-223
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Citations:**
  - `/data/home/amartyavarma/s4-qqa/STATUS.md:42 -> run G B 2.2205 -> 1.3729; KL 0.0204 -> 0.0135`
  - `/data/home/amartyavarma/qwen-1ch/stage4/runs/q32-s4-ft-qqa-s3text-kl-from10242w731-j21443/metrics.jsonl:233 -> step 223 B 1.3729`

### qqa-run-h

**s4-qqa run H (noamimi-asrinit-kl: run F + text KL)**  
group `other` · status **finished** · dir `/data/home/amartyavarma/s4-qqa`

- **Jobs:** 21493
- **Dates:** 2026-10-15 11:06 .. 11:58 PT
- **Base model:** unknown
- **Init:** FT_FROM j10242 w-731 + ASR adapter j19174 w-3695
- **Parents (resolved):** qwen1ch-s4-ft-hovfix@731, qwen1ch-s4-asr2-ga1@3695
- **Produces:** none recorded
- **Data:** as run F + run G's text KL
- **Recipe:** run F + KL co-training
- **Steps:** 0 -> 223
- **Evals:** norm_word_ce = B 1.4908 @223; kl = 0.0204 -> 0.0136
- **Checkpoints / exports:** w-223
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Citations:**
  - `/data/home/amartyavarma/s4-qqa/STATUS.md:24 -> run H B 2.8002 -> 1.4908; KL 0.0204 -> 0.0136`
  - `/data/home/amartyavarma/qwen-1ch/stage4/runs/q32-s4-ft-qqa-noamimi-asrinit-kl-from10242w731-j21493/metrics.jsonl:233 -> step 223 B 1.4908`

### qqa-scaling

**s4-qqa scaling runs (noamimi-scale, s3text-scale, s3textnc-scale, noamimi-asrinit-scale)**  
group `other` · status **finished** · dir `/data/home/amartyavarma/s4-qqa`

- **Jobs:** 21335, 21373, 21504, 21734
- **Dates:** 2026-10-15 05:52 .. 20:56 PT
- **Base model:** unknown
- **Init:** FT_FROM j10242 w-731
- **Parents (resolved):** qwen1ch-s4-ft-hovfix@731
- **Produces:** none recorded
- **Data:** as run A; exports at 14/28/59/117/223
- **Recipe:** identical to runs E / C / D / F except export_steps / keep_steps
- **Steps:** 0 -> 223 each
- **Evals:** norm_word_ce = B @223: noamimi 1.5573, s3text 1.3633, s3textnc 1.3655, asrinit 1.4793
- **Checkpoints / exports:** w-14; w-28; w-59; w-117; w-223 per run
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage4/ckpt-weka/q32-s4-ft-qqa-s3text-scale-from10242w731-j21373
- **HF repos:** none named in run files
- **Notes:** Grouped. Run-to-run noise ~0.02 (STATUS.md:74).
- **Citations:**
  - `/data/home/amartyavarma/s4-qqa/SCALING_JOBS.txt:1 -> scaling jobs and exports 14,28,59,117`
  - `/data/home/amartyavarma/s4-qqa/STATUS.md:71 -> stage-3 (run-C mode) 21373: 2.2205 -> 1.3633`
  - `/data/home/amartyavarma/s4-qqa/STATUS.md:70 -> text duplex 21504 -> 1.3656`
  - `/data/home/amartyavarma/s4-qqa/STATUS.md:69 -> ASR adapter 21734 -> 1.4793`
  - `/data/home/amartyavarma/qwen-1ch/stage4/runs/q32-s4-ft-qqa-noamimi-scale-from10242w731-j21335/metrics.jsonl:237 -> step 223 B 1.5573`
  - `memory:project-qqa-overnight-1015.md:12 -> scaling run 21373 saves at 14/28/59/117/223`

### qwen1ch-s4-asr2-ga1

**q32-s4-asr2-ga1 (ASR adapter v2, adapter-only on frozen stage-3 trunk)**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage4-asr2`

- **Jobs:** 19161, 19163, 19169, 19174, 19253
- **Dates:** 2026-10-12 06:41 .. 16:31 PT
- **Base model:** unknown
- **Init:** stage-3 j8815 w-5900 (frozen) + Voxtral
- **Parents (resolved):** qwen1ch-s3-lr03@5900, base:voxtral-mini-4b-realtime-2602
- **Produces:** qwen1ch-s4-asr2-ga1@3695
- **Data:** ASR adapter task (A clip -> transcript on B); WER on 5180 MLS words
- **Recipe:** launch_asr2_ga1.sh, trainable vadapter only (62.9M params); 19161 = stream eval
- **Steps:** 0 -> 3696
- **Evals:** kl = B-KL vs teacher 0.20057 @3695; other = WER 0.321, clip transcript word CE 0.4143 @3695
- **Checkpoints / exports:** w-3600; w-3695/w-3696
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage4-asr2/ckpt-weka/q32-s4-asr2-ga1-j19253/w-3696; /data/home/amartyavarma/qwen-1ch/stage4-asr2/ckpt-weka/q32-s4-asr2-ga1-j19174
- **HF repos:** none named in run files
- **Notes:** Referenced as 'j19174 w-3695' by s4-qqa runs B/F/H. Decomp: ASR adapter 0.53 nats worse than muting A on B word CE (stage4-decomp/RESULTS.md).
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage4-asr2/logs/s4asr2-19253.log:7 -> trainable vadapter only, trunk FROZEN`
  - `/data/home/amartyavarma/qwen-1ch/stage4-asr2/logs/s4asr2-19253.log:19 -> EVAL step 3695 WER 0.321, clip word CE 0.4143, B-KL 0.20057`
  - `/data/home/amartyavarma/qwen-1ch/stage4-asr2/logs/s4asr2-19253.log:30 -> saved dcp-3696 + w-3696`

### qwen1ch-s4-asr2-gain

**q32-s4-asr2-gain / nonorm (no-RMSNorm adapter variants)**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/stage4-asr2`

- **Jobs:** 19379, 19399, 19404
- **Dates:** 2026-10-12 17:31 .. 18:50 PT
- **Base model:** unknown
- **Init:** stage-3 w-5900 frozen + adapter continuing from asr2 (ga1)
- **Parents (resolved):** qwen1ch-s4-asr2-ga1@3695
- **Produces:** qwen1ch-s4-asr2-nonorm@3995
- **Data:** as asr2-ga1
- **Recipe:** gain (learnable gain, 19399 cancelled at 3700) and nonorm (19404 -> 3995)
- **Steps:** -> 3995 (nonorm)
- **Evals:** kl = B-KL 5.29895 @3995 (nonorm); other = WER 0.459 @3995 (nonorm)
- **Checkpoints / exports:** w-3995
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage4-asr2/ckpt-weka/q32-s4-asr2-nonorm-j19404; /data/home/amartyavarma/qwen-1ch/stage4-asr2/ckpt-weka/q32-s4-asr2-gain-j19399
- **HF repos:** none named in run files
- **Notes:** Parent assumption: continue from ga1 (step numbers continue past 3695); not verified line-by-line.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage4-asr2/logs/s4asr2-19404.log:357 -> EVAL step 3995 WER 0.459, B-KL 5.29895`
  - `/data/home/amartyavarma/qwen-1ch/stage4-asr2/logs/s4asr2-19404.log:365 -> saved w-3995`

### qwen1ch-s4-adonly

**q32-s4-adonly-asr3695 / full-asr3695 / noamimi-adonly (stage-4 with ASR adapter, Oct 12)**  
group `base-stage` · status **abandoned** · dir `/data/home/amartyavarma/qwen-1ch/stage4-adonly`

- **Jobs:** 19262, 19263, 19267, 19287, 19277, 19278, 19299, 19408
- **Dates:** 2026-10-12 16:38 .. 2026-10-13 02:36 PT
- **Base model:** unknown
- **Init:** stage-3 j8815 w-5900 + ASR adapter asr3695
- **Parents (resolved):** qwen1ch-s3-lr03@5900, qwen1ch-s4-asr2-ga1@3695
- **Produces:** none recorded
- **Data:** stage-3 podcast rows only (S3_KL=0); 304 eval rows
- **Recipe:** adonly = adapter-only on frozen trunk (stage4-adonly); full = backbone unfrozen, 5900 -> 6900 planned (stage4-full j19299, killed); noamimi-adonly = no A Mimi, gain adapter (stage4-noamimi j19408)
- **Steps:** adonly 5900 -> 5945; noamimi 5900 -> 6000; full killed at ~5900
- **Evals:** other = raw B word CE: adonly 2.9289 @5900; noamimi 7.6021 @6000
- **Checkpoints / exports:** w-5945 (adonly); w-6000 (noamimi)
- **Local paths:** /data/home/amartyavarma/qwen-1ch/stage4-adonly/ckpt-weka/q32-s4-adonly-asr3695-j19267; /data/home/amartyavarma/qwen-1ch/stage4-noamimi/ckpt-weka/q32-s4-noamimi-adonly-j19408
- **HF repos:** none named in run files
- **Notes:** Grouped Oct-12 stage-4 attempts. NOAMIMI.md:1 says 'teed up, NOT launched' but j19408 log shows it ran to 6000 then cancelled. Stage-4 decomposition (stage4-decomp, eval-only) not a training run.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/stage4-adonly/logs/adonly-19267.log:57 -> EVAL step 5900 B word 2.9289`
  - `/data/home/amartyavarma/qwen-1ch/stage4-adonly/logs/adonly-19267.log:107 -> saved w-5945`
  - `/data/home/amartyavarma/qwen-1ch/stage4-noamimi/NOAMIMI.md:6 -> copy of stage4-full, code of the killed run 19299`
  - `/data/home/amartyavarma/qwen-1ch/stage4-noamimi/logs/noamimi-19408.log:161 -> EVAL step 6000 B word 7.6021`
  - `memory:qwen-stage4-voxtral-plan.md:45 -> next planned run: no A Mimi, adapter without RMS norm`

### hera2-ft

**hera2-6974 fine-tunes (ft-hov-sys75, ft-hov-speech8, ft-hov-vp, ft-hov-vp8)**  
group `other` · status **finished** · dir `/data/home/amartyavarma/qwen-1ch/hera-ft`

- **Jobs:** 10219, 10252, 10289, 10306, 10471, 10492, 13398, 13515, 13651
- **Dates:** 2026-10-08 23:09 .. 2026-10-10 20:16 PT (launch times)
- **Base model:** unknown
- **Init:** hera2 step 6974 (/scratch/amartyavarma/hera2-6974) + Kalliope adapter6974
- **Parents (resolved):** ext:hera2@6974
- **Produces:** hera2-ft-hov-sys75@176, hera2-ft-hov-speech8@365, hera2-ft-hov-vp8@365
- **Data:** stage-2 handover (hov) data; sys75 / speech8 / vp (voice prompt) variants
- **Recipe:** hera code (32B duplex context model + sam 8B + 0.6B light; Voxtral on A), hera_ft.sbatch / hera_ft8.sbatch
- **Steps:** sys75 -> 176; speech8 -> 365; vp8 -> 365
- **Evals:** other = unknown (no eval lines found in logs)
- **Checkpoints / exports:** export-hf/hera2-6974-ft-hov-j10306-step176-bf16; export-hf/hera2-6974-ft-hov-speech8-j10492-step365-bf16; export-hf/hera2-6974-ft-hov-vp8-j13651-step365-bf16
- **Local paths:** /data/home/amartyavarma/qwen-1ch/hera-ft/export-hf; /data/home/amartyavarma/qwen-1ch/hera-ft/ckpt-weka
- **HF repos:** none named in run files
- **Notes:** hera2 is an external/colleague model (reference ~/ref/hera, adapter InternalCan/hera3-adapter-v1 per memory:qwen-stage4-voxtral-plan.md:38); its own pretraining is not ours. Grouped for breadth.
- **Citations:**
  - `/data/home/amartyavarma/qwen-1ch/hera-ft/runs/launches.txt:1 -> 10219 hera-ft-smoke 2026-10-08 23:09`
  - `/data/home/amartyavarma/qwen-1ch/hera-ft/runs/launches.txt:8 -> 13651 hera2-6974-ft-hov-vp8 2026-10-10 20:16`
  - `/data/home/amartyavarma/qwen-1ch/hera-ft/hera_ft.sbatch:59 -> INIT_SAM=/scratch/amartyavarma/hera2-6974`
  - `/data/home/amartyavarma/qwen-1ch/hera-ft/hera_ft.sbatch:60 -> KALLIOPE_ADAPTER=qwen-1ch-data/hera-ft-init/adapter6974`
  - `/data/home/amartyavarma/qwen-1ch/hera-ft/code/README.md:3 -> hera2 full-duplex model description`

### mistral-s1-2lane

**Mistral-Medium-3.5-128B stage 1 two-lane s1kl0 runs (LR sweep + bal/tc runs)**  
group `base-stage` · status **abandoned** · dir `/data/home/amartyavarma/mistral-distill-bf16/train`

- **Jobs:** 46, 60, 77, 91, 105, 110, 111, 112, 126, 127, 153, 190, 281, 282, 295, 312, 467, 469, 482, 519
- **Dates:** 2026-09-30 12:34 .. 2026-10-01 05:13 PT
- **Base model:** mistral-medium-3.5-128b (models/mistral-m35-128b-bf16)
- **Init:** stock Mistral-Medium-3.5-128B bf16
- **Parents (resolved):** base:mistral-medium-3.5-128b
- **Produces:** mistral-s1-2lane@500, mistral-s1-2lane@393
- **Data:** corpus text-KL vs frozen stock Mistral teacher on its chat template (T19 config: LOSS_PAD none, NW 0, </s> as lane-B target)
- **Recipe:** train_s1m.py, 12 nodes / 96 ranks, 192 rows/step, epoch 6865 steps; LRs 2e-6 (j111), bench 1e-6 / 5e-7 / 2.5e-7 (j467/469/482), 5e-7 (j519)
- **Steps:** j519 -> 500 (eval 490); j111 -> 393
- **Evals:** kl = j519 0.04085 @490; j111 0.07303 @392
- **Checkpoints / exports:** j519 w-500; j111 w-393
- **Local paths:** unknown
- **HF repos:** [InternalCan/mistral-1ch-s1-ckpts](https://huggingface.co/InternalCan/mistral-1ch-s1-ckpts)
- **Notes:** Superseded by the 1ch format (j709). The Mistral base node id is a guess at the canonical spelling.
- **Citations:**
  - `/data/home/amartyavarma/mistral-distill-bf16/README.md:1 -> Mistral-Medium-3.5-128B stage-1 text-KL distillation (bf16)`
  - `/data/home/amartyavarma/mistral-distill-bf16/README.md:12 -> T19 config`
  - `/data/home/amartyavarma/mistral-distill-bf16/train/logs/s1m-519.log:1039 -> EVAL step 490 kl 0.04085`
  - `/data/home/amartyavarma/mistral-distill-bf16/train/logs/s1m-111.log:98857 -> EVAL step 392 kl 0.07303`
  - `/data/home/amartyavarma/hf-backup-staging/readme/mistral-1ch-s1-ckpts/README.md:12 -> j91..j519 earlier stage-1 runs in pre-1ch format (LR sweeps)`
  - `/data/home/amartyavarma/hf-backup-staging/readme/mistral-1ch-s1-ckpts/README.md:22 -> j519-w-500-bf16`

### mistral-t19-handoff

**Mistral T19 stage-1 runs from the handoff (j21278 w-100, j21338 w-45, fp8 j21329)**  
group `base-stage` · status **unknown** · dir `/data/home/amartyavarma/mistral-distill-bf16 (handoff docs; hera3s2/mistral paths on prior cluster)`

- **Jobs:** 21222, 21278, 21329, 21338 (prior cluster)
- **Dates:** before 2026-09-30 (prior cluster)
- **Base model:** mistral-medium-3.5-128b
- **Init:** stock Mistral-Medium-3.5-128B
- **Parents (resolved):** base:mistral-medium-3.5-128b
- **Produces:** none recorded
- **Data:** as mistral-s1-2lane
- **Recipe:** T19 config, 8 nodes x 8 H200 = 64 ranks, cap70/cap75
- **Steps:** 21278 -> 100; 21338 -> 45; 21329 (fp8) -> 58
- **Evals:** none recorded
- **Checkpoints / exports:** w-100; dcp-100; dcp-45; w-45
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Notes:** Prior-cluster job ids; never extended here. fp8 is forbidden by the user.
- **Citations:**
  - `/data/home/amartyavarma/mistral-distill-bf16/HANDOFF.md:25 -> fp8 run 21329 killed`
  - `/data/home/amartyavarma/mistral-distill-bf16/HANDOFF.md:68 -> 21278 bf16 export w-100 (Weka-only), the served T19 model`
  - `/data/home/amartyavarma/mistral-distill-bf16/HANDOFF.md:70 -> 21338 dcp-45 + w-45`

### mistral-s1-1ch

**Mistral 1ch stage 1 (m35-s1-1ch-none-eos-bal-lr5e-7-r192-w96, job 709)**  
group `base-stage` · status **stopped-with-save** · dir `/data/home/amartyavarma/mistral-distill-bf16/train1ch`

- **Jobs:** 709
- **Dates:** 2026-09-30 22:21 .. 2026-10-01 08:07 PT (cancelled after save)
- **Base model:** mistral-medium-3.5-128b
- **Init:** stock Mistral-Medium-3.5-128B (/scratch/amartyavarma/mistral-m35-128b-bf16)
- **Parents (resolved):** base:mistral-medium-3.5-128b
- **Produces:** mistral-s1-1ch@531
- **Data:** corpus text-KL in single-channel format (SOA/EOA/SOB, </s> EOB), 192 rows/step
- **Recipe:** train1ch/train_1ch.py, 12 nodes / 96 ranks, LR 5e-7, BAL, max_steps/horizon 6865, stock RoPE
- **Steps:** 0 -> 531 (horizon 6865)
- **Evals:** genghisbench = 20/20, median 11 questions (v4 oracle); kl = 0.04095 @490
- **Checkpoints / exports:** w-531; dcp-531
- **Local paths:** /data/home/amartyavarma/mistral-distill-bf16/train1ch/ckpt-weka/m35-s1-1ch-none-eos-bal-lr5e-7-r192-w96-j709/w-531; /data/home/amartyavarma/ckpt-backup/m35-s1-1ch-none-eos-bal-lr5e-7-r192-w96-j709
- **HF repos:** [InternalCan/mistral-1ch-s1-ckpts](https://huggingface.co/InternalCan/mistral-1ch-s1-ckpts)
- **Notes:** Template for the Qwen clean-1ch redo (memory:qwen-clean-1ch-plan.md:11).
- **Citations:**
  - `/data/home/amartyavarma/mistral-distill-bf16/train1ch/logs/s1m-1ch-709.log:1 -> job 709, 12 nodes, world 96, 2026-09-30 22:21:46`
  - `/data/home/amartyavarma/mistral-distill-bf16/train1ch/logs/s1m-1ch-709.log:547 -> EVAL step 490 kl 0.04095`
  - `/data/home/amartyavarma/mistral-distill-bf16/train1ch/logs/s1m-1ch-709.log:590 -> saved dcp-531 + Weka w-531`
  - `/data/home/amartyavarma/mistral-distill-bf16/train1ch/runs/cfg-709.json:7 -> max_steps 6865; :8 horizon 6865`
  - `memory:genghisbench.md:15 -> Mistral stage-1 end (j709 w-531): 20/20, median 11`
  - `/data/home/amartyavarma/hf-backup-staging/readme/mistral-1ch-s1-ckpts/README.md:11 -> j709 = 1ch stage-1 run, w-531 init of all Mistral stage-2 runs, GB 20/20`
  - `memory:qwen-mixrope-mistral-stock.md:11 -> Mistral uses stock RoPE`

### mistral-s2-1ch-early

**Mistral stage 2 1ch early run (m35-s2-1ch-tim1-s100, j2959)**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/mistral-distill-bf16/stage2_1ch`

- **Jobs:** 2835, 2933, 2959
- **Dates:** 2026-10-01 08:44 .. 10:17 PT
- **Base model:** unknown
- **Init:** j709 w-531
- **Parents (resolved):** mistral-s1-1ch@531
- **Produces:** none recorded
- **Data:** tts-pretrain-128m speech (5120-frame rows) + klpre-split1 text KL
- **Recipe:** stage2_1ch/train_s2_1ch.py; 100 steps
- **Steps:** 0 -> 100
- **Evals:** kl = 0.03740 @100
- **Checkpoints / exports:** w-100
- **Local paths:** /data/home/amartyavarma/mistral-distill-bf16/stage2_1ch/ckpt-weka/m35-s2-1ch-tim1-s100-j2959/w-100
- **HF repos:** none named in run files
- **Notes:** nword not logged in this early version.
- **Citations:**
  - `/data/home/amartyavarma/mistral-distill-bf16/stage2_1ch/logs/s2-1ch-2959.log:154 -> EVAL step 100 kl 0.03740`
  - `/data/home/amartyavarma/hf-backup-staging/readme/mistral-1ch-s2-ckpts/README.md:12 -> j2959 is an early 100-step run`
  - `memory:qwen-clean-1ch-plan.md:21 -> Mistral stage 2 stays at ROW_FRAMES 5120`

### mistral-s2-kl20

**Mistral stage 2 kl20 lr2e-5 (m35-s2-1ch-k8lf05-fce-ln2e-5-kl20-tim1-lr2e-5, j7434)**  
group `base-stage` · status **abandoned** · dir `/data/home/amartyavarma/mistral-distill-bf16/stage2_1ch`

- **Jobs:** 7391, 7393, 7406, 7421, 7425, 7434
- **Dates:** 2026-10-03 02:57 .. 05:26 PT
- **Base model:** unknown
- **Init:** j709 w-531 + Ministral-3-8B voice tower
- **Parents (resolved):** mistral-s1-1ch@531, base:ministral-3-8b-base-2512
- **Produces:** none recorded
- **Data:** tts-pretrain-128m speech + klpre-split1 text KL
- **Recipe:** K=8 light sub-steps, light_frac 0.5, factorised CE, light LR 2e-5, W_KL 20, trunk LR 2e-5; 7393-7425 abandoned launches
- **Steps:** 0 -> 101
- **Evals:** norm_word_ce = 2.7342 @100; kl = 0.09480 @100
- **Checkpoints / exports:** w-101
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Notes:** KL too high; replaced by lr5e-6 runs.
- **Citations:**
  - `/data/home/amartyavarma/mistral-distill-bf16/stage2_1ch/logs/s2-1ch-7434.log:165 -> EVAL step 100 KL 0.09480 nword 2.7342`
  - `/data/home/amartyavarma/mistral-distill-bf16/stage2_1ch/GENGHIS_LOG.txt:1 -> earlier launches j7393/j7406/j7421/j7425 abandoned`
  - `memory:mistral-s2-decision-criteria.md:11 -> Mistral stage-2 launched 10-03 as job 7421`

### mistral-s2-lr5e6

**Mistral stage 2 lr5e-6 kl20 (j7454)**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/mistral-distill-bf16/stage2_1ch`

- **Jobs:** 7454
- **Dates:** 2026-10-03 05:36 .. 08:30 PT
- **Base model:** unknown
- **Init:** j709 w-531
- **Parents (resolved):** mistral-s1-1ch@531
- **Produces:** none recorded
- **Data:** as above
- **Recipe:** trunk LR 5e-6, W_KL 20
- **Steps:** 0 -> 271
- **Evals:** norm_word_ce = 2.6924 @250; genghisbench = w-250 19/20, median 11 (v5); kl = 0.07509 @250
- **Checkpoints / exports:** w-271
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Citations:**
  - `/data/home/amartyavarma/mistral-distill-bf16/stage2_1ch/logs/s2-1ch-7454.log:318 -> EVAL step 250 KL 0.07509 nword 2.6924`
  - `/data/home/amartyavarma/mistral-distill-bf16/stage2_1ch/GENGHIS_LOG.txt:2 -> w-250 19/20 (j7454)`

### mistral-s2-baseline

**Mistral stage-2 agreed baseline (j7466 kl40 lr5e-6 -> j7489 kl100 lr2.5e-6)**  
group `base-stage` · status **stopped-with-save** · dir `/data/home/amartyavarma/mistral-distill-bf16/stage2_1ch`

- **Jobs:** 7466, 7489
- **Dates:** 2026-10-03 08:31 .. 13:26 PT
- **Base model:** unknown
- **Init:** j709 w-531 (fresh) + Ministral-3-8B voice tower
- **Parents (resolved):** mistral-s1-1ch@531, base:ministral-3-8b-base-2512
- **Produces:** mistral-s2-baseline@250, mistral-s2-baseline@427
- **Data:** tts-pretrain-128m speech + klpre-split1 text KL (152 eval rows)
- **Recipe:** j7466: W_KL 40, trunk LR 5e-6, K=8 light sub-steps, light_frac 0.5; j7489: resume dcp-250 with W_KL 100, trunk LR 2.5e-6
- **Steps:** 0 -> 427
- **Evals:** norm_word_ce = 2.6731 @400; 2.6976 @250; kl = 0.06941 @400; other = cb0 2.5033, light 5.9694 @400
- **Checkpoints / exports:** w-250; dcp-250; w-427; dcp-427
- **Local paths:** /data/home/amartyavarma/mistral-distill-bf16/stage2_1ch/ckpt-weka/m35-s2-1ch-k8lf05-fce-ln2e-5-kl100-tim1-lr2.5e-6-w152-j7489/w-427
- **HF repos:** [InternalCan/mistral-1ch-s2-ckpts](https://huggingface.co/InternalCan/mistral-1ch-s2-ckpts)
- **Notes:** No later Mistral training found; the Mistral line stopped here (10-03). Ministral base id spelling is a guess.
- **Citations:**
  - `/data/home/amartyavarma/mistral-distill-bf16/stage2_1ch/logs/s2-1ch-7466.log:318 -> EVAL step 250 KL 0.07296 nword 2.6976`
  - `/data/home/amartyavarma/mistral-distill-bf16/stage2_1ch/logs/s2-1ch-7489.log:219 -> EVAL step 400 KL 0.06941 nword 2.6731 cb0 2.5033 light 5.9694`
  - `/data/home/amartyavarma/hf-backup-staging/readme/mistral-1ch-s2-ckpts/README.md:11 -> j7489-dcp-427 = agreed Mistral stage-2 baseline`
  - `/data/home/amartyavarma/hf-backup-staging/readme/mistral-1ch-s2-ckpts/README.md:12 -> chain j7466 -> j7489 (stopped at 427)`
  - `/data/home/amartyavarma/hf-backup-staging/readme/mistral-1ch-s2-ckpts/README.md:9 -> Ministral-3-8B voice tower + light decoder`

### samft-mistral-lane

**samft-mistral-lane-1k (SAM tower on Mistral lane vocab)**  
group `base-stage` · status **finished** · dir `/data/home/amartyavarma/ref/sam-graft`

- **Jobs:** 7603
- **Dates:** ~2026-10-03 (unknown)
- **Base model:** unknown
- **Init:** SAM 8B sam-51960
- **Parents (resolved):** base:sam-51960
- **Produces:** samft-mistral-lane@1000
- **Data:** Mistral stage-2 speech rows (152 eval rows)
- **Recipe:** ft_tower.py
- **Steps:** 0 -> 1000
- **Evals:** other = unknown (see README table)
- **Checkpoints / exports:** step-1000
- **Local paths:** unknown
- **HF repos:** [InternalCan/sam-graft-ckpts](https://huggingface.co/InternalCan/sam-graft-ckpts)
- **Notes:** Whether any Mistral run consumed it is unknown (Mistral stage 2 stopped 10-03).
- **Citations:**
  - `/data/home/amartyavarma/hf-backup-staging/readme/sam-graft-ckpts/README.md:12 -> samft-mistral-lane-1k job 7603, Mistral-lane equivalent`

### prometheus

**Prometheus (external Qwen3-32B stage-2-style speech+text run; reference only)**  
group `other` · status **finished** · dir `/data/home/amartyavarma/prom_ckpt`

- **Jobs:** unknown
- **Dates:** checkpoints 2026-08-05 (step 2001) and 2026-08-09 (step 26013); downloaded 2026-10-03
- **Base model:** Qwen/Qwen3-32B
- **Init:** base Qwen3-32B with stock RoPE theta 1e6
- **Parents (resolved):** base:qwen3-32b
- **Produces:** prometheus@2001, prometheus@26013
- **Data:** schedule speech, speech, text (data_state.json)
- **Recipe:** external (not trained by us); world size 72; trunk peak LR 5e-5, warmup 500, cosine -> 0.1x, clip 0.5 (as replicated by promA)
- **Steps:** -> 26013
- **Evals:** other = listening m-a +0.16 at 2k steps -> +0.56 at 26k
- **Checkpoints / exports:** checkpoint-2001-2026-08-05; checkpoint-26013-2026-08-09
- **Local paths:** /data/home/amartyavarma/prom_ckpt/checkpoint-26013-2026-08-09; /data/home/amartyavarma/prom_ckpt/checkpoint-2001-2026-08-05
- **HF repos:** none named in run files
- **Notes:** Reference run, not ours; used for comparisons (qwen-1ch/prom_compare). Demeter/Hestia were initialised from it. PPM does NOT start from Prometheus.
- **Citations:**
  - `/data/home/amartyavarma/prom_ckpt/checkpoint-26013-2026-08-09/data_state.json:2 -> global_step 26013`
  - `/data/home/amartyavarma/prom_ckpt/checkpoint-26013-2026-08-09/data_state.json:3 -> world_size 72`
  - `/data/home/amartyavarma/promck-dl/README_dedup.txt:1 -> duplicate Prometheus ckpts symlinked to prom_ckpt`
  - `memory:qwen-mixrope-mistral-stock.md:16 -> Prometheus used base Qwen3-32B with stock RoPE theta 1e6`
  - `/data/home/amartyavarma/qwen-1ch/stage2/prod_promA.env:3 -> Prometheus trunk budget`
  - `memory:mistral-s2-decision-criteria.md:41 -> Prometheus m-a +0.16 at 2k -> +0.56 at 26k`

### demeter-hestia

**Demeter / Hestia (wandb canopy-labs/demeter run demeter-32b-ga3; stage-3 reference)**  
group `other` · status **finished** · dir `unknown (external; wandb meta in /data/home/amartyavarma/demeter-causalsam/meta)`

- **Jobs:** unknown
- **Dates:** unknown
- **Base model:** unknown
- **Init:** Prometheus (stage-2-style init comparable to ours)
- **Parents (resolved):** prometheus@unknown
- **Produces:** demeter-hestia@19500
- **Data:** duplex speech, 192 speech rows/step, epoch 4.82M rows = 25,117 steps; Gemma text-duplex conversations (gemma-synth-*)
- **Recipe:** external; 4 resumed pieces stitched
- **Steps:** -> 20505
- **Evals:** other = stage-3 judging bar (normalised word CE at equal epoch fraction)
- **Checkpoints / exports:** hestia-ctx-19500
- **Local paths:** unknown
- **HF repos:** [InternalCan/demeter-32b-ckpts](https://huggingface.co/InternalCan/demeter-32b-ckpts)
- **Notes:** Reference only. The Prometheus step Demeter started from is unknown.
- **Citations:**
  - `memory:qwen-s3-success-criteria.md:9 -> Hestia = demeter-32b-ga3, 20,505 steps, 192 rows/step, epoch 25,117 steps; hestia-ctx-19500`
  - `memory:demeter-init-comparable.md:11 -> Demeter initialised from Prometheus`
  - `/data/home/amartyavarma/demeter-causalsam/initcmp.py:20 -> D='InternalCan/demeter-32b-ckpts'`
  - `/data/home/amartyavarma/qwen-1ch/stage3/data/txtdup_build.py:1 -> Demeter's Gemma text-duplex conversations`

### demeter-causalsam

**demeter-32b-causalsam checkpoint-3960 (external, downloaded for comparison)**  
group `other` · status **unknown** · dir `/data/home/amartyavarma/demeter-causalsam`

- **Jobs:** unknown
- **Dates:** downloaded 2026-10-13 11:25 PT
- **Base model:** unknown
- **Init:** unknown (Demeter lineage)
- **Parents (resolved):** demeter-hestia@unknown
- **Produces:** none recorded
- **Data:** unknown
- **Recipe:** external
- **Steps:** -> 3960
- **Evals:** none recorded
- **Checkpoints / exports:** checkpoint-3960
- **Local paths:** /scratch/amartyavarma/demeter-causalsam/fp32 (node-local)
- **HF repos:** [InternalCan/demeter-32b-causalsam](https://huggingface.co/InternalCan/demeter-32b-causalsam)
- **Notes:** Parent guessed; meta has wandb_demeter_afa85vf2.json / wandb_demeter-ab-aloe_p0qyep13.json (not parsed).
- **Citations:**
  - `/data/home/amartyavarma/demeter-causalsam/dl.sh:7 -> hf download InternalCan/demeter-32b-causalsam checkpoint-3960`

### gemma-not-trained

**Gemma: text-data source only (no Gemma model was trained)**  
group `other` · status **unknown** · dir `/data/home/amartyavarma/ppm`

- **Jobs:** unknown
- **Dates:** 2026-10-11 (txtdup_*.sbatch written 15:34-16:12 PT)
- **Base model:** unknown
- **Init:** n/a
- **Parents (resolved):** unknown
- **Produces:** none recorded
- **Data:** Gemma-synthesised two-speaker WLT conversations: InternalCan/gemma-synth-2m-wlts-high-overlap (1,316,588) + InternalCan/gemma-synth-r2-wlts (1,967,466) = 3.28M; built into text-duplex CE stores (gemma-txtdup-v1, ppm txtdup-full)
- **Recipe:** ppm/txtdup_*.sbatch = CPU data-build jobs running qwen-1ch/stage3/data/txtdup_build.py (CAP 7500, 36 procs); no training
- **Steps:** n/a
- **Evals:** none recorded
- **Checkpoints / exports:** unknown
- **Local paths:** /scratch/amartyavarma/ppm/txtdup-full (node-local); /data/home/amartyavarma/data/gemma-synth-2m-wlts-high-overlap; /data/home/amartyavarma/data/gemma-synth-r2-wlts
- **HF repos:** [InternalCan/gemma-synth-2m-wlts-high-overlap](https://huggingface.co/InternalCan/gemma-synth-2m-wlts-high-overlap) *(not in model listing: dataset/bucket or deleted)*, [InternalCan/gemma-synth-r2-wlts](https://huggingface.co/InternalCan/gemma-synth-r2-wlts) *(not in model listing: dataset/bucket or deleted)*
- **Notes:** 'Gemma txtdup' = PPM/stage-3 text-duplex CE rows made from Gemma-generated conversations; Gemma is never a trained model or init in any lineage. Not a training run (row kept to answer the question).
- **Citations:**
  - `/data/home/amartyavarma/ppm/txtdup_full.sbatch:7 -> 'T4: two-track text-duplex CE store over ALL Gemma WLT conversations (3.28M)'`
  - `/data/home/amartyavarma/ppm/txtdup_full.sbatch:11 -> runs txtdup_build.py (data build)`
  - `/data/home/amartyavarma/qwen-1ch/stage3/data/txtdup_build.py:4 -> gemma-synth-2m-wlts-high-overlap 1,316,588 conversations`
  - `/data/home/amartyavarma/qwen-1ch/stage3/data/txtdup_build.py:5 -> gemma-synth-r2-wlts 1,967,466 conversations`
  - `/data/home/amartyavarma/qwen-1ch/stage3/README_S3.md:179 -> Demeter's Gemma text-duplex conversations (HF InternalCan/...)`
  - `memory:role-flip-never-chat.md:26 -> Gemma txtdup = symmetric two-party data that may be flipped`

## PPM 32B pretraining, 8,1,1 continuation, pose5, other pretraining

| run | status | jobs | dates | init (parent node@step) | steps | headline evals | outputs |
|---|---|---|---|---|---|---|---|
| [`ppm32b`](#ppm32b) | stopped-with-save | 18821, 18870, 18872, 18873, 18906 (+21) | 2026-10-11 17:41 PT (first stepping job 18873; 18821/18870/… | base:qwen3-32b | 0 -> 16861 (target 17366 = one epoch, never reached; continued as the… | word CE(norm) last EVAL @16825: speech (podcast held-out) 1.9530, gemma txtdup 0.7685, chat 0.5911, pose rows 1.8690, speech_s3 (Demeter rows) 2.0251. Step 0:… | dcp-7932 (HF ppm-v1); w-8869 (Weka export, later deleted); w-10351 (Weka export, later deleted) / HF: ppm-v1, ppm-v1-b, ppm-main-w12230, pp… |
| [`ppm-linkexp-a`](#ppm-linkexp-a) | finished | 26303 | 2026-10-20 21:10 PT .. 2026-10-20 22:09 PT | ppm32b@16762 | 16762 -> 17062 (300 experiment steps) | word CE(norm) pose rows 1.8842 @17062 (1.8706 @16762); pose cb0 1.6841 @17062 (1.6846 @16762); pose light 3.7950 @17062 (4.7587 @16762 right after the link was… | dcp-17062 (STOP_AFTER_SAVE) |
| [`ppm811-epoch2`](#ppm811-epoch2) | abandoned | 26391, 27030 | 2026-10-20 23:13 PT .. 2026-10-21 02:09 PT (26391); 2026-10… | ppm32b@16861 | 16861 -> 17012 (26391) -> 17120 (27030) | word CE(norm) @16996 (26391): pose rows 1.8725, speech 1.9609, gemma 0.7680, chat 0.5906; @17119 (27030): pose rows 1.8810, speech 1.9629, gemma 0.7709, chat 0… | dcp-16914, dcp-16967, dcp-17012 (26391; HF ppm-v1-b); dcp-17120 (27030; HF ppm-v1-b) / HF: ppm-v1-b |
| [`ppm811-falsestarts`](#ppm811-falsestarts) | abandoned | 27100, 27105, 27110 | 2026-10-21 23:26 PT .. 2026-10-22 01:46 PT | ppm32b@16861 | 16861 -> 16967 (27100); 16861 -> 16865 (27105); 16861 -> 16862 (27110) | word CE(norm) @16967 (27100): pose rows 1.8810, speech 1.9631, gemma 0.7696, chat 0.7514 (vs 0.5909 at 16861); pose cb0 1.6696 @16967; pose light 3.8217 @16967 | dcp-16865 (HF ppm-v1-b; the record names job 27100); dcp-16967 (27100) / HF: ppm-v1-b |
| [`ppm811-v4`](#ppm811-v4) | finished | 27114 | 2026-10-22 01:47 PT .. 2026-10-22 11:23 PT | ppm32b@16861 | 16861 -> 17352 (target 17352; ended when the fresh pose ran out) | word CE(norm) @17352: pose rows 1.8928, speech 1.9649, gemma 0.7677, chat 0.5900, speech_s3 2.0371; pose cb0 1.6531 @17352; pose light 3.7329 @17352 (start eva… | dcp-17352 (final save; HF ppm-v1-c, 320 files, PASS); intermediate dcp-16914-j27114 .. dcp-17333 (HF ppm-v1-b/-c) / HF: ppm-v1-c, ppm-v1-b |
| [`ppm811-append2`](#ppm811-append2) | finished | 27204 | 2026-10-22 11:37 PT .. 2026-10-22 17:20 PT | ppm811-v4@17352 | 17352 -> 17641 (target 17641; pose exhausted) | word CE(norm) @17641: pose rows 1.8987, speech 1.9667, gemma 0.7689, chat 0.5897; pose cb0 1.6499 @17641; pose light 3.7302 @17641 | dcp-17641 (final save; HF ppm-v1-c PASS) / HF: ppm-v1-c |
| [`ppm811-append3`](#ppm811-append3) | stopped-with-save | 27653 | 2026-10-22 18:22 PT .. 2026-10-22 22:29 PT | ppm811-append2@17641 | 17641 -> 17850 (target 18022, never reached on this lineage) | word CE(norm) last EVAL @17828: pose rows 1.9020, speech 1.9656, gemma 0.7672, chat 0.5900. The model at 17850 (resume eval in 31101): speech 1.9658, gemma 0.7… | dcp-17850 (pause save; HF bundle ppm-v1-d, shards ppm-v1-c, 320 files 737.4 GB PASS) / HF: ppm-v1-c, ppm-v1-d |
| [`ppm811-decay`](#ppm811-decay) | stopped-with-save | 27810 | 2026-10-22 23:36 PT .. 2026-10-23 01:11 PT | ppm811-append3@17850 | 17850 -> 17928 (target 18022; LR multiplier 0.6337 at the last eval) | word CE(norm) @17914: pose rows 1.8999, speech 1.9614, gemma 0.7613, chat 0.5864 (no constant-LR eval at matched steps; main's last was @17828: 1.9020 / 1.9656… | dcp-17928 (stop save) -> HF InternalCan/ppm-decay-17928 (17/17 PASS, 390 files, 737.4 GB) / HF: ppm-decay-17928 |
| [`pose5`](#pose5) | paused | 31101 | 2026-10-27 11:36 PT .. 2026-10-27 16:27 PT (paused) | ppm811-append3@17850 | 17850 -> 18098 (target 19382) | word CE(norm) @18092: pose rows 1.9069, speech 1.9675, gemma 0.7669, chat 0.5899 (@17850: 1.9017 / 1.9658 / 0.7678 / 0.5896); pose cb0 1.6431 @18092 (1.6463 @1… | dcp-17903, dcp-17956, dcp-18009, dcp-18062 (rotated hourly saves; HF ppm-v1-d); dcp-18098 (pause save; COMPLETE 16/16, HF ppm-v1-d PASS) /… |
| [`ppm14b`](#ppm14b) | stopped-with-save | 26513, 26559, 26560, 26561, 26562 (+2) | 2026-10-21 05:06 PT .. 2026-10-21 20:27 PT (stopped at 4304… | base:qwen3-14b | 0 -> 4304 (26571, intended stop) -> 4502 (28012, mistaken resume); ta… | word CE(norm) @4460: speech 2.0859, gemma 0.8580, chat 0.6416, speech_s3 2.1599 (@4283: 2.0868 / 0.8594 / 0.6426) | w-2208, w-4091 (Weka exports; HF InternalCan/ppm14b, w-4091 PASS); dcp-4304; dcp-4502 (node-local) / HF: ppm14b |
| [`ppm-m128r`](#ppm-m128r) | abandoned | 26678, 26764, 26776 | 2026-10-21 14:53 PT .. 2026-10-21 20:18 PT | base:mistral-medium-3.5-128b | 0 -> 210 (26678) -> 213 (26764, diverged, cancelled without a save) -… | word CE(norm) @210: pose rows 3.7691, speech 3.0669, gemma 1.6581, chat 1.1269; pose cb0 7.6264 @210 (did not learn); pose light 7.5547 @210 | dcp-210 (all-frozen); dcp-216 / HF: ppm-m128r-v1 |
| [`ppm-misc-notes`](#ppm-misc-notes) | unknown (note) | unknown | unknown | unknown | n/a | unknown | unknown |


### ppm32b

**PPM 32B (pure PPM; run_base ppm-v1; prod_ppm.env)**  
group `ppm32b` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/code`

- **Jobs:** 18821, 18870, 18872, 18873, 18906, 18910, 18926, 18975, 18983, 19013, 19082, 19127, 20363, 20364, 20492, 20904, 20942, 21765, 22224, 22362, 22613, 25610, 25614, 25627, 26271, 26325
- **Dates:** 2026-10-11 17:41 PT (first stepping job 18873; 18821/18870/18872 died at start 16:48-17:40) .. 2026-10-20 23:08 PT (STOP_AFTER_SAVE at 16861, job 26325); paused 2026-10-17 02:23 PT (dcp-12829) .. 2026-10-19 08:25 PT
- **Base model:** Qwen/Qwen3-32B (local copy /data/home/amartyavarma/models/Qwen3-32B)
- **Init:** fresh from stock Qwen3-32B: 708 backbone tensors loaded (embedding copied to channels A and B), 1440 random tensors (towers, decoders, audio/pose tables), seed 20261011. NOT Prometheus / not a stage-2 Qwen ckpt.
- **Parents (resolved):** base:qwen3-32b
- **Produces:** ppm32b@7932, ppm32b@8869, ppm32b@10351, ppm32b@11488, ppm32b@12529, ppm32b@12829, ppm32b@13533, ppm32b@14503, ppm32b@15973, ppm32b@16610, ppm32b@16762, ppm32b@16861
- **Data:** GA per step = pose, speech, speech, text. Pose: packed TikTok/SAM pose rows /scratch/amartyavarma/pose-s3-samdata/rows (2.21M train, channel B). Speech: the 4.45M full 600 s podcast rows of the stage-3 store s3-rows, plus D2 extra slices (configs/extra_plan_d2.json) from step 170; eval salt s3eval-v1 held out. Text: Gemma WLT text-duplex store txtdup-full (CAP 7500) + corpus15m chat splits 2,3.
- **Recipe:** code /data/home/amartyavarma/ppm/code (train_ppm.py, ppm.sbatch), env prod_ppm.env; 16 nodes, world 128. Trunk: 64-layer Qwen3-32B, full FT, plain RoPE, CD 2. Speech and pose pairs: 36-layer tower + 28-layer 0.6B decoder each. LRs: trunk 5e-5, tower 3e-4, audio 2e-4, decoder 4e-4; warmup 100, then CONSTANT (PPM_SCHED and PPM_DEC_SCHED constant, user: no decay). Decoder K=8 nanosteps on 100% of frames (D31, from step 399). D32 pose cb0 head added at 1083, then removed (D36/D37). D33 backbone-to-pose-decoder link decp_bb on from 1470. PPM_MAX_STEPS 17400 (resolved 17366). ~28.5 s/step.
- **Steps:** 0 -> 16861 (target 17366 = one epoch, never reached; continued as the 8,1,1 lineage)
- **Evals:** norm_word_ce = last EVAL @16825: speech (podcast held-out) 1.9530, gemma txtdup 0.7685, chat 0.5911, pose rows 1.8690, speech_s3 (Demeter rows) 2.0251. Step 0: speech 3.3000, gemma 2.9164, chat 1.0677.; pose_cb0_ce = 1.6829 @16825 (7.7919 @0); pose_light_ce = 3.7889 @16825, with the D33 link on; genghisbench = w-8869: 10/20 successes (median 13.5 questions); w-4420: 8/20; reference base Qwen3-32B: 10/20 (median 16.5); other = speech cb0 0.7146, speech light 2.2137 @16825. Trajectory (pose-row word / speech word / gemma word / pose cb0 / pose light): @7909 1.9518/2.0071/0.8073/1.7818/4.0310; @11453 1.9077/1.9799/0.7852/1.7269/3.8859; @12783 1.8963/1.9728/0.7808/1.7137/3.8710; @13538 1.8904/1.9696/0.7798/1.7077/3.8449; @14476 1.8845/1.9620/0.7752/1.6997/3.8330; @15983 1.8736/1.9575/0.7713/1.6893/3.8118; @16610 1.8716/1.9560/0.7701/1.6845/3.7992
- **Checkpoints / exports:** dcp-7932 (HF ppm-v1); w-8869 (Weka export, later deleted); w-10351 (Weka export, later deleted); dcp-11488 (NOT on HF: rotated out before hold; bf16 inference conversion local-11488 exists); dcp-12529 (inference conversion local-12529); dcp-12829 (HF ppm-v1); dcp-13533 (HF ppm-v1); w-14503 (HF ppm-v1-b; Weka copy deleted at the w-15973 export); w-15973 (Weka, still on disk, + HF ppm-v1-b); dcp-16610 (HF ppm-v1-b); dcp-16762 (HF ppm-v1-b); dcp-16861 (HF ppm-v1-b; 320 files, 737.4 GB, PASS); other Weka exports w-2208/2948/3683/4420/9610/12230/13771/15237 (rotated)
- **Local paths:** /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-<N> (node-local on the 16 hosts; which are still present is unknown); /data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-15973; /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488; /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-12529; /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-12829
- **HF repos:** [InternalCan/ppm-v1](https://huggingface.co/InternalCan/ppm-v1), [InternalCan/ppm-v1-b](https://huggingface.co/InternalCan/ppm-v1-b), [InternalCan/ppm-main-w12230](https://huggingface.co/InternalCan/ppm-main-w12230), [InternalCan/ppm-main-w12465](https://huggingface.co/InternalCan/ppm-main-w12465), [InternalCan/ppm-main-dcp-12465](https://huggingface.co/InternalCan/ppm-main-dcp-12465)
- **Notes:** Memory (ppm-vs-811-naming.md:8) says this lineage shares HF repo ppm-v1-c, but the records put dcp-16861 and every PPM save in ppm-v1 / ppm-v1-b; ppm-v1-c only starts at 8,1,1 dcp-17125. dcp-16762 also seeds the link-exp branch (ppm-linkexp-a). w-7932 was never a Weka export: the w-7558 export was REFUSED (20492.log:1931) and the next one was w-8869, so a 'w-7932' must be a later conversion of the HF dcp-7932 (which file is unknown). dcp-11488 has no HF copy, only the inference conversion local-11488. Which node-local dcps still exist today is unknown.
- **Citations:**
  - `/data/home/amartyavarma/ppm/DESIGN.md:6 -> backbone = base Qwen3-32B, full FT, no LoRA`
  - `/data/home/amartyavarma/ppm/DESIGN.md:19-24 -> pose 2.21M rows, 4.45M podcast rows, Gemma 3.28M text-duplex`
  - `/data/home/amartyavarma/ppm/logs/ppm-18873.log:1 -> job 18873 2026-10-11 17:41:21 PT, env prod_ppm.env`
  - `/data/home/amartyavarma/ppm/logs/ppm-18873.log:4 -> [init] backbone <- /data/home/amartyavarma/models/Qwen3-32B, 708 tensors, 1440 random`
  - `/data/home/amartyavarma/ppm/logs/ppm-18873.log:7 -> resume auto: none (fresh start)`
  - `/data/home/amartyavarma/ppm/logs/ppm-18873.log:10 -> EVAL step 0 (speech 3.3000, gemma 2.9164, chat 1.0677, pose cb0 7.7919)`
  - `/data/home/amartyavarma/ppm/logs/ppm-18873.log:563 -> STOPPED at step 163`
  - `/data/home/amartyavarma/ppm/logs/ppm-18821.log (tail), ppm-18870.log:7, ppm-18872.log (tail) -> early starts died (exit 143)`
  - `/data/home/amartyavarma/ppm/logs/ppm-20363.log:1 -> ABORT: PPM_ENV not set (no steps)`
  - `/data/home/amartyavarma/ppm/logs/ppm-18906.log:8 -> resume dcp-163; chain continues through each job's [resume] line 9-11`
  - `/data/home/amartyavarma/ppm/logs/ppm-20492.log:2321 -> job 20492 last step 7932`
  - `/data/home/amartyavarma/ppm/logs/ppm-20492.log:2294 -> EVAL step 7909`
  - `/data/home/amartyavarma/ppm/logs/ppm-21765.log:770 -> last step 11488; :731 EVAL 11453`
  - `/data/home/amartyavarma/ppm/logs/ppm-22613.log:715 -> STOPPED at step 12829 (10-17 02:23 PT); :665 EVAL 12783`
  - `/data/home/amartyavarma/ppm/logs/ppm-25610.log:10 -> 10-19 08:26 PT resume from dcp-12829`
  - `/data/home/amartyavarma/ppm/logs/ppm-25627.log:4227 -> STOPPED at step 16610; :1050 EVAL 13538; :2018 EVAL 14476; :3576 EVAL 15983; :4223 EVAL 16610`
  - `/data/home/amartyavarma/ppm/logs/ppm-25627.log:2046 -> export w-14503; :3565 export w-15973 (deleted w-14503)`
  - `/data/home/amartyavarma/ppm/logs/ppm-20942.log:1164 -> export w-8869; :2696 -> export w-10351`
  - `/data/home/amartyavarma/ppm/logs/ppm-26271.log:691 -> STOPPED at step 16762`
  - `/data/home/amartyavarma/ppm/logs/ppm-26325.log:138 -> resume dcp-16762`
  - `/data/home/amartyavarma/ppm/logs/ppm-26325.log:595 -> last EVAL step 16825 (speech 1.9530, gemma 0.7685, chat 0.5911, pose 1.8690, pose cb0 1.6829, pose light 3.7889)`
  - `/data/home/amartyavarma/ppm/logs/ppm-26325.log:634 -> STOP_AFTER_SAVE at step 16861, 10-20 23:08:07 PT`
  - `/data/home/amartyavarma/ppm/logs/ppm-26325.log:636 -> done 17366 steps (resolved target)`
  - `/data/home/amartyavarma/ppm/code/prod_ppm.env:2 -> PPM_RUN=ppm-v1`
  - `/data/home/amartyavarma/ppm/code/prod_ppm.env:25-35 -> LRs and warmup 100`
  - `/data/home/amartyavarma/ppm/code/prod_ppm.env:38-40 -> horizon 17400, PPM_SCHED/DEC_SCHED constant`
  - `/data/home/amartyavarma/ppm/code/prod_ppm.env:46 -> PPM_DEC_K=8`
  - `/data/home/amartyavarma/ppm/code/prod_ppm.env:59-61 -> pose / speech / text roots`
  - `/data/home/amartyavarma/ppm/code/prod_ppm.env:78 -> PPM_CHAT_SPLIT=2,3`
  - `/data/home/amartyavarma/ppm/code/prod_ppm.env:88 -> PPM_MAX_STEPS=17400`
  - `/data/home/amartyavarma/ppm/code/prod_ppm.env:102 -> PPM_SAVE_ROOT=/scratch/amartyavarma/ppm/ckpt`
  - `/data/home/amartyavarma/ppm/code/prod_ppm.env:123-126 -> D2 extra plan from step 170, holdout salt s3eval-v1`
  - `/data/home/amartyavarma/ppm/code/prod_ppm.env:167 -> PPM_DECP_BB=1 (D33 link)`
  - `/data/home/amartyavarma/ppm/RESUME_PPM_12829.txt:1-8 -> dcp-12829 resume; HF chain InternalCan/ppm-v1 then ppm-v1-b from dcp-14278`
  - `/data/home/amartyavarma/setup-logs/hfb/records.jsonl:22 -> dcp-7932 job 20904 repo InternalCan/ppm-v1`
  - `/data/home/amartyavarma/setup-logs/hfb/records.jsonl:125 -> dcp-11488 dcp-skipped (rotated out before hold)`
  - `/data/home/amartyavarma/setup-logs/hfb/records.jsonl:147 -> dcp-12829 InternalCan/ppm-v1`
  - `/data/home/amartyavarma/setup-logs/hfb/records.jsonl:156 -> dcp-13533 InternalCan/ppm-v1`
  - `/data/home/amartyavarma/setup-logs/hfb/records.jsonl:166 -> w-14503 InternalCan/ppm-v1-b`
  - `/data/home/amartyavarma/setup-logs/hfb/records.jsonl:178 -> w-15973 InternalCan/ppm-v1-b`
  - `/data/home/amartyavarma/setup-logs/hfb/records.jsonl:185 -> dcp-16610 InternalCan/ppm-v1-b`
  - `/data/home/amartyavarma/setup-logs/hfb/records.jsonl:187 -> dcp-16762 job 26271 InternalCan/ppm-v1-b`
  - `/data/home/amartyavarma/setup-logs/hfb/records.jsonl:188 -> dcp-16861 job 26325 InternalCan/ppm-v1-b, 320 files, PASS, 2026-10-21 07:00 UTC`
  - `/data/home/amartyavarma/setup-logs/hfb/state/hf_repos.json:5-27 -> repo chain ppm-v1 -> ppm-v1-b (from dcp-14278) -> ppm-v1-c (from dcp-17125) -> ppm-v1-d (from dcp-17850)`
  - `/data/home/amartyavarma/ppm/gb/ppm-v1-w-8869/gb_s100-119.json:4-6 -> GenghisBench 10/20, median 13.5`
  - `/data/home/amartyavarma/ppm/gb/ppm-v1-w-4420/gb_s100-119.json:4-5 -> 8/20`
  - `/data/home/amartyavarma/ppm/gb/qwen3-32b-base/gb_s100-119.json:4-6 -> base 10/20, median 16.5`
  - `ls /data/home/amartyavarma/ppm/ckpt-weka/ppm-v1 -> only w-15973 present; ls /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1 -> local-11488, local-12529, local-12829, local-17641, local-17850, local-18098`
  - `memory:ppm-vs-811-naming.md:7-9 -> pure PPM = prod_ppm.env, last job 26325, final dcp-16861; Aloe ref FTs used w-15973`
  - `memory:ppm-pretrain-no-decay.md:11-13 -> constant LR, no decay`

### ppm-linkexp-a

**link-exp arm A (PPM pose-only, D33 link removed, 300 steps)**  
group `ppm32b` · status **finished** · dir `/data/home/amartyavarma/ppm/link-exp`

- **Jobs:** 26303
- **Dates:** 2026-10-20 21:10 PT .. 2026-10-20 22:09 PT
- **Base model:** unknown
- **Init:** ppm32b dcp-16762 (/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-16762, PPM job 26271's stop save); decp_bb AdamW state set aside
- **Parents (resolved):** ppm32b@16762
- **Produces:** ppm-linkexp-a@17062
- **Data:** PPM's data, pose micros only (ga_seq ['pose'])
- **Recipe:** link-exp/code/train_ppm.py + linkA.env; resolved-cfg diff vs 26271: ga_seq pose only, decp_bb_off 1, exp_steps 300, periodic saves/evals off, final_save 0; 16 nodes
- **Steps:** 16762 -> 17062 (300 experiment steps)
- **Evals:** norm_word_ce = pose rows 1.8842 @17062 (1.8706 @16762); pose_cb0_ce = 1.6841 @17062 (1.6846 @16762); pose_light_ce = 3.7950 @17062 (4.7587 @16762 right after the link was removed)
- **Checkpoints / exports:** dcp-17062 (STOP_AFTER_SAVE)
- **Local paths:** /scratch/amartyavarma/ppm/ckpt/ppm-linkexp-a/dcp-17062 (current presence unknown)
- **HF repos:** none named in run files
- **Notes:** A short experiment that the 8,1,1 'link off' decision was ported from. No HF record of dcp-17062 found. Nothing downstream is known to use it.
- **Citations:**
  - `/data/home/amartyavarma/ppm/link-exp/logs/linkA-26303.log:1 -> job 26303, 10-20 21:10:35 PT, env link-exp/code/linkA.env`
  - `/data/home/amartyavarma/ppm/link-exp/logs/linkA-26303.log:266 -> resume dcp-16762, decp_bb AdamW set aside`
  - `/data/home/amartyavarma/ppm/link-exp/logs/linkA-26303.log:277 -> EVAL step 16762`
  - `/data/home/amartyavarma/ppm/link-exp/logs/linkA-26303.log:979 -> EVAL step 17062`
  - `/data/home/amartyavarma/ppm/link-exp/logs/linkA-26303.log:982 -> STOPPED at step 17062, 22:09 PT`
  - `/data/home/amartyavarma/ppm/link-exp/CONFIG_DIFF_vs_26271.txt:3-20 -> config diff (pose-only GA, decp_bb_off, exp_steps 300)`
  - `/data/home/amartyavarma/ppm/code-811/prod_811_fresh16861_v4.env:180 -> 8,1,1 LINK OFF ported from ppm/link-exp = job 26303`

### ppm811-epoch2

**8,1,1 first legs: PPM-data epoch 2 (26391) then first fresh-data leg from 17012 (27030); abandoned branch**  
group `ppm811` · status **abandoned** · dir `/data/home/amartyavarma/ppm/code-811`

- **Jobs:** 26391, 27030
- **Dates:** 2026-10-20 23:13 PT .. 2026-10-21 02:09 PT (26391); 2026-10-21 21:12 PT .. 2026-10-21 23:21 PT (27030)
- **Base model:** unknown
- **Init:** ppm32b dcp-16861 (8,1,1 switch; weights + optimizers restored, decp_bb AdamW set aside because the link is off)
- **Parents (resolved):** ppm32b@16861
- **Produces:** ppm811-epoch2@17012, ppm811-epoch2@17120
- **Data:** 26391 (prod_811.env): rest of PPM's pose epoch 1, then epoch 2 as fresh node permutations (seed 20261021), 8 pose rows per step; speech continues the D2 stream, then epoch 2; text continues. 27030 (prod_811_fresh.env): fresh pose (new-pose-llm-data plan 1) from 17012, speech flip on repeats, text flip (Gemma + chat).
- **Recipe:** code-811 (PPM_S811=1): GA pose*8,speech,text; BB_MODE full,frozen*7,full,full; decp_bb link OFF; LRs constant as PPM; DEC_K 8 (64 pose-light updates per step); max_steps 19063 (26391)
- **Steps:** 16861 -> 17012 (26391) -> 17120 (27030)
- **Evals:** norm_word_ce = @16996 (26391): pose rows 1.8725, speech 1.9609, gemma 0.7680, chat 0.5906; @17119 (27030): pose rows 1.8810, speech 1.9629, gemma 0.7709, chat 0.7949 (rise from the chat role flip); pose_cb0_ce = 1.6608 @16996; 1.6567 @17119; pose_light_ce = 3.7492 @16996; 3.7415 @17119
- **Checkpoints / exports:** dcp-16914, dcp-16967, dcp-17012 (26391; HF ppm-v1-b); dcp-17120 (27030; HF ppm-v1-b)
- **Local paths:** /scratch/amartyavarma/ppm/ckpt-old811f/ppm-v1 (dcp-17120 moved here, not exported to Weka)
- **HF repos:** [InternalCan/ppm-v1-b](https://huggingface.co/InternalCan/ppm-v1-b)
- **Notes:** Abandoned branch. The user ordered a restart from the end of PPM (dcp-16861) on fresh data, which is ppm811-v4. HF step names dcp-16914/16967/17012 belong to this branch; v4's own 16914 is stored as dcp-16914-j27114. Nothing downstream is known to use it.
- **Citations:**
  - `/data/home/amartyavarma/ppm/code-811/README_811.md:3-5 -> 8,1,1 launched 10-20 23:13 PT as job 26391 from PPM dcp-16861`
  - `/data/home/amartyavarma/ppm/code-811/README_811.md:21-31 -> 8,1,1 schedule`
  - `/data/home/amartyavarma/ppm/code-811/README_811.md:104-111 -> data continuation (epoch-2 permutations)`
  - `/data/home/amartyavarma/ppm/code-811/prod_811.env:13 -> GA pose*8,speech,text; :94 -> PPM_MAX_STEPS=19063; :183 -> PPM_DECP_BB_OFF=1`
  - `/data/home/amartyavarma/ppm/logs/ppm-26391.log:1 -> env code-811/prod_811.env`
  - `/data/home/amartyavarma/ppm/logs/ppm-26391.log:140 -> resume dcp-16861; decp_bb AdamW set aside`
  - `/data/home/amartyavarma/ppm/logs/ppm-26391.log:709 -> EVAL step 16996`
  - `/data/home/amartyavarma/ppm/logs/ppm-26391.log:729 -> STOPPED at step 17012 (10-21 02:09 PT)`
  - `/data/home/amartyavarma/ppm/logs/ppm-27030.log:2 -> env prod_811_fresh.env`
  - `/data/home/amartyavarma/ppm/logs/ppm-27030.log:145 -> resume dcp-17012`
  - `/data/home/amartyavarma/ppm/logs/ppm-27030.log:685 -> EVAL step 17119`
  - `/data/home/amartyavarma/ppm/logs/ppm-27030.log:690 -> STOPPED at step 17120 (23:21 PT)`
  - `/data/home/amartyavarma/ppm/code-811/prod_811_fresh16861_v4.env:244 -> user: the 8,1,1 run overfit because epoch 2 repeats data -> fresh data`
  - `/data/home/amartyavarma/ppm/code-811/fresh/STATUS.md:8-9 -> 27030 stopped by user order; dcp-17120 moved to ckpt-old811f (not exported)`
  - `/data/home/amartyavarma/ppm/code-811/fresh/STATUS.md:17 -> chat eval rose 0.59 -> 0.82 on the 17012 branch (text flip)`
  - `/data/home/amartyavarma/ppm/code-811/fresh/STATUS.md:19 -> HF step names dcp-16914/16967/17012 hold the OLD branch`
  - `/data/home/amartyavarma/setup-logs/hfb/records.jsonl:189-191 -> dcp-16914/16967/17012 job 26391 ppm-v1-b`
  - `/data/home/amartyavarma/setup-logs/hfb/records.jsonl:194 -> dcp-17120 job 27030 ppm-v1-b PASS`

### ppm811-falsestarts

**8,1,1 fresh-from-16861 false starts (27100 chat flipped, 27105 nochatflip, 27110 v3)**  
group `ppm811` · status **abandoned** · dir `/data/home/amartyavarma/ppm/code-811`

- **Jobs:** 27100, 27105, 27110
- **Dates:** 2026-10-21 23:26 PT .. 2026-10-22 01:46 PT
- **Base model:** unknown
- **Init:** ppm32b dcp-16861 (explicit PPM_RESUME_FROM)
- **Parents (resolved):** ppm32b@16861
- **Produces:** none recorded
- **Data:** fresh pose plan 1 from 16861. 27100 (prod_811_fresh16861.env): Gemma and chat both role-flipped. 27105 (_nochatflip.env): chat unflipped. 27110 (_v3.env): only exhausted data flipped, plus unused chat split 1.
- **Recipe:** code-811, 8,1,1 schedule as ppm811-v4; envs prod_811_fresh16861.env / _nochatflip.env / _v3.env
- **Steps:** 16861 -> 16967 (27100); 16861 -> 16865 (27105); 16861 -> 16862 (27110)
- **Evals:** norm_word_ce = @16967 (27100): pose rows 1.8810, speech 1.9631, gemma 0.7696, chat 0.7514 (vs 0.5909 at 16861); pose_cb0_ce = 1.6696 @16967; pose_light_ce = 3.8217 @16967
- **Checkpoints / exports:** dcp-16865 (HF ppm-v1-b; the record names job 27100); dcp-16967 (27100)
- **Local paths:** unknown
- **HF repos:** [InternalCan/ppm-v1-b](https://huggingface.co/InternalCan/ppm-v1-b)
- **Notes:** Abandoned within hours as the user corrected the flip rules; superseded by v4 (27114).
- **Citations:**
  - `/data/home/amartyavarma/ppm/logs/ppm-27100.log:3 -> env prod_811_fresh16861.env, 10-21 23:26 PT`
  - `/data/home/amartyavarma/ppm/logs/ppm-27100.log:688 -> EVAL step 16967; :692 STOPPED at step 16967`
  - `/data/home/amartyavarma/ppm/logs/ppm-27105.log:579 -> STOPPED at step 16865`
  - `/data/home/amartyavarma/ppm/logs/ppm-27110.log:565 -> STOPPED at step 16862`
  - `/data/home/amartyavarma/ppm/code-811/prod_811_fresh16861_v4.env:262-276 -> restart history: chat flip reverted (user 01:35 PT), only exhausted data flipped (01:50), v4 = podcast flipped from 16861`
  - `/data/home/amartyavarma/setup-logs/hfb/records.jsonl:195 -> dcp-16865 ppm-v1-b`

### ppm811-v4

**8,1,1 fresh v4 (fresh pose plan 1; podcast flipped from 16861)**  
group `ppm811` · status **finished** · dir `/data/home/amartyavarma/ppm/code-811`

- **Jobs:** 27114
- **Dates:** 2026-10-22 01:47 PT .. 2026-10-22 11:23 PT
- **Base model:** unknown
- **Init:** ppm32b dcp-16861 (/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-16861, explicit PPM_RESUME_FROM); the start eval equals PPM's eval at 16861 on 184 common metrics
- **Parents (resolved):** ppm32b@16861
- **Produces:** ppm811@17352
- **Data:** Pose: fresh new-pose-llm-data plan 1 (85,076 h, 528,857 rows; Mimi lufs23-local-v1; deduped vs PPM TikTok and eval videos); 8 pose rows per step, each used once. Speech: every podcast row flipped A<->B from 16861. Gemma txtdup unflipped. Chat unflipped, repeats replaced by unused corpus15m split 1.
- **Recipe:** code-811 train_ppm.py + ppm811.sbatch; env prod_811_fresh16861_v4.env. PPM_S811=1; GA pose*8,speech,text; BB_MODE full,frozen*7,full,full (one backbone-full pose micro); backbone task weights 1/2/1; DEC_K 8; decp_bb link OFF; LRs constant as PPM; host alias slinky-7:slinky-10; max_steps 17352; ~67 s/step
- **Steps:** 16861 -> 17352 (target 17352; ended when the fresh pose ran out)
- **Evals:** norm_word_ce = @17352: pose rows 1.8928, speech 1.9649, gemma 0.7677, chat 0.5900, speech_s3 2.0371; pose_cb0_ce = 1.6531 @17352; pose_light_ce = 3.7329 @17352 (start eval @16861 was 4.8819 because the link is off); other = speech cb0 0.7180, speech light 2.2335 @17352
- **Checkpoints / exports:** dcp-17352 (final save; HF ppm-v1-c, 320 files, PASS); intermediate dcp-16914-j27114 .. dcp-17333 (HF ppm-v1-b/-c)
- **Local paths:** /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17352 (current presence unknown)
- **HF repos:** [InternalCan/ppm-v1-c](https://huggingface.co/InternalCan/ppm-v1-c), [InternalCan/ppm-v1-b](https://huggingface.co/InternalCan/ppm-v1-b)
- **Notes:** Main 8,1,1 lineage segment 1. Same run_base ppm-v1, ckpt dir and HF chain as pure PPM; only the env in the job log tells them apart. The pose eval set is PPM's old TikTok-encode pose eval.
- **Citations:**
  - `/data/home/amartyavarma/ppm/logs/ppm-27114.log:3 -> job 27114, 10-22 01:47:12 PT, env prod_811_fresh16861_v4.env`
  - `/data/home/amartyavarma/ppm/logs/ppm-27114.log:146 -> resume dcp-16861`
  - `/data/home/amartyavarma/ppm/logs/ppm-27114.log:165 -> START EVAL vs PPM's eval at 16861: 184 common metrics`
  - `/data/home/amartyavarma/ppm/logs/ppm-27114.log:164 -> EVAL step 16861 (pose light 4.8819)`
  - `/data/home/amartyavarma/ppm/logs/ppm-27114.log:1099 -> EVAL step 17352`
  - `/data/home/amartyavarma/ppm/logs/ppm-27114.log:1102 -> fresh pose exhausted at step 17352; final save dcp-17352`
  - `/data/home/amartyavarma/ppm/logs/ppm-27114.log:902 -> export w-17171 REFUSED (/data floor)`
  - `/data/home/amartyavarma/ppm/code-811/prod_811_fresh16861_v4.env:13,183,206,209,249,252-276,94 -> knobs (GA, link off, S811, BB_MODE, alias, fresh/flip block, max 17352)`
  - `/data/home/amartyavarma/ppm/code-811/fresh/STATUS.md:6 -> fresh pose 85,076 h / 528,857 rows`
  - `/data/home/amartyavarma/ppm/code-811/README_811.md:3,21-31,42-52 -> purpose, schedule, weights`
  - `/data/home/amartyavarma/setup-logs/hfb/records.jsonl:208 -> dcp-17352 job 27114 InternalCan/ppm-v1-c PASS`
  - `/data/home/amartyavarma/setup-logs/hfb/records.jsonl:198 -> dcp-16914-j27114 ppm-v1-b`
  - `memory:pose-data-append-plan.md:13 -> v4 job 27114 ends at 17352 with a final save`
  - `memory:mimi-lufs23-local-correct.md:11-13 -> fresh data Mimi lufs23-local-v1 = correct encode`

### ppm811-append2

**8,1,1 delivery-2 append (npl2)**  
group `ppm811` · status **finished** · dir `/data/home/amartyavarma/ppm/code-811`

- **Jobs:** 27204
- **Dates:** 2026-10-22 11:37 PT .. 2026-10-22 17:20 PT
- **Base model:** unknown
- **Init:** ppm811 dcp-17352 (27114 final save)
- **Parents (resolved):** ppm811-v4@17352
- **Produces:** ppm811@17641
- **Data:** Pose: unused plan-1 tail rows first (TAIL=1), then npl2 = delivery 2 (hf://buckets/InternalCan/pose-llm-data/new-pose-llm-data-delivery2; 271,367 rows, 43,661 h). Other streams as v4.
- **Recipe:** env prod_811_fresh_append.env = v4 + PPM_FRESH_POSE_ROOT2=npl2, PPM_FRESH_POSE_TAIL=1, max_steps 17641; recipe otherwise = v4
- **Steps:** 17352 -> 17641 (target 17641; pose exhausted)
- **Evals:** norm_word_ce = @17641: pose rows 1.8987, speech 1.9667, gemma 0.7689, chat 0.5897; pose_cb0_ce = 1.6499 @17641; pose_light_ce = 3.7302 @17641; other = speech cb0 0.7175, speech light 2.2380 @17641
- **Checkpoints / exports:** dcp-17641 (final save; HF ppm-v1-c PASS)
- **Local paths:** /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17641 (current presence unknown); /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17641
- **HF repos:** [InternalCan/ppm-v1-c](https://huggingface.co/InternalCan/ppm-v1-c)
- **Notes:** Main 8,1,1 lineage segment 2.
- **Citations:**
  - `/data/home/amartyavarma/ppm/logs/ppm-27204.log:3 -> 10-22 11:37:21 PT, env prod_811_fresh_append.env`
  - `/data/home/amartyavarma/ppm/logs/ppm-27204.log:147 -> resume dcp-17352`
  - `/data/home/amartyavarma/ppm/logs/ppm-27204.log:881 -> EVAL step 17641`
  - `/data/home/amartyavarma/ppm/logs/ppm-27204.log:884 -> pose exhausted at 17641; final save dcp-17641`
  - `/data/home/amartyavarma/ppm/code-811/prod_811_fresh_append.env:289-293 -> ROOT2 npl2, TAIL 1, MAX_STEPS 17641`
  - `/data/home/amartyavarma/ppm/code-811/fresh/append/npl2/SUMMARY.md:3 -> source delivery2`
  - `/data/home/amartyavarma/ppm/code-811/fresh/append/npl2/SUMMARY.md:6-7 -> rows [528857, 271367], hours [85076, 43661]; ends [17352, 17641]`
  - `/data/home/amartyavarma/ppm/code-811/fresh/append/LAUNCHED.txt:1 -> 27204 27205 27206 17641`
  - `/data/home/amartyavarma/setup-logs/hfb/records.jsonl:214 -> dcp-17641 job 27204 ppm-v1-c PASS`
  - `memory:pose-data-append-plan.md:27 -> delivery 2 appended at 17352, run 27204 to 17641`

### ppm811-append3

**8,1,1 delivery-3 append (npl3); paused at 17850**  
group `ppm811` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/code-811`

- **Jobs:** 27653
- **Dates:** 2026-10-22 18:22 PT .. 2026-10-22 22:29 PT
- **Base model:** unknown
- **Init:** ppm811 dcp-17641 (27204 final save)
- **Parents (resolved):** ppm811-append2@17641
- **Produces:** ppm811@17850
- **Data:** Pose: unused npl2 tail rows first (TAIL=1), then npl3 = delivery 3 (hf://buckets/InternalCan/pose-llm-data/new-pose-llm-data-delivery3; 395,285 rows, 63,606 h). Other streams as v4.
- **Recipe:** env prod_811_fresh_append3.env = prod_811_fresh_append.env + ROOT2 npl2,npl3, max_steps 18022; recipe otherwise = v4
- **Steps:** 17641 -> 17850 (target 18022, never reached on this lineage)
- **Evals:** norm_word_ce = last EVAL @17828: pose rows 1.9020, speech 1.9656, gemma 0.7672, chat 0.5900. The model at 17850 (resume eval in 31101): speech 1.9658, gemma 0.7678, chat 0.5896, pose rows 1.9017.; pose_cb0_ce = 1.6453 @17828; 1.6463 @17850; pose_light_ce = 3.7153 @17828; 3.7304 @17850; genghisbench = local-17850: 4/8 successes on a partial run (12 of 20 seeds missing, median 13.0); other = speech cb0 0.7174, speech light 2.2305 @17828
- **Checkpoints / exports:** dcp-17850 (pause save; HF bundle ppm-v1-d, shards ppm-v1-c, 320 files 737.4 GB PASS)
- **Local paths:** /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17850 (deleted on most hosts 10-25); /scratch/amartyavarma/pose5-resume/dcp-17850; ckpt-keep-pose5/dcp-17850 pins on slinky-1/2/7/14; /scratch/amartyavarma/ppmvox/init-17850-slots/dcp-17850 (slinky-15); /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17850
- **HF repos:** [InternalCan/ppm-v1-c](https://huggingface.co/InternalCan/ppm-v1-c), [InternalCan/ppm-v1-d](https://huggingface.co/InternalCan/ppm-v1-d)
- **Notes:** Memory says 27653 ran to 18022; it actually stopped at 17850 (STOP_AFTER_SAVE, pause for the monoplex FT + decay branch) and the main lineage never trained 17850-18022 on npl3 alone. dcp-17850 is the parent of the decay branch, pose5 and (per other agents) the mono17850 / Xabi main-17850 FTs.
- **Citations:**
  - `/data/home/amartyavarma/ppm/logs/ppm-27653.log:3 -> 10-22 18:22:16 PT, env prod_811_fresh_append3.env`
  - `/data/home/amartyavarma/ppm/logs/ppm-27653.log:147 -> resume dcp-17641`
  - `/data/home/amartyavarma/ppm/logs/ppm-27653.log:771 -> EVAL step 17828`
  - `/data/home/amartyavarma/ppm/logs/ppm-27653.log:796-797 -> STOP_AFTER_SAVE / STOPPED at step 17850 (22:29 PT)`
  - `/data/home/amartyavarma/ppm/logs/ppm-31101.log:169 -> EVAL step 17850 (resume eval of dcp-17850)`
  - `/data/home/amartyavarma/ppm/code-811/prod_811_fresh_append3.env:296-308 -> delivery-3 block, ROOT2 npl2,npl3, MAX_STEPS 18022`
  - `/data/home/amartyavarma/ppm/code-811/fresh/append/npl3/SUMMARY.md:3,6-7 -> delivery3 source; 395,285 rows / 63,606 h; ends [17352, 17641, 18022]`
  - `/data/home/amartyavarma/ppm/code-811/fresh/append/LAUNCHED3.txt:1 -> 27653 27654 27655 18022`
  - `/data/home/amartyavarma/ppm/code-811/fresh/EVALS_append3.md:5-12 -> eval table 17641..17828`
  - `/data/home/amartyavarma/ppm/decay/PROMPT.md:7 -> main run paused at its ~22:29 PT hourly save (for the monoplex FT, then the decay branch)`
  - `/data/home/amartyavarma/ppm/control/OPS_PLAYBOOK.md:110 -> FT on 8 hosts, then the decay branch on all 16`
  - `/data/home/amartyavarma/setup-logs/hfb/records.jsonl:218-219 -> repo rollover to ppm-v1-d; dcp-17850 repo ppm-v1-d, shard_repo ppm-v1-c, PASS`
  - `/data/home/amartyavarma/ppm/pose5/STATUS.md:52-56 -> dcp-17850 HF location; node-local D1-deleted 10-25 on 11 hosts; slinky-15 copy`
  - `/data/home/amartyavarma/ppm/pose5/STATUS.md:716 -> pins ckpt-keep-pose5/dcp-17850 on s1/2/7/14`
  - `/data/home/amartyavarma/ppm/gb/ppm-v1-local-17850/gb_s100-119.json:4-7 -> GenghisBench 4/8, 12 seeds missing`
  - `memory:pose-data-append-plan.md:28 -> delivery 3 run 27653 (planned to 18022)`

### ppm811-decay

**8,1,1 LR-decay branch (linear to 0 at 18022), ppmdecay**  
group `ppm811` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/code-811-decay`

- **Jobs:** 27810
- **Dates:** 2026-10-22 23:36 PT .. 2026-10-23 01:11 PT
- **Base model:** unknown
- **Init:** ppm811 dcp-17850 (/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17850)
- **Parents (resolved):** ppm811-append3@17850
- **Produces:** ppm811-decay@17928
- **Data:** same stream as ppm811-append3 (npl3 continuation)
- **Recipe:** code-811-decay (copy of code-811 + PPM_LR_DECAY_FROM/_TO knob); env prod_811_decay.env = prod_811_fresh_append3.env + PPM_LR_DECAY_FROM=17850, PPM_LR_DECAY_TO=18022 (every LR group linear to 0), PPM_RUN=ppmdecay; 16 hosts; ~68 s/step
- **Steps:** 17850 -> 17928 (target 18022; LR multiplier 0.6337 at the last eval)
- **Evals:** norm_word_ce = @17914: pose rows 1.8999, speech 1.9614, gemma 0.7613, chat 0.5864 (no constant-LR eval at matched steps; main's last was @17828: 1.9020 / 1.9656 / 0.7672 / 0.5900); pose_cb0_ce = 1.6418 @17914; pose_light_ce = 3.6917 @17914; other = speech cb0 0.7142, speech light 2.2152 @17914
- **Checkpoints / exports:** dcp-17928 (stop save) -> HF InternalCan/ppm-decay-17928 (17/17 PASS, 390 files, 737.4 GB)
- **Local paths:** /scratch/amartyavarma/ppm/ckpt/ppmdecay/dcp-17928 (node-local copy deleted 10-23 02:12 PT)
- **HF repos:** [InternalCan/ppm-decay-17928](https://huggingface.co/InternalCan/ppm-decay-17928)
- **Notes:** Separate lineage (run_base ppmdecay). It decayed only about 37% of the way (multiplier 0.634 at the 17914 eval) before the stop, so it is not a fully annealed model. Nothing downstream is known to use it.
- **Citations:**
  - `/data/home/amartyavarma/ppm/decay/PROMPT.md:3-11 -> goal: branch the main run from dcp-S, decay all LRs linearly to 0 at 18022`
  - `/data/home/amartyavarma/ppm/code-811-decay/prod_811_decay.env:316-317,320 -> LR_DECAY_FROM 17850, TO 18022, PPM_RUN ppmdecay`
  - `/data/home/amartyavarma/ppm/logs/ppmdecay-27810.log:2 -> 10-22 23:36:51 PT, env prod_811_decay.env`
  - `/data/home/amartyavarma/ppm/logs/ppmdecay-27810.log:146 -> resume dcp-17850`
  - `/data/home/amartyavarma/ppm/logs/ppmdecay-27810.log:640 -> EVAL step 17914`
  - `/data/home/amartyavarma/ppm/logs/ppmdecay-27810.log:658 -> STOPPED at step 17928 (10-23 01:11 PT)`
  - `/data/home/amartyavarma/ppm/code-811-decay/EVALS_decay.md:7-10 -> decay evals 17850..17914, lr mult 0.6337`
  - `/data/home/amartyavarma/ppm/control/OPS_PLAYBOOK.md:116 -> stopped at 17928 (user: nodes to the 14B run); HF ppm-decay-17928 verified; node-local deleted 02:12 PT`
  - `/data/home/amartyavarma/ppm/code-811-decay/hfb/logs/hfb_decay_final-28015.log:22 -> HF backup DONE dcp-17928 -> InternalCan/ppm-decay-17928, 17/17 PASS`

### pose5

**pose5: 8,1,1 continuation from dcp-17850 + delivery 5 (npl5) + s2p cb0 + G0COND**  
group `pose` · status **paused** · dir `/data/home/amartyavarma/ppm/pose5`

- **Jobs:** 31101
- **Dates:** 2026-10-27 11:36 PT .. 2026-10-27 16:27 PT (paused)
- **Base model:** unknown
- **Init:** ppm811 dcp-17850 (/scratch/amartyavarma/pose5-resume/dcp-17850); tower_p.s_emb (s2p) and inp.g0 (G0COND) added zero-init in their own optimizer groups
- **Parents (resolved):** ppm811-append3@17850
- **Produces:** pose5@18098
- **Data:** Fresh pose plans [plan 1, npl2, npl3, npl5]. Steps 17850-18021 read exactly 27653's npl3 rows, then each rank's unused npl3 tail, then npl5 = delivery 5 (hf://buckets/InternalCan/pose-llm-data/new-pose-llm-data-delivery5; 1,391,232 rows, 223,599 h; npl5 starts at steps 18022-18034). Podcast flipped; Gemma and chat unflipped as v4.
- **Recipe:** code pose5/code-811-s2p; env prod_811_fresh_append5_s2p.env (= prod_811_fresh_append3.env verbatim + the POSE5 block). PPM_S2P=cb0: zero-init speech-cb0 embedding into the pose tower at the same frame. PPM_G0COND=1: zero-init g_proj of frame-0 root xy + shoulder width at each clip's SLOT. PPM_MAX_STEPS 19382, KEEP_CKPT 1, host alias slinky-7:slinky-10,slinky-15:slinky-9. 8,1,1 schedule, constant LRs; ~67.3 s/step.
- **Steps:** 17850 -> 18098 (target 19382)
- **Evals:** norm_word_ce = @18092: pose rows 1.9069, speech 1.9675, gemma 0.7669, chat 0.5899 (@17850: 1.9017 / 1.9658 / 0.7678 / 0.5896); pose_cb0_ce = 1.6431 @18092 (1.6463 @17850); pose_light_ce = 3.7045 @18092 (3.7304 @17850); genghisbench = local-18098 run dir exists but has 0 attempts (not run yet); other = speech cb0 0.7175, speech light 2.2265 @18092
- **Checkpoints / exports:** dcp-17903, dcp-17956, dcp-18009, dcp-18062 (rotated hourly saves; HF ppm-v1-d); dcp-18098 (pause save; COMPLETE 16/16, HF ppm-v1-d PASS)
- **Local paths:** /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-18098; /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-18098 (pose6 bf16 conversion)
- **HF repos:** [InternalCan/ppm-v1-d](https://huggingface.co/InternalCan/ppm-v1-d)
- **Notes:** The same 8,1,1 lineage (run_base ppm-v1); dcp-18098 is also callable ppm811@18098. It ran only ~64-76 steps into npl5 before the pause. It resumes from dcp-18098 after motvox's hand-back, but RESUME_HOLD is in place. No Weka export was written: the first was due ~17:40 PT, after the pause. The PT stamps in the log (16:27) and the PAUSED.json times (18:18 / 18:25) differ, presumably because PAUSED.json was backfilled at 18:19 (STATUS.md 18:19 entry).
- **Citations:**
  - `/data/home/amartyavarma/ppm/pose5/PROMPT.md:3-11 -> user request; delivery 5 source, ~230k h, zero overlap`
  - `/data/home/amartyavarma/ppm/pose5/prod_811_fresh_append5_s2p.env:310-344 -> POSE5 block (ROOT2 +npl5 :324, S2P cb0 :330, MAX_STEPS 19382 :333, alias :337, G0COND :341, KEEP_CKPT 1 :344)`
  - `/data/home/amartyavarma/ppm/pose5/npl5/SUMMARY.md:3 -> delivery5 source, 552,329 releases`
  - `/data/home/amartyavarma/ppm/pose5/npl5/SUMMARY.md:8-9 -> npl5 1,391,232 rows / 223,599 h; ends [17352, 17641, 18022, 19382]; npl5 starts at steps [18022, 18034]`
  - `/data/home/amartyavarma/ppm/logs/ppm-31101.log:3 -> job 31101, 10-27 11:36:18 PT, env pose5/kit/state/run.env (same knobs as prod_811_fresh_append5_s2p.env)`
  - `/data/home/amartyavarma/ppm/logs/ppm-31101.log:42 -> FRESH POSE APPEND 4 plans`
  - `/data/home/amartyavarma/ppm/logs/ppm-31101.log:149 -> resume /scratch/amartyavarma/pose5-resume/dcp-17850; inp.g0 / tower_p.s_emb not in ckpt (init)`
  - `/data/home/amartyavarma/ppm/logs/ppm-31101.log:172 -> code pose5/code-811-s2p/train_ppm.py`
  - `/data/home/amartyavarma/ppm/logs/ppm-31101.log:169 -> EVAL step 17850; :835 -> EVAL step 18092`
  - `/data/home/amartyavarma/ppm/logs/ppm-31101.log:842-846 -> save dcp-18098; STOP_AFTER_SAVE; STOPPED at step 18098 (16:27 PT); done 19382 (target)`
  - `/data/home/amartyavarma/ppm/pose5/EVALS_pose5.md:5-14 -> eval table 17850..18092`
  - `/data/home/amartyavarma/ppm/pose5/STATUS.md:553 -> stage 31091 aborted; ppm811 31092 never started`
  - `/data/home/amartyavarma/ppm/pose5/STATUS.md:584-588 -> 31101 RUNNING; POSE RUN LIVE`
  - `/data/home/amartyavarma/ppm/pose5/STATUS.md:756 -> PAUSED at step 18098, dcp-18098 COMPLETE 16/16`
  - `/data/home/amartyavarma/ppm/pose5/STATUS.md:757 -> pose6 converts dcp-18098 -> ckpt-infer/ppm-v1/local-18098`
  - `/data/home/amartyavarma/ppm/pose5/STATUS.md:758 -> dcp-18098 HF-VERIFIED 16/16`
  - `/data/home/amartyavarma/ppm/pose5/STATUS.md:806 -> user: do not resume; motvox runs from this pause; RESUME_HOLD written`
  - `/data/home/amartyavarma/ppm/pose5/kit/state/PAUSED.json:2 -> step 18098, paused_job 31101, complete_16 + hf_verified_16 true`
  - `/data/home/amartyavarma/setup-logs/hfb/records.jsonl:220-224 -> dcp-17903/17956/18009/18062/18098 job 31101 ppm-v1-d`
  - `/data/home/amartyavarma/ppm/gb/ppm-v1-local-18098/gb_s100-119.json:4-5 -> 0 successes, n 0`
  - `memory:pose5-pause-mot-plan.md:13-16,45-48 -> plan and pause at 18098`

### ppm14b

**ppm14b (PPM architecture on Qwen3-14B, speech + text only; head-to-head vs INTL 14B)**  
group `ppm14b` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/ppm14b` · also audited in part_14b.json

- **Jobs:** 26513, 26559, 26560, 26561, 26562, 26571, 28012
- **Dates:** 2026-10-21 05:06 PT .. 2026-10-21 20:27 PT (stopped at 4304); mistaken resume 2026-10-23 01:21 .. 02:00 PT (to 4502)
- **Base model:** Qwen/Qwen3-14B (local /data/home/amartyavarma/models/Qwen3-14B)
- **Init:** fresh from Qwen3-14B: 444 tensors loaded (embedding copied to A and B), 724 random tensors, seed 20261011
- **Parents (resolved):** base:qwen3-14b
- **Produces:** ppm14b@2208, ppm14b@4091, ppm14b@4304, ppm14b@4502
- **Data:** PPM's speech + text data, order and seeds (GA speech,speech,text = PPM's GA minus pose), D2 extra rows from 170; no pose
- **Recipe:** ppm14b/code + ppm14b.env (= next-arch/next.env verbatim except the P14 lines: ILP_MODE=sum, PPM's 4B speech tower and 28-layer 0.6B decoder, plain RoPE); 40-layer Qwen3-14B backbone; LR trunk 5e-5; constant LR; max_steps 17400 (log /17366); ~11.6 s/step
- **Steps:** 0 -> 4304 (26571, intended stop) -> 4502 (28012, mistaken resume); target 17366
- **Evals:** norm_word_ce = @4460: speech 2.0859, gemma 0.8580, chat 0.6416, speech_s3 2.1599 (@4283: 2.0868 / 0.8594 / 0.6426); other = speech cb0 0.7832, speech light 2.3387 @4460
- **Checkpoints / exports:** w-2208, w-4091 (Weka exports; HF InternalCan/ppm14b, w-4091 PASS); dcp-4304; dcp-4502 (node-local)
- **Local paths:** /scratch/amartyavarma/ppm14b/ckpt/ppm14b/dcp-4304; /scratch/amartyavarma/ppm14b/ckpt/ppm14b/dcp-4502; /data/home/amartyavarma/ppm/ppm14b/ckpt-weka/ppm14b/w-4091
- **HF repos:** [InternalCan/ppm14b](https://huggingface.co/InternalCan/ppm14b)
- **Notes:** Memory says ppm14b stopped at dcp-4304, but the mistaken 10-23 resume (job 28012) trained on to dcp-4502 before STOP_AFTER_SAVE. Treat 4304 as the intended stop and 4502 as an accidental extension. Whether dcp-4304/4502 are on HF is unknown: only w-2208 and w-4091 have HF backup records. \|\| Second independent audit (part_14b.json): Not in the brief's list but part of the 14B family; another part may also own group ppm14b. Memory says it stopped at dcp-4304, but the logs show a mistaken resume on 10-23 (job 28012) that stopped with save at 4502. Steps 317-477 of 26513 were discarded when it relaunched from dcp-316.
- **Citations:**
  - `/data/home/amartyavarma/ppm/ppm14b/ppm14b.env:1-8 -> purpose and diff vs next.env`
  - `/data/home/amartyavarma/ppm/ppm14b/ppm14b.env:10,19-20,41,104,118 -> PPM_RUN ppm14b, qwen3-14b, 40 layers, LR, MAX_STEPS 17400, save root`
  - `/data/home/amartyavarma/ppm/ppm14b/logs/ppm14b-26513.log:21 -> [init] backbone <- /data/home/amartyavarma/models/Qwen3-14B`
  - `/data/home/amartyavarma/ppm/ppm14b/logs/ppm14b-26571.log:160 -> resume dcp-316`
  - `/data/home/amartyavarma/ppm/ppm14b/logs/ppm14b-26571.log:2468 -> export w-2208; :4376 -> export w-4091`
  - `/data/home/amartyavarma/ppm/ppm14b/logs/ppm14b-26571.log:4572 -> EVAL step 4283`
  - `/data/home/amartyavarma/ppm/ppm14b/logs/ppm14b-26571.log:4597 -> STOPPED at step 4304 (10-21 20:27 PT)`
  - `/data/home/amartyavarma/ppm/ppm14b/logs/ppm14b-28012.log:160 -> 10-23 01:21 PT resume dcp-4304`
  - `/data/home/amartyavarma/ppm/ppm14b/logs/ppm14b-28012.log:712 -> EVAL step 4460`
  - `/data/home/amartyavarma/ppm/ppm14b/logs/ppm14b-28012.log:757 -> STOPPED at step 4502 (02:00 PT)`
  - `/data/home/amartyavarma/ppm/ppm14b/hf_backup/HF_BACKUP-w-4091.json:2,62 -> repo InternalCan/ppm14b, PASS`
  - `/data/home/amartyavarma/ppm/ppm14b/RELAUNCH_PROMPT.md:7-10 -> stopped on purpose at 4304 on 10-22; resume plan`
  - `memory:intl-14b-is-ppm-next.md:11-15 -> ppm14b was resumed by mistake on 10-23 (the user meant ppm-next); ppm14b stopped dcp-4304`
  - `/data/home/amartyavarma/ppm/ppm14b/ppm14b.env:1-6 -> definition: PPM arch on Qwen3-14B, head-to-head vs INTL, the differing knobs`
  - `/data/home/amartyavarma/ppm/ppm14b/logs/ppm14b-26513.log:17,21,38 -> job 26513 10-21 05:05 PT, base init, step-0 eval 3.8509`
  - `/data/home/amartyavarma/ppm/ppm14b/control/RELAUNCH_NOTE.txt -> 26513 hung at step 477 10-21 06:47 PT; relaunched from the newest dcp`
  - `/data/home/amartyavarma/ppm/ppm14b/logs/ppm14b-26560.log:29 -> cfgdiff abort (xdep)`
  - `/data/home/amartyavarma/ppm/ppm14b/logs/ppm14b-26571.log:161,4597 -> resumed dcp-316, STOPPED at 4304 10-21 20:27 PT`
  - `/data/home/amartyavarma/ppm/ppm14b/control/STOP_REASON.txt -> stopped to free 16 nodes for the new-pose-llm-data download`
  - `/data/home/amartyavarma/ppm/ppm14b/logs/ppm14b-28012.log:19,161,757 -> 10-23 01:20 PT resumed dcp-4304, STOPPED at 4502 02:00 PT`
  - `/data/home/amartyavarma/ppm/ppm14b/control/STOP_AFTER_SAVE.done-4502 -> 'wrong run - the user wanted the INTL 14B'`
  - `/data/home/amartyavarma/ppm/ppm14b/logs/ppm14b-28012.log:712 -> EVAL step 4460 2.0859`
  - `/data/home/amartyavarma/ppm/next-arch/logs/ppm-next-23277.log:1311 -> INTL EVAL step 4433 2.0186`
  - `/data/home/amartyavarma/ppm/mot/logs/ppm-mot-29514.log:1768 -> MoT EVAL step 4453 2.0010`
  - `/data/home/amartyavarma/ppm/ppm14b/hf_backup/HF_BACKUP-w-4091.json:2 -> InternalCan/ppm14b`
  - `memory:intl-14b-is-ppm-next.md:15 -> memory says ppm14b stopped at dcp-4304`

### ppm-m128r

**PPM on Mistral Medium 3.5 128B, 'io' train set (backbone frozen), 2-node spare run**  
group `other` · status **abandoned** · dir `/data/home/amartyavarma/ppm/ppm-mistral128b`

- **Jobs:** 26678, 26764, 26776
- **Dates:** 2026-10-21 14:53 PT .. 2026-10-21 20:18 PT
- **Base model:** Mistral Medium 3.5 128B (local /data/home/amartyavarma/models/mistral-m35-128b-bf16)
- **Init:** fresh from mistral-m35-128b-bf16: 796 tensors (vision tower dropped), 1440 random tensors, seed 20261011
- **Parents (resolved):** base:mistral-medium-3.5-128b
- **Produces:** ppm-m128r@210, ppm-m128r@216
- **Data:** PPM's recipe on node-local subsets for slinky-15 + slinky-18 (pose = 2 disjoint 32-shard subsets of slinky-10's store; speech slinky-18's store / a slinky-10 subset; Mistral token lanes)
- **Recipe:** ppm-mistral128b/code + prod_m128r.env (= prod_m128.env + [M128R]): PPM_TRAIN_SET=io (all 88 backbone layers frozen, everything else trains as PPM); decoders FSDP; 2 nodes / 16 GPUs; from step 210, layers 0-3 train (trunk_io group, 100-step warmup); max_steps 2156; ~70 s/step
- **Steps:** 0 -> 210 (26678) -> 213 (26764, diverged, cancelled without a save) -> 216 (26776)
- **Evals:** norm_word_ce = @210: pose rows 3.7691, speech 3.0669, gemma 1.6581, chat 1.1269; pose_cb0_ce = 7.6264 @210 (did not learn); pose_light_ce = 7.5547 @210
- **Checkpoints / exports:** dcp-210 (all-frozen); dcp-216
- **Local paths:** /scratch/amartyavarma/ppm/ckpt/ppm-m128r-v1/dcp-216 (slinky-15, slinky-18); /scratch/amartyavarma/ppm/ckpt-keep/ppm-m128r-v1/dcp-210
- **HF repos:** [InternalCan/ppm-m128r-v1](https://huggingface.co/InternalCan/ppm-m128r-v1) *(not in model listing: dataset/bucket or deleted)*
- **Notes:** Abandoned by the user (Mistral not to be trained). With the backbone frozen the model did not learn to read audio or pose (pose cb0 stayed ~7.6). No HF copy.
- **Citations:**
  - `/data/home/amartyavarma/ppm/ppm-mistral128b/README.md:6-9 -> PPM exactly with the Mistral 128B backbone`
  - `/data/home/amartyavarma/ppm/ppm-mistral128b/code/prod_m128r.env:1-10 -> io train set, backbone frozen, 2 spare nodes`
  - `/data/home/amartyavarma/ppm/ppm-mistral128b/logs/m128r-26678.log:1 -> job 26678 10-21 14:53:50 PT, env prod_m128r.env`
  - `/data/home/amartyavarma/ppm/ppm-mistral128b/logs/m128r-26678.log:20 -> [init] backbone <- mistral-m35-128b-bf16`
  - `/data/home/amartyavarma/ppm/ppm-mistral128b/logs/m128r-26678.log:309 -> STOPPED at step 210`
  - `/data/home/amartyavarma/ppm/ppm-mistral128b/logs/m128r-26776.log:36 -> EVAL step 210`
  - `/data/home/amartyavarma/ppm/ppm-mistral128b/logs/m128r-26776.log:94 -> STOPPED at step 216 (20:17 PT)`
  - `/data/home/amartyavarma/ppm/ppm-mistral128b/README.md:189-203 -> switch at 210 (layers 0-3), 26764 diverged, stopped at 216 on the user's word (Mistral not to be trained, only optimised); saves kept`
  - `/data/home/amartyavarma/ppm/ppm-mistral128b/hf_backup/hf_one.sh:4 -> repo InternalCan/ppm-m128r-v1`
  - `/data/home/amartyavarma/ppm/ppm-mistral128b/hf_backup/STATUS.txt (last line) -> exports: none, verified: none`

### ppm-misc-notes

**Other PPM-scope dirs that did not produce a training lineage**  
group `other` · status **unknown** · dir `/data/home/amartyavarma/ppm`

- **Jobs:** unknown
- **Dates:** unknown
- **Base model:** unknown
- **Init:** n/a
- **Parents (resolved):** unknown
- **Produces:** none recorded
- **Data:** n/a
- **Recipe:** n/a
- **Steps:** n/a
- **Evals:** none recorded
- **Checkpoints / exports:** unknown
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Notes:** Policy here: a separate row only for dirs with a real training job and steps. Not rows: megatron-port (port/equivalence and speed tests on spares, no run); interleave-ppm (two aborted launches, 22792 at 5 steps and 22813 at 84 steps; its dcp-84 moved to /scratch/amartyavarma/ppm-ilp/ckpt-old; superseded by next-arch = intl14b, another agent's scope); interleave3 (tests only); code-frozenbb + runs-fbbtest (frozen-backbone full run never launched, only small GPU tests); code-d2 and code-speed (code snapshots of the PPM trainer for the D2 data switch at 170 and the T25 speed path, both folded into ppm32b; no own run); runs-811test (8,1,1 unit tests); abl_textemb, posedyn, scaling-restore (analysis / restore jobs); pose6 (pointers only). The decay branch, link-exp, ppm14b and Mistral-128B io runs did train and have their own rows.
- **Citations:**
  - `/data/home/amartyavarma/ppm/megatron-port/STATUS.md:3-22 -> Megatron port + speed work, bit-exact tests only, no training run`
  - `/data/home/amartyavarma/ppm/interleave-ppm/README.md:158,190-193 -> 22792 cancelled at step 5; 22813 STOP_AFTER_SAVE at step 84; relaunch cancelled (nodes to next-arch 14B)`
  - `/data/home/amartyavarma/ppm/interleave3/README.md (whole) -> design + memory probes/tests only; no launch found`
  - `/data/home/amartyavarma/ppm/code-frozenbb/fbb_full.env:1 -> frozen-backbone full-scale run 'NOT launched'; runs-fbbtest/ and logs/frozenbb/ = 2-8 GPU tests`
  - `/data/home/amartyavarma/ppm/code-811/README_811.md:125-151 -> runs-811test/ = 2-GPU reduced-depth tests (tA..tR5) of the 8,1,1 code`
  - `/data/home/amartyavarma/ppm/abl_textemb/abl.py:2-8 -> teacher-forced eval ablation on the PPM step-10386 ckpt (no training)`
  - `/data/home/amartyavarma/ppm/posedyn/onestep.py (docstring) -> 10-27 rollout analysis jobs 30858-30861 (onestep-30858.log: Traceback); no training`
  - `/data/home/amartyavarma/ppm/scaling-restore/logs/cleanup-init-scal-rmi7932-26215.log, cleanup-init-scal-rmi13533-26214.log -> restore/cleanup of init ckpts 7932 / 13533 for the scaling FTs; no training`
  - `/data/home/amartyavarma/ppm/pose6/STATUS.md:1-4,319-345 -> pose6 = inference + mono17850 / mono17850sam FTs (jobs 31240, 31443; covered by another agent); produced conversions ckpt-infer/ppm-v1/local-17850 and local-18098`

## 14B family: INTL 14B (ppm-next), ppm14b, MoT, motvox, 14B FTs and ASR adapters

| run | status | jobs | dates | init (parent node@step) | steps | headline evals | outputs |
|---|---|---|---|---|---|---|---|
| [`intl14b`](#intl14b) | stopped-with-save | 22875, 22899, 22904, 22905, 22954 (+6) | 2026-10-17 03:55 PT .. 2026-10-23 06:15 PT (trained segment… | base:qwen3-14b | 0 -> 6982 (target 17400; data plan ends 17366) | word CE(norm) 1.9902 @6981 (eval_speech/word_ce_norm, last eval = best); 1.9989 @6241 (the eval nearest the w-6242 export that the FTs use) | w-3683; w-4818; w-5951 / HF: ppm-next-14b |
| [`aloe-next-6242`](#aloe-next-6242) | finished | 23522 | 2026-10-18 14:08 PT .. 2026-10-18 14:37 PT | intl14b@6242 | 0 -> 253 (target 253) | word CE(norm) 1.8494 @253 (eval_speech/word_ce_norm; step 0 1.9022) | export-42; export-84; export-126 / HF: aloe-next-6242 |
| [`aloe-nextvox-6242`](#aloe-nextvox-6242) | finished | 24098, 24185, 24191, 24193 | 2026-10-18 16:30 PT (submit) / 16:54 PT (run 24193) .. 2026… | intl14b@6242, asr14b@480 | 0 -> 253 (target 253) | word CE(norm) 1.8558 @253 (eval_speech/word_ce_norm; step 0 2.1949; PPM reference j23081 step 0 2.0372) | export-42; export-84; export-126 / HF: aloe-nextvox-6242 |
| [`asr14b`](#asr14b) | stopped-with-save | 23526, 24039 | 2026-10-18 14:13 PT .. 2026-10-18 16:18 PT | intl14b@6242, qwen1ch-s4-asr2-ga1@3695 | 0 -> 480 (max 15200; stopped by the user) | word CE(norm) 0.3158 @480 (transcript normalised word CE on MLS dev; final = best); KL text-duplex KL vs step 0 = 0 (frozen backbone) | w-120; w-240; w-360 / HF: asr-adapter-14b |
| [`mot`](#mot) | paused | 28071, 28155, 29514, 30289 | 2026-10-23 06:15 PT .. 2026-10-27 11:27 PT | base:qwen3-14b | 0 -> 14839 (target 17400; data plan ends 17366) | word CE(norm) 1.9239 @14810 (eval_speech/word_ce_norm, last eval); best 1.9226 @14471 | w-1493; w-2486; w-4267 / HF: ppm-mot-14b, ppm-mot-14b-state |
| [`motasr`](#motasr) | paused | 30741 | 2026-10-26 12:55 PT .. 2026-10-26 16:34 PT | mot@10830, qwen1ch-s4-asr2-ga1@3695 | 0 -> 1220 (max 15200) | word CE(norm) 0.2657 @1200 (best, transcript normalised word CE; WER 0.193); KL text-duplex KL vs step 0 = 0 | w-960; w-1080; w-1200 / HF: asr-adapter-mot |
| [`motasr2`](#motasr2) | stopped-with-save | 31090 | 2026-10-27 11:28 PT .. 2026-10-27 17:34 PT | mot@14810, qwen1ch-s4-asr2-ga1@3695 | 0 -> 2094 (max 15200; stopped by the user) | word CE(norm) 0.2527 @1920 (best; transcript normalised word CE, WER 0.554); last eval 0.2569 @2040; KL text-duplex KL vs step 0 = 0 | w-1680 (HF only); w-1800; w-1920 / HF: asr-adapter-mot |
| [`motvox`](#motvox) | running | 31626 | 2026-10-27 18:44 PT .. running (step 14985 at 2026-10-27 19… | mot@14839, motasr2@1920 | 14839 -> 14985 so far (target 17366; +50 warm-up steps before 14840) | word CE(norm) eval_vpod/word_ce_norm 1.9029 @14964 (latest; 5.8745 at init step 14839; vpod_absent 1.9796); text-in eval_speech 1.9225 @14964 | dcp-14962 (first hourly save, node-local) / HF: ppm-motvox-14b, ppm-motvox-14b-state |
| [`small-qwen-notes`](#small-qwen-notes) | unknown (note) | unknown | unknown | unknown | n/a | unknown | unknown |


### intl14b

**INTL 14B = ppm-next (next-arch, interleaved Qwen3-14B)**  
group `intl14b` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/next-arch`

- **Jobs:** 22875, 22899, 22904, 22905, 22954, 22956, 23276, 23277, 23497, 28025, 28069
- **Dates:** 2026-10-17 03:55 PT .. 2026-10-23 06:15 PT (trained segments: 10-17 04:34 .. 10-18 13:44 PT, then 10-23 02:12 .. 06:15 PT)
- **Base model:** Qwen/Qwen3-14B (local /data/home/amartyavarma/models/Qwen3-14B)
- **Init:** fresh from base Qwen3-14B in job 22904 (444 tensors loaded, 170 random, seed 20261011); 22875 (0->36) and 22899 (0->69) were earlier fresh starts, logs moved to logs/aborted
- **Parents (resolved):** base:qwen3-14b
- **Produces:** intl14b@6242, intl14b@6982
- **Data:** PPM ppm-v1 data without pose: GA (speech, speech, text); speech = node-local s3-rows podcast rows plus D2 extra slices (slinky-4/7/15/18) from step 170; text = Gemma txtdup-full (CAP 7500) + corpus15m chat splits 2,3. Plan digest steps 0..17366 combined 7964360b90a9d30c. Evals: speech (salt ppm-v1 holdout), speech_s3 (103 Demeter rows), gemma, gemma_s3, chat, xdep
- **Recipe:** next-arch/code train_ilp.py mode interleave; next.env + next.sbatch; Qwen3-14B 40 layers; interleaved [user_text, asst_text, asst_speech] slots; RoPE/3; cb0 head on the backbone (no 4B tower); 14-layer light decoder, K 8 over 100% of frames; LR trunk 5e-5, spec 6e-5, proj 3e-5, audio 2e-4, tower 3e-4, decoder 4e-4; warmup 100 then constant; horizon 17400; world 128 (16 nodes x 8 GPUs)
- **Steps:** 0 -> 6982 (target 17400; data plan ends 17366)
- **Evals:** norm_word_ce = 1.9902 @6981 (eval_speech/word_ce_norm, last eval = best); 1.9989 @6241 (the eval nearest the w-6242 export that the FTs use); other = eval_speech_s3/word_ce_norm 2.0717 @6241; eval_gemma 0.8080 and eval_chat 0.6189 @6981
- **Checkpoints / exports:** w-3683; w-4818; w-5951; w-6241; w-6242; w-6982; dcp-6982 (node-local)
- **Local paths:** /data/home/amartyavarma/ppm/next-arch/ckpt-weka/ppm-next/w-6242; /data/home/amartyavarma/ppm/next-arch/ckpt-weka/ppm-next/w-6982; /data/home/amartyavarma/ppm/next-arch/ckpt-keep/w-3683; /scratch/amartyavarma/ppm-next/ckpt/ppm-next/dcp-6982 (16 hosts)
- **HF repos:** [InternalCan/ppm-next-14b](https://huggingface.co/InternalCan/ppm-next-14b)
- **Notes:** Resumed 10-23 02:12 PT after the wrong ppm14b resume. Paused again at 6981/6982 for the MoT launch (06:12-06:15 PT); never resumed since (last job 28069). w-6242 is the parent of aloe-next-6242, aloe-nextvox-6242, asr14b and the si20k-{tdx,tin,txt}-intl FTs (si20k-vox-intl was prepared on w-6242); those si20k rows are not in this part. Held-out sets: speech = ppm-v1 salt; the user's s3 set = eval_speech_s3.
- **Citations:**
  - `/data/home/amartyavarma/ppm/next-arch/next.env:1-3 -> next-arch = interleave-ppm with cb0 head, 14-layer decoder, RoPE/3, Qwen3-14B backbone, run ppm-next`
  - `/data/home/amartyavarma/ppm/next-arch/next.env:13,16,20-23 -> PPM_CB0_HEAD=1, ROPE_POS_DIV=3, qwen3-14b, 40 layers, DEC_LAYERS=14`
  - `/data/home/amartyavarma/ppm/next-arch/next.env:27 -> GA speech,speech,text`
  - `/data/home/amartyavarma/ppm/next-arch/next.env:42-56 -> LRs, warmup 100, horizon 17400, constant schedule`
  - `/data/home/amartyavarma/ppm/next-arch/next.env:63-69 -> decoder K 8, lr 4e-4`
  - `/data/home/amartyavarma/ppm/next-arch/next.env:75-95 -> data roots, CAP 7500, chat split 2,3`
  - `/data/home/amartyavarma/ppm/next-arch/next.env:105 -> PPM_MAX_STEPS=17400`
  - `/data/home/amartyavarma/ppm/next-arch/logs/aborted/ppm-next-22875.log:18,22,461 -> 10-17 03:55 PT fresh base init, stopped at 36`
  - `/data/home/amartyavarma/ppm/next-arch/logs/aborted/ppm-next-22899.log:22,493 -> fresh base init again, stopped at 69`
  - `/data/home/amartyavarma/ppm/next-arch/logs/ppm-next-22904.log:22 -> [init] backbone <- /data/home/amartyavarma/models/Qwen3-14B, 444 tensors, 170 random`
  - `/data/home/amartyavarma/ppm/next-arch/logs/ppm-next-22904.log:31 -> plan steps 0..17366 combined 7964360b90a9d30c`
  - `/data/home/amartyavarma/ppm/next-arch/logs/ppm-next-22904.log:482 -> stopped at 58`
  - `/data/home/amartyavarma/ppm/next-arch/logs/ppm-next-22905.log:31,758 -> 58 -> 384`
  - `/data/home/amartyavarma/ppm/next-arch/logs/ppm-next-22954.log:18 -> aborted in preflight (/scratch < 35 GB)`
  - `/data/home/amartyavarma/ppm/next-arch/logs/ppm-next-22956.log:31,3675 -> 384 -> 3565`
  - `/data/home/amartyavarma/ppm/next-arch/logs/ppm-next-23276.log:29 -> cfgdiff abort (export_every)`
  - `/data/home/amartyavarma/ppm/next-arch/logs/ppm-next-23277.log:20 -> D2 extra slices from step 170, holdout salt s3eval-v1`
  - `/data/home/amartyavarma/ppm/next-arch/logs/ppm-next-23277.log:31,3164 -> 3565 -> 6241`
  - `/data/home/amartyavarma/ppm/next-arch/logs/ppm-next-23497.log:41 -> EVAL step 6241 speech word_ce_norm 1.9989, speech_s3 2.0717`
  - `/data/home/amartyavarma/ppm/next-arch/logs/ppm-next-23497.log:431 -> STOPPED at step 6242 (10-18 13:44 PT)`
  - `/data/home/amartyavarma/ppm/next-arch/logs/ppm-next-28025.log:20 -> job 28025 2026-10-23 02:12:22 PT`
  - `/data/home/amartyavarma/ppm/next-arch/logs/ppm-next-28025.log:161,1310 -> resumed dcp-6242, STOPPED at 6981 10-23 06:12 PT`
  - `/data/home/amartyavarma/ppm/next-arch/logs/ppm-next-28069.log:171 -> EVAL step 6981 speech word_ce_norm 1.9902, gemma 0.8080, chat 0.6189`
  - `/data/home/amartyavarma/ppm/next-arch/logs/ppm-next-28069.log:561 -> STOPPED at step 6982 (10-23 06:15:44 PT)`
  - `/data/home/amartyavarma/ppm/next-arch/control/STOP_AFTER_SAVE.done-6981 -> 'pause ppm-next for the MoT (save-and-exit)' 10-23 06:12 PT`
  - `/data/home/amartyavarma/ppm/next-arch/JOB -> 28069 (last job)`
  - `/data/home/amartyavarma/ppm/next-arch/hf_backup/HF_BACKUP-w-6982.json:2-3,51 -> InternalCan/ppm-next-14b w-6982 PASS`
  - `/data/home/amartyavarma/ppm/next-arch/hf_backup/spec-w-*.json -> HF exports w-3683, w-4818, w-5951, w-6241, w-6242, w-6982`
  - `/data/home/amartyavarma/ppm/next-arch/hf_backup/hf_watch-28028.log:3 -> w-6982 HF VERIFY PASS 10-23 06:20 PT`
  - `memory:intl-14b-is-ppm-next.md:11,14 -> 10-23 01:10 PT the user asked to finish INTL; the agent wrongly resumed ppm14b; INTL jobs 22875..23497 stopped at 6242`
  - `/data/home/amartyavarma/ppm/si20k-tdx-intl/README.md:10, /data/home/amartyavarma/ppm/si20k-tin-intl/README.md:16, /data/home/amartyavarma/ppm/si20k-txt-intl/README.md:14 -> si20k FTs init from w-6242`

### aloe-next-6242

**aloe-next-6242 (Aloe FT of 14B next-arch w-6242)**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-next-6242`

- **Jobs:** 23522
- **Dates:** 2026-10-18 14:08 PT .. 2026-10-18 14:37 PT
- **Base model:** unknown
- **Init:** /data/home/amartyavarma/ppm/next-arch/ckpt-weka/ppm-next/w-6242 (consolidated bf16 DCP), fresh AdamW
- **Parents (resolved):** intl14b@6242
- **Produces:** aloe-next-6242@253
- **Data:** data_aloe_v2: stage2-handover stores m1n0..3, first chunks only, 8,067 train rows x 2 orientations = 16,134 micros; eval 162 rows x 2 = 324 micros; plan digest ba2d70729ea437ab, seed 20261011
- **Recipe:** aloe-v2init recipe (job 22383) unchanged, trainer code/train_aloe_next.py, aloe_next.env/.sbatch; 253 steps, warmup 8, linear decay to 0; LR trunk 3e-6 (clip 0.5), spec 6e-5, proj 3e-5, audio 6e-5, cb0_head (tower_s) 1.5e-5, decoder 1.5e-5; decoder K 1 on ~10% of B's frames; 1 row/rank/step on 64 ranks (8 nodes)
- **Steps:** 0 -> 253 (target 253)
- **Evals:** norm_word_ce = 1.8494 @253 (eval_speech/word_ce_norm; step 0 1.9022); other = eval_asstB 1.8652, eval_userB 1.8333 @253
- **Checkpoints / exports:** export-42; export-84; export-126; export-169; export-211; export-253; export-253-dcp
- **Local paths:** /data/home/amartyavarma/ppm/aloe-next-6242/exports/export-253
- **HF repos:** [InternalCan/aloe-next-6242](https://huggingface.co/InternalCan/aloe-next-6242)
- **Notes:** Partner run aloe-v2init-8869 (PPM w-8869) is not in this part. The consolidated DCP path named in the README (/data/home/amartyavarma/ppm/aloe-next-6242/export-253) is no longer on disk; only exports/export-253 is (ls 10-27). No deployment found.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-next-6242/README.md:3-5 -> architecture comparison at equal pretraining loss vs aloe-v2init-8869`
  - `/data/home/amartyavarma/ppm/aloe-next-6242/README.md:10-17 -> recipe, data, plan digest, 253 steps`
  - `/data/home/amartyavarma/ppm/aloe-next-6242/README.md:29 -> init w-6242, AdamW fresh`
  - `/data/home/amartyavarma/ppm/aloe-next-6242/README.md:35-40 -> per-module LRs`
  - `/data/home/amartyavarma/ppm/aloe-next-6242/aloe_next.env:9,21-26 -> QQA_STEPS=253 and LRs`
  - `/data/home/amartyavarma/ppm/aloe-next-6242/logs/aloe-next-6242-23522.log:1 -> job 23522 2026-10-18 14:08:44 PT on 8 nodes`
  - `/data/home/amartyavarma/ppm/aloe-next-6242/logs/aloe-next-6242-23522.log:215 -> EVAL step 0 1.9022`
  - `/data/home/amartyavarma/ppm/aloe-next-6242/logs/aloe-next-6242-23522.log:683 -> EVAL step 253 1.8494 (asstB 1.8652, userB 1.8333)`
  - `/data/home/amartyavarma/ppm/aloe-next-6242/logs/aloe-next-6242-23522.log:686 -> consolidated export-253 written`
  - `/data/home/amartyavarma/ppm/aloe-next-6242/exports/BACKUP_LOG.txt:1,27,28 -> private HF InternalCan/aloe-next-6242 export-253 + export-253-dcp VERIFY PASS`

### aloe-nextvox-6242

**aloe-nextvox-6242 (Aloe FT of 14B w-6242, Voxtral latents in the user-text slot)**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-nextvox-6242`

- **Jobs:** 24098, 24185, 24191, 24193
- **Dates:** 2026-10-18 16:30 PT (submit) / 16:54 PT (run 24193) .. 2026-10-18 17:26 PT
- **Base model:** unknown
- **Init:** next-arch w-6242 + Voxtral adapter from asr14b w-480 (sha b6ca8779f8d0b1bd; local copy adapter-asr14b-w480)
- **Parents (resolved):** intl14b@6242, asr14b@480
- **Produces:** aloe-nextvox-6242@253
- **Data:** data_aloe_v2vox, byte-identical to aloe-v2vox-raw j23081: same 8,067 x 2 train / 162 x 2 eval Aloe rows (s2h-m1 none bucket). Channel A has no text and no Mimi; the partner's WLT FLAC at raw gain goes through the frozen Voxtral Realtime encoder into the user_text slot
- **Recipe:** copy of PPM aloe-v2vox-raw j23081 on the 14B; trainer code/train_aloe_nextvox.py; 253 steps, warmup 8, linear decay to 0; LR trunk 3e-6, spec 6e-5, proj 3e-5, audio 6e-5, cb0_head 1.5e-5, decoder 1.5e-5, vadapter 5e-4, vgain 5e-4; 64 ranks on 8 nodes
- **Steps:** 0 -> 253 (target 253)
- **Evals:** norm_word_ce = 1.8558 @253 (eval_speech/word_ce_norm; step 0 2.1949; PPM reference j23081 step 0 2.0372); other = eval_asstB 1.8709, eval_userB 1.8402 @253
- **Checkpoints / exports:** export-42; export-84; export-126; export-169; export-211; export-253 (consolidated DCP); export-253-dcp (HF)
- **Local paths:** /data/home/amartyavarma/ppm/aloe-nextvox-6242/export-253; /data/home/amartyavarma/ppm/aloe-nextvox-6242/adapter-asr14b-w480
- **HF repos:** [InternalCan/aloe-nextvox-6242](https://huggingface.co/InternalCan/aloe-nextvox-6242)
- **Notes:** This is the 14B counterpart of PPM's Voxtral-on-A Aloe FT. Deployed in the ppm/aloe-nextvox-live demo (LOADED.json). Whether that demo is up now is unknown. Jobs 24098, 24185 and 24191 aborted or crashed before training (README:80-85).
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-nextvox-6242/README.md:3 -> user 10-18: Voxtral latents instead of the user text stream`
  - `/data/home/amartyavarma/ppm/aloe-nextvox-6242/README.md:7-9 -> reference aloe-v2vox-raw job 23081`
  - `/data/home/amartyavarma/ppm/aloe-nextvox-6242/README.md:15-27 -> data, channel A, schedule, adapter LR`
  - `/data/home/amartyavarma/ppm/aloe-nextvox-6242/README.md:37 -> adapter init asr14b w-480, sha b6ca8779f8d0b1bd`
  - `/data/home/amartyavarma/ppm/aloe-nextvox-6242/README.md:48-56 -> per-module LRs`
  - `/data/home/amartyavarma/ppm/aloe-nextvox-6242/README.md:80-91 -> 24098 preflight abort, 24185/24191 crashes, 24193 running; step-0 eval 2.1949 vs PPM 2.0372`
  - `/data/home/amartyavarma/ppm/aloe-nextvox-6242/LAUNCHED_JOB -> job ids + times`
  - `/data/home/amartyavarma/ppm/aloe-nextvox-6242/logs/aloe-nextvox-24193.log:823 -> EVAL step 253 1.8558 (asstB 1.8709, userB 1.8402)`
  - `/data/home/amartyavarma/ppm/aloe-nextvox-6242/README.md:104 -> 10-18 17:34:27 PT FINAL EXPORT export-253`
  - `/data/home/amartyavarma/ppm/aloe-nextvox-6242/exports/BACKUP_LOG.txt:27-28,56-57 -> HF InternalCan/aloe-nextvox-6242 export-N and export-253-dcp VERIFY PASS`
  - `/data/home/amartyavarma/ppm/aloe-nextvox-live/LOADED.json:2-4 -> demo serves /data/home/amartyavarma/ppm/aloe-nextvox-6242/export-253, label 'Aloe nextvox - 14B next-arch w-6242 - step 253'`

### asr14b

**14B ASR adapter (asr14b, w-480)**  
group `asr-adapter` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/asr-adapter-14b`

- **Jobs:** 23526, 24039
- **Dates:** 2026-10-18 14:13 PT .. 2026-10-18 16:18 PT
- **Base model:** unknown
- **Init:** frozen backbone = next-arch export w-6242; adapter init = j19174 w-3695 (sha256 45162ffc..., the same file as PPM's adapter init)
- **Parents (resolved):** intl14b@6242, qwen1ch-s4-asr2-ga1@3695
- **Produces:** asr14b@480
- **Data:** streamed MLS-eng train (stream_mls.py, seed 20261007, 2 node streams), 15006-position packs, PPM's pinned order files; pack lists byte-identical to PPM asrppm-nonorm. Eval: 384 MLS dev clips CE + 128-clip WER
- **Recipe:** train_asr_14b.py, asr14b.sbatch, stream.sbatch. Only the vadapter trains (6 tensors, 62.9M); backbone and Voxtral are frozen. Latents enter the user_text slot. AdamW LR 5e-4, warmup 50, WSD flat with linear decay to 0 over 12160..15200, max 15200. 16 packs/step (1 node 8 x GA 2, then 2 nodes 16 x GA 1 from step 361). Eval every 120 steps
- **Steps:** 0 -> 480 (max 15200; stopped by the user)
- **Evals:** norm_word_ce = 0.3158 @480 (transcript normalised word CE on MLS dev; final = best); kl = text-duplex KL vs step 0 = 0 (frozen backbone); other = WER 0.372 @480; PPM asrppm-nonorm final 0.2565 @1972
- **Checkpoints / exports:** w-120; w-240; w-360; w-480; r-480.pt (resume state)
- **Local paths:** /data/home/amartyavarma/ppm/asr-adapter-14b/ckpt-weka/asr14b/w-480; /data/home/amartyavarma/ppm/aloe-nextvox-6242/adapter-asr14b-w480; /scratch/amartyavarma/asr14b/ckpt/asr14b/r-480.pt (slinky-15, slinky-17)
- **HF repos:** [InternalCan/asr-adapter-14b](https://huggingface.co/InternalCan/asr-adapter-14b)
- **Notes:** This is the PPM asrppm-nonorm recipe ported to the 14B. Its adapter init (j19174 w-3695) has no canonical node id here, so it is left out of parents. w-480 seeds aloe-nextvox-6242 and si20k-txt-intl.
- **Citations:**
  - `/data/home/amartyavarma/ppm/asr-adapter-14b/README.md:3-6 -> stopped at 480 by the user 10-18 16:18 PT; w-480 sha b6ca8779f8d0b1bd; CE 0.3158, WER 0.372; HF InternalCan/asr-adapter-14b/asr14b/w-480; r-480.pt locations`
  - `/data/home/amartyavarma/ppm/asr-adapter-14b/README.md:13 -> base next-arch w-6242`
  - `/data/home/amartyavarma/ppm/asr-adapter-14b/README.md:28-51 -> mapping to PPM asrppm-nonorm (j21990 -> j22237, final 1972): data, batch, LR, schedule`
  - `/data/home/amartyavarma/ppm/asr-adapter-14b/README.md:66-73 -> job 23526 10-18 14:13 PT; adapter init sha 45162ffc; step-0 CE 4.6225`
  - `/data/home/amartyavarma/ppm/asr-adapter-14b/README.md:78-79 -> moved to 2 nodes as job 24039 at step 360`
  - `/data/home/amartyavarma/ppm/asr-adapter-14b/logs/asr14b-24039.log:247 -> EVAL step 480 CE 0.3158, WER 0.372, KL 0`
  - `/data/home/amartyavarma/ppm/asr-adapter-14b/logs/asr14b-24039.log:249 -> STOPPED at 480, export w-480`

### mot

**MoT 14B (Mixture-of-Transformers, 3 Qwen3-14B experts U / A_t / A_s)**  
group `mot` · status **paused** · dir `/data/home/amartyavarma/ppm/mot`

- **Jobs:** 28071, 28155, 29514, 30289
- **Dates:** 2026-10-23 06:15 PT .. 2026-10-27 11:27 PT
- **Base model:** Qwen/Qwen3-14B (x3 experts)
- **Init:** MOT_INIT=base: every expert of layer i = Qwen3-14B model.layers.i (three bit-identical copies), head.norm_t = model.norm, other tensors by INTL's step-0 rule and seed
- **Parents (resolved):** base:qwen3-14b
- **Produces:** mot@10830, mot@13814, mot@14810, mot@14839
- **Data:** identical to INTL 14B, step for step: plan steps 0..17366 combined 7964360b90a9d30c (= INTL job 22904); world 128; GA (speech, speech, text) = 256 speech + 128 text micros per step; no pose
- **Recipe:** mot/code (= next-arch/code + model_mot.py), mot.env (= next.env verbatim plus the MoT lines), mot.sbatch, launch.sh base. PPM_MOT=1: 3 experts x 13.21B per layer stack, one global causal attention, RoPE/3; 42.71B sharded + 0.36B replicated decoder. LRs and schedule = INTL (trunk 5e-5 constant after warmup 100, horizon 17400). Keep-1 hourly saves; exports every 360 min
- **Steps:** 0 -> 14839 (target 17400; data plan ends 17366)
- **Evals:** norm_word_ce = 1.9239 @14810 (eval_speech/word_ce_norm, last eval); best 1.9226 @14471; other = Matched steps vs INTL: 1.9809 @6254 (INTL 1.9989 @6241); 1.9733 @6930 (INTL 1.9902 @6981). @14810: gemma 0.7636, chat 0.5959, speech_s3 1.9992
- **Checkpoints / exports:** w-1493; w-2486; w-4267; w-5264; w-6828; w-7824; w-8824; w-9828; w-10830; w-11824; w-12818; w-13814; w-14810; dcp-14839 (pause state)
- **Local paths:** /data/home/amartyavarma/ppm/mot/ckpt-weka/ppm-mot/w-14810; /data/home/amartyavarma/ppm/mot-asr/base/w-10830; /scratch/amartyavarma/ppm-mot/ckpt/ppm-mot/dcp-14839 (16 hosts)
- **HF repos:** [InternalCan/ppm-mot-14b](https://huggingface.co/InternalCan/ppm-mot-14b), [InternalCan/ppm-mot-14b-state](https://huggingface.co/InternalCan/ppm-mot-14b-state)
- **Deployed:** MoT 14B live demo (ppm/mot/live, motlive) (paused)
- **Notes:** Paused 10-27 for the pose5 / motasr2 switch; the plain-resume kit is guard/stop/mot_resume_pause.sh. It actually continues as motvox from dcp-14839. Deployment: the ppm/mot/live real-time demo (paused since 10-27 11:28 PT, last weights w-13814). mot-bench and scan14 trained nothing (a speed benchmark and a data key scan). Debug dirs 28061 and 28063 have no training log.
- **Citations:**
  - `/data/home/amartyavarma/ppm/mot/STATUS.md:5-9 -> changes vs INTL; params 42.71B + 0.36B`
  - `/data/home/amartyavarma/ppm/mot/STATUS.md:53-54 -> data schedule identical to INTL; launch check digest 7964360b90a9d30c`
  - `/data/home/amartyavarma/ppm/mot/STATUS.md:122 -> state backup repo InternalCan/ppm-mot-14b-state`
  - `/data/home/amartyavarma/ppm/mot/mot.env:1-11 -> MoT = next.env verbatim except MoT lines`
  - `/data/home/amartyavarma/ppm/mot/mot.env:49,64,77,78,127,132,138 -> GA, LR trunk 5e-5, horizon 17400, constant, max 17400, export 360 min, keep 1`
  - `/data/home/amartyavarma/ppm/mot/logs/ppm-mot-28071.log:20,161,164 -> job 28071 10-23 06:15:50 PT, [mot-init] base, plan combined 7964360b90a9d30c`
  - `/data/home/amartyavarma/ppm/mot/logs/ppm-mot-28071.log:171 -> step-0 eval speech 6.0038`
  - `/data/home/amartyavarma/ppm/mot/logs/ppm-mot-28071.log:1067 -> STOPPED at 497`
  - `/data/home/amartyavarma/ppm/mot/logs/ppm-mot-28155.log:162,3399 -> 497 -> 3270`
  - `/data/home/amartyavarma/ppm/mot/logs/ppm-mot-29514.log:162,3181 -> 3270 -> 5829`
  - `/data/home/amartyavarma/ppm/mot/logs/ppm-mot-30289.log:162,9793 -> 5829 -> 14839, STOPPED 10-27 11:27:31 PT`
  - `/data/home/amartyavarma/ppm/mot/logs/ppm-mot-30289.log:9759 -> EVAL step 14810 speech 1.9239, gemma 0.7636, chat 0.5959, speech_s3 1.9992`
  - `/data/home/amartyavarma/ppm/mot/logs/ppm-mot-30289.log:9412 -> EVAL step 14471 speech 1.9226 (min)`
  - `/data/home/amartyavarma/ppm/mot/logs/ppm-mot-30289.log:990,1683 -> EVAL step 6254 1.9809; step 6930 1.9733`
  - `/data/home/amartyavarma/ppm/mot/PAUSE_STATE.env:2-3,22,28-33 -> paused at 14839 (job 30289), stopped 10-27 11:27:43 PT, dcp-14839 kept on 16 hosts + HF verified`
  - `/data/home/amartyavarma/ppm/mot/hf_backup/HF_BACKUP-w-*.json:2 -> InternalCan/ppm-mot-14b exports w-1493 .. w-14810, all PASS`
  - `/data/home/amartyavarma/ppm/mot/live/STATUS.md:87,101,114,115 -> live demo served w-10830, then w-11824 -> w-12818 -> w-13814 (hot swaps)`
  - `/data/home/amartyavarma/ppm/mot/live/STATUS.md:117 -> live demo stopped 10-27 11:28 PT, last weights w-13814`
  - `memory:mot-14b-plan.md:24-26 -> init from base x3 (user 10-23 05:50 PT); same data/LR/evals as INTL`
  - `memory:pose5-pause-mot-plan.md:38-43 -> after pose run MoT continues as audio-in motvox from dcp-14839, not a plain resume`
  - `/data/home/amartyavarma/ppm/mot-bench/README.md:1-3 -> mot-bench = 32B MoT vs dense speed benchmark (10-22), not a training run`

### motasr

**motasr (Voxtral ASR adapter on frozen MoT w-10830)**  
group `asr-adapter` · status **paused** · dir `/data/home/amartyavarma/ppm/mot-asr`

- **Jobs:** 30741
- **Dates:** 2026-10-26 12:55 PT .. 2026-10-26 16:34 PT
- **Base model:** unknown
- **Init:** frozen MoT export w-10830 (staged at /data/home/amartyavarma/ppm/mot-asr/base/w-10830); adapter init j19174 w-3695 (sha 45162ffc, = PPM / asr14b)
- **Parents (resolved):** mot@10830, qwen1ch-s4-asr2-ga1@3695
- **Produces:** motasr@1200, motasr@1220
- **Data:** the same MLS-eng streams and packs as asr14b (pack lists 32/32 byte-identical to the 14B run), stream root ppm/mot-asr/stream
- **Recipe:** asr14b recipe (= PPM asrppm-nonorm) copied exactly: train_asr_mot.py, motasr.sbatch, adapter-only, LR 5e-4, world 16 x GA 1 on slinky-10 + slinky-18
- **Steps:** 0 -> 1220 (max 15200)
- **Evals:** norm_word_ce = 0.2657 @1200 (best, transcript normalised word CE; WER 0.193); kl = text-duplex KL vs step 0 = 0
- **Checkpoints / exports:** w-960; w-1080; w-1200; w-1220; r-1220.pt
- **Local paths:** /data/home/amartyavarma/ppm/mot-asr/ckpt-weka/motasr/w-1200; /data/home/amartyavarma/ppm/mot-asr/ckpt-weka/motasr/w-1220; /data/home/amartyavarma/ppm/mot-asr/resume-stage/r-1220.pt
- **HF repos:** [InternalCan/asr-adapter-mot](https://huggingface.co/InternalCan/asr-adapter-mot)
- **Notes:** Paused by the user ('pause it and use the spare compute') and never resumed. Superseded by motasr2. Resume command: STATUS.md:67-73.
- **Citations:**
  - `/data/home/amartyavarma/ppm/mot-asr/STATUS.md:3-5 -> user order; recipe = asr14b; base MoT w-10830; slinky-10 + 18, world 16 x GA 1`
  - `/data/home/amartyavarma/ppm/mot-asr/STATUS.md:62-65 -> paused by the user 10-26 16:35 PT at step 1220; exports w-960/1080/1200/1220; best 0.2657`
  - `/data/home/amartyavarma/ppm/mot-asr/STATUS.md:76 -> job 30741 started 12:55 PT 10-26`
  - `/data/home/amartyavarma/ppm/mot-asr/STATUS.md:78-81 -> in-job proofs; adapter init sha 45162ffc; step-0 CE 7.0413`
  - `/data/home/amartyavarma/ppm/mot-asr/STATUS.md:94 -> HF watcher -> InternalCan/asr-adapter-mot`
  - `/data/home/amartyavarma/ppm/mot-asr/STATUS.md:99 -> base staged at ppm/mot-asr/base/w-10830`
  - `/data/home/amartyavarma/ppm/mot-asr/logs/motasr-30741.log:1351 -> EVAL step 1200 CE 0.2657, WER 0.193`
  - `/data/home/amartyavarma/ppm/mot-asr/logs/motasr-30741.log:1375 -> STOPPED at 1220, best 0.2657 @1200`
  - `memory:motasr-run.md:11,17 -> launch 10-26 12:55 PT job 30741; paused at 1220, best 0.2657 @1200`

### motasr2

**motasr2 (fresh Voxtral ASR adapter on frozen MoT w-14810)**  
group `asr-adapter` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/mot-asr`

- **Jobs:** 31090
- **Dates:** 2026-10-27 11:28 PT .. 2026-10-27 17:34 PT
- **Base model:** unknown
- **Init:** frozen MoT export w-14810 (/data/home/amartyavarma/ppm/mot/ckpt-weka/ppm-mot/w-14810, 29 steps before the dcp-14839 pause state); FRESH adapter (j19174 w-3695, sha 45162ffc); data from step 0 (stream2)
- **Parents (resolved):** mot@14810, qwen1ch-s4-asr2-ga1@3695
- **Produces:** motasr2@1920
- **Data:** the same MLS-eng streams as motasr / asr14b, fresh stream root ppm/mot-asr/stream2
- **Recipe:** motasr recipe exactly (= asr14b = PPM asrppm-nonorm): train_asr_mot.py, motasr.sbatch, adapter-only LR 5e-4, world 16 x GA 1 on slinky-10 + slinky-18, ~9.0 s/step
- **Steps:** 0 -> 2094 (max 15200; stopped by the user)
- **Evals:** norm_word_ce = 0.2527 @1920 (best; transcript normalised word CE, WER 0.554); last eval 0.2569 @2040; kl = text-duplex KL vs step 0 = 0; other = beats PPM asrppm-nonorm final 0.2565 @1972
- **Checkpoints / exports:** w-1680 (HF only); w-1800; w-1920; w-2040; w-2094; r-2094.pt
- **Local paths:** /data/home/amartyavarma/ppm/mot-asr/ckpt-weka/motasr2/w-1920; /data/home/amartyavarma/ppm/mot-asr/ckpt-weka/motasr2/w-2094; /data/home/amartyavarma/ppm/mot-asr/resume-stage/r-2094-motasr2.pt
- **HF repos:** [InternalCan/asr-adapter-mot](https://huggingface.co/InternalCan/asr-adapter-mot)
- **Notes:** Stopped by the user ('you can stop the asr adapter'). w-1920 is the adapter init of motvox. The trainer's keep-newest-3+best rule pruned w-1680 from /data; it is still on HF.
- **Citations:**
  - `/data/home/amartyavarma/ppm/mot-asr/STATUS.md:7-13 -> stopped with save at 2094, exit 17:34:30 PT 10-27; exports w-1800/1920/2040/2094; best w-1920 0.2527; w-1680 HF only at InternalCan/asr-adapter-mot/motasr2/w-1680`
  - `/data/home/amartyavarma/ppm/mot-asr/STATUS.md:18 -> resume needs base w-14810 at ppm/mot/ckpt-weka/ppm-mot/w-14810`
  - `/data/home/amartyavarma/ppm/mot-asr/STATUS.md:24 -> FRESH adapter, data from step 0, stream2`
  - `/data/home/amartyavarma/ppm/mot-asr/STATUS.md:34-35 -> launch proofs; step-0 CE 7.2876; ~9.0 s/step`
  - `/data/home/amartyavarma/ppm/mot-asr/STATUS.md:54-55 -> step 1920 CE 0.2527 (best); step 2040 0.2569`
  - `/data/home/amartyavarma/ppm/mot-asr/STATUS.md:56-58 -> job 31090 running from 11:28:42 PT; base w-14810, 29 steps before the pause`
  - `/data/home/amartyavarma/ppm/mot-asr/logs/motasr2-31090.log:2270 -> STOPPED at 2094; best 0.2527 (WER 0.554) @1920`
  - `memory:motasr-run.md:19,21 -> motasr2 base w-14810, job 31090; stopped 10-27 17:34 PT; best w-1920 0.2527`
  - `/data/home/amartyavarma/ppm/motvox/motvox.env:310-311 -> w-1920 is motvox's adapter init`

### motvox

**motvox (audio-in MoT 14B: MoT + motasr2 adapter, podcast audio-in rows)**  
group `motvox` · status **running** · dir `/data/home/amartyavarma/ppm/motvox`

- **Jobs:** 31626
- **Dates:** 2026-10-27 18:44 PT .. running (step 14985 at 2026-10-27 19:57 PT)
- **Base model:** unknown
- **Init:** MoT dcp-14839 (node-local /scratch/amartyavarma/ppm-mot/ckpt/ppm-mot/dcp-14839, read-only; weights + 3 optimizers + scheduler restored) + motasr2 adapter w-1920 (sha f459e014...), trained jointly and added unscaled into the U slot (fresh AdamW for the adapter)
- **Parents (resolved):** mot@14839, motasr2@1920
- **Produces:** none recorded
- **Data:** Podcast stream = audio-in (vpod) rows only. Stems come from HF InternalCan/podcast-raw-audio-data-mass (subsets 700k-70kh-v1 + intelligent-30kh-v1; 1,321,858 FLAC, ~100,012 h). Usable: 532,484 rows -> 1,064,968 entries, 12.0% of MoT's 4.45M rows. Text duplex is exactly MoT's (TEXT plan 98f53c4e618b9c32; VPOD plan 31733cfc7eee8572). Training user-audio gain targets LUFS ~ truncated N(-25, 5) on [-40, -15]
- **Recipe:** motvox/code (MoT trainer + MOTVOX_MODE=1), motvox.env, motvox.sbatch, launch.sh; rest of the MoT recipe unchanged (LR trunk 5e-5 constant, world 128 on 16 hosts, cycle vpod,vpod,text). Adapter: joint, LR 5e-4 constant, clip 1.0, 16-step LR warmup, plus 50 adapter-only warm-up steps before logged step 14840. Voxtral frozen; prompt rule 'mot'; end step 17366. ~22.7 s/step, peak 63.7 GiB
- **Steps:** 14839 -> 14985 so far (target 17366; +50 warm-up steps before 14840)
- **Evals:** norm_word_ce = eval_vpod/word_ce_norm 1.9029 @14964 (latest; 5.8745 at init step 14839; vpod_absent 1.9796); text-in eval_speech 1.9225 @14964; other = eval_gemma 0.7627, eval_chat 0.5958 @14964; fixed-gain vpod -40/-25/-15 LUFS = 1.9027/1.9027/1.9029
- **Checkpoints / exports:** dcp-14962 (first hourly save, node-local)
- **Local paths:** /scratch/amartyavarma/ppm-motvox/ckpt (16 hosts); /data/home/amartyavarma/ppm/motvox/ckpt-weka (export root; no export yet)
- **HF repos:** [InternalCan/ppm-motvox-14b](https://huggingface.co/InternalCan/ppm-motvox-14b) *(not in model listing: dataset/bucket or deleted)*, [InternalCan/ppm-motvox-14b-state](https://huggingface.co/InternalCan/ppm-motvox-14b-state)
- **Notes:** Running on the 16 prod hosts. It borrowed them while the 8,1,1 pose run 31101 is paused at dcp-18098; after motvox, the hosts go back to pose5. It is ~1 h into the run, so it has no export yet. Stems licence is recorded as 'unknown'; main said to proceed (STATUS.md:63,111).
- **Citations:**
  - `/data/home/amartyavarma/ppm/motvox/JOB -> 31626`
  - `/data/home/amartyavarma/ppm/motvox/STATUS.md:3 -> user 10-27 17:37 PT order`
  - `/data/home/amartyavarma/ppm/motvox/STATUS.md:16-17 -> adapter = motasr2 best w-1920; init MoT dcp-14839; gain-aug rule`
  - `/data/home/amartyavarma/ppm/motvox/STATUS.md:26-28 -> vpod rows only, text duplex = MoT, adapter unscaled into U, own run name ppm-motvox`
  - `/data/home/amartyavarma/ppm/motvox/STATUS.md:63-64 -> stems repo, 1,321,858 FLAC, ~100,012 h; usable 532,484 rows, 12.0% of MoT rows`
  - `/data/home/amartyavarma/ppm/motvox/STATUS.md:108 -> SUBMITTED motvox-train 31626 10-27 18:44 PT (16 hosts)`
  - `/data/home/amartyavarma/ppm/motvox/STATUS.md:109 -> launched from the pose pause dcp-18098 (swap mode A)`
  - `/data/home/amartyavarma/ppm/motvox/STATUS.md:119 -> MoT dcp-14839 restored; plan digests as expected; step-0 eval vpod 5.87`
  - `/data/home/amartyavarma/ppm/motvox/STATUS.md:125 -> warm-up done; 22.2-22.7 s/step; ETA 17,366 ~10:50 PT 10-28`
  - `/data/home/amartyavarma/ppm/motvox/STATUS.md:130-131 -> gate PASS at eval 14964; first hourly save dcp-14962`
  - `/data/home/amartyavarma/ppm/motvox/EVALS_motvox.md:5-7 -> evals at 14839, 14884, 14964`
  - `/data/home/amartyavarma/ppm/motvox/motvox.env:291,307-322,329,351,355-361 -> MOTVOX knobs (init, end 17366, adapter w-1920, joint, LR, warm-up, Voxtral frozen, prompt rule mot, gain aug)`
  - `/data/home/amartyavarma/ppm/motvox/motvox.env:67,137,141,155 -> LR trunk 5e-5, export root, keep 1, save root`
  - `/data/home/amartyavarma/ppm/motvox/logs/motvox-train-31626.log:2 -> adapter init w-1920 sha check`
  - `/data/home/amartyavarma/ppm/motvox/logs/motvox-train-31626.log:163-164 -> resume from MoT dcp-14839, digests == saved`
  - `/data/home/amartyavarma/ppm/motvox/logs/motvox-train-31626.log:1123 -> EVAL step 14964: speech 1.9225, gemma 0.7627, chat 0.5958, vpod 1.9029, vpod_absent 1.9796`
  - `/data/home/amartyavarma/ppm/motvox/logs/motvox-train-31626.log:1165 -> step 14985/17366 at 10-27 19:57:58 PT, 22.7 s, peak 63.7 GiB`
  - `/data/home/amartyavarma/ppm/motvox/RESUMABILITY.md:14 -> HF repos InternalCan/ppm-motvox-14b[-state] not yet created at writing`
  - `memory:pose5-pause-mot-plan.md:38-43 -> motvox plan: init dcp-14839 + w-1920, ~100k h audio-in, LUFS aug`

### small-qwen-notes

**Qwen3-4B / Qwen3-0.6B usage (notes only)**  
group `other` · status **unknown** · dir `/data/home/amartyavarma/ppm`

- **Jobs:** unknown
- **Dates:** unknown
- **Base model:** unknown
- **Init:** unknown
- **Parents (resolved):** unknown
- **Produces:** none recorded
- **Data:** n/a
- **Recipe:** n/a
- **Steps:** n/a
- **Evals:** none recorded
- **Checkpoints / exports:** unknown
- **Local paths:** /data/home/amartyavarma/models/Qwen3-0.6B
- **HF repos:** none named in run files
- **Notes:** No training run starts from pretrained Qwen3-4B or Qwen3-0.6B weights. In PPM 32B (and ppm14b, ppm-mistral128b) '4B' and '0.6B' are the architectures of randomly initialised speech/pose towers and depth decoders inside one model. INTL 14B / MoT drop the 4B tower and halve the decoder to 14 layers. The only use of real Qwen3-0.6B weights found is as a frozen feature extractor for the gesturemetric probe. So base:qwen3-4b and base:qwen3-0.6b should have no edges. The grep covered *.env/*.md/*.json at -maxdepth 3 under ppm.
- **Citations:**
  - `/data/home/amartyavarma/ppm/DESIGN.md:11-12 -> PPM speech/pose pairs: Qwen3-4B tower (random init, D14) + 0.6B decoder (random init, D15)`
  - `/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer/README.md:26 -> PPM 32B = Qwen3-32B + two Qwen3-4B-architecture towers + two Qwen3-0.6B-architecture depth decoders`
  - `/data/home/amartyavarma/ppm/megatron-port/README.md:17-18 -> towers Qwen3-4B arch 36 layers; decoders Qwen3-0.6B light arch 28 layers`
  - `/data/home/amartyavarma/ppm/ppm-mistral128b/README.md:40 -> the Mistral-128B variant keeps PPM's Qwen3-4B-arch towers`
  - `/data/home/amartyavarma/ppm/next-arch/next.env:1,13,23 -> INTL / MoT: no 4B tower (cb0 head), 14-layer light decoder`
  - `/data/home/amartyavarma/ppm/ppm14b/ppm14b.env:4 -> ppm14b uses PPM's 4B speech tower + 28-layer 0.6B decoder`
  - `/data/home/amartyavarma/ppm/agents1026/gesturemetric/RESULT.md:66,113 -> gesture metric uses frozen local Qwen3-0.6B layer-14 word features in a small 0.87M / ~8-10M-param model`
  - `memory:pose-podcast-model-plan.md:13,19 -> plan: 32B backbone + Qwen3-4B + 0.6B, both random init`
  - `ls /data/home/amartyavarma/models -> Qwen3-0.6B, Qwen3-8B, Qwen3-14B, Qwen3-32B present; no Qwen3-4B directory`

## 32B speech FTs, PPMVOX, 32B ASR adapters, si20k / xfer

| run | status | jobs | dates | init (parent node@step) | steps | headline evals | outputs |
|---|---|---|---|---|---|---|---|
| [`ppmvox-v1`](#ppmvox-v1) | stopped-with-save | 28694, 28723, 28729, 28734 | 2026-10-24 03:07 .. 2026-10-24 04:34 PT (28694/28723/28729… | ppm32b@16861, asrppm-w16861@16861-w840, base:voxtral-mini-4b-realtime-2602 | 16861 -> 17035 (vox 174 of planned 500) | word CE(norm) eval_vpod/word_ce_norm 1.9498 @16861 -> 1.8662 @16994 (last eval; v1 eval set, not comparable with v3+ eval set) | w-16911; w-17018; w-17035 (FINAL bf16) |
| [`ppmvox-v2`](#ppmvox-v2) | abandoned | unknown | 2026-10-24 ~04:35 PT (env written) | ppm32b@16861, asrppm-w16861@16861 | none run | unknown | unknown |
| [`ppmvox-v3`](#ppmvox-v3) | stopped-with-save | 28998, 29005 | 2026-10-24 05:16 .. 2026-10-24 07:19 PT (28998 aborted at p… | ppm32b@16861, asrppm-w16861@16861, base:voxtral-mini-4b-realtime-2602 | 16861 -> 17127 (vox 266 of 500; stopped on user order 'plateaued') | word CE(norm) eval_vpod/word_ce_norm 2.0231 @16861 -> 1.9217 @17071 (last logged eval) | w-16911; w-17032; w-17127 (FINAL bf16) / HF: ppmvox-v3-16861, ppm-voxtral-w17127 |
| [`ppmvox-v3x`](#ppmvox-v3x) | abandoned | 29218 | 2026-10-24 (pending, scancelled ~07:18 PT) | ppmvox-v3@17127 | none run | unknown | unknown |
| [`ppmvox-v5`](#ppmvox-v5) | stopped-with-save | 29813, 29830, 29853, 29859, 29863 (+1) | 2026-10-25 00:25 .. 2026-10-25 03:11 PT | ppm811-append3@17850, asrppm-811-17850@17850, base:voxtral-mini-4b-realtime-2602 | 17850 -> 17900 (warm-up, 29813) -> 18100 (vox 250 of 500, SAVE_NOW th… | word CE(norm) eval_vpod/word_ce_norm 2.0342 @17850 -> 1.9744 @17900 -> 1.9177 @18084 (last eval) | ckpt-wu step 17900; dcp-18100 (full); w-18100 (bf16) / HF: ppmvox-v5-17850, ppmvox-v5-18100-merged-state |
| [`ppmvox-v5b`](#ppmvox-v5b) | stopped-with-save | 30104 | 2026-10-25 03:32 .. 2026-10-25 05:10 PT | ppmvox-v5@18100 | 18100 -> 18163 (vox 313); STOP_AFTER_SAVE on user go to restart MoT | word CE(norm) eval_vpod/word_ce_norm 1.9183 @18100 -> best 1.9155 @18142 -> 1.9167 @18163 (final) | dcp-18163 (full, incl voxenc AdamW); bf16 export w-18163 NOT written / HF: ppmvox-v5b-18163-state, ppmvox-v5b |
| [`asrppm-rmsnorm-11488`](#asrppm-rmsnorm-11488) | abandoned | 21908, 21909, 21917, 21972 | 2026-10-16 01:54 .. 2026-10-16 03:20 PT | ppm32b@11488, base:voxtral-mini-4b-realtime-2602, qwen1ch-s4-asr2-ga1@3695 | 0 -> 305 (21917) -> 496 (21972) | word CE(norm) transcript normalised word CE best 3.7845 @480 (WER 1.002): did not learn | w-305; w-360; w-480 |
| [`asrppm-nonorm`](#asrppm-nonorm) | finished | 21986, 21990, 22237 | 2026-10-16 03:20 .. 2026-10-16 07:49 PT | ppm32b@11488, base:voxtral-mini-4b-realtime-2602, qwen1ch-s4-asr2-ga1@3695 | 0 -> 1972 (STOP_AFTER_SAVE; user declared finished at 1972) | word CE(norm) transcript normalised word CE best 0.2549 @1680 (WER 0.811); 0.2565 at the 1972 resume eval | w-1680; w-1800; w-1920 |
| [`asrppm-w15973`](#asrppm-w15973) | stopped-with-save | 26064 | 2026-10-20 15:18 .. 2026-10-20 18:32 PT | ppm32b@15973, base:voxtral-mini-4b-realtime-2602 | 0 -> 1441 (STOP_AFTER_SAVE) | word CE(norm) transcript normalised word CE best 0.2629 @1320 (WER 0.823) | w-1200; w-1320; w-1440 / HF: asr-adapter-ppm |
| [`asrppm-w16861`](#asrppm-w16861) | stopped-with-save | 28311, 28462 | 2026-10-24 00:14 .. 2026-10-24 04:02 PT | ppm32b@16861, base:voxtral-mini-4b-realtime-2602, qwen1ch-s4-asr2-ga1@3695 | 0 -> 340 (28311) -> 1346 (28462, STOP_AFTER_SAVE; target 1500 never r… | word CE(norm) transcript normalised word CE best 0.2662 @1200 (WER 0.979); w-480: 0.2899 (WER 0.572); w-840: 0.2871 | w-480; w-840; w-1080 / HF: asrppm-w16861 |
| [`asrppm-811-17850`](#asrppm-811-17850) | stopped-with-save | 29671, 29676 | 2026-10-24 22:09 .. 2026-10-25 00:26 PT | ppm811-append3@17850, base:voxtral-mini-4b-realtime-2602, qwen1ch-s4-asr2-ga1@3695 | 0 -> 835 (STOP_AFTER_SAVE; target 1500) | word CE(norm) transcript normalised word CE best 0.2878 @720 (WER 0.743) | w-480; w-600; w-720 / HF: asrppm-811-17850 |
| [`aloe-ft-21980`](#aloe-ft-21980) | finished | 21980 | 2026-10-16 03:15 .. 2026-10-16 04:12 PT | ppm32b@11488 | 0 -> 366 | word CE(norm) eval_speech/word_ce_norm 1.8303 @0 -> 1.7944 @366 | export-244; export-305; export-366 |
| [`aloe-vox-ft`](#aloe-vox-ft) | stopped-with-save | 22275, 22284 | 2026-10-16 15:41 .. 2026-10-16 16:23 PT | ppm32b@12465, asrppm-nonorm@11488 | 0 -> 72 (STOP_AFTER_SAVE; planned 430) | word CE(norm) 1.9360 @0 -> 1.8697 @72 | export-72 |
| [`aloe-voxall-ft`](#aloe-voxall-ft) | stopped-with-save | 22303 | 2026-10-16 16:31 .. 2026-10-16 16:36 PT | ppm32b@12465, asrppm-nonorm@11488 | 0 -> 11 (STOP_AFTER_SAVE; planned 498) | word CE(norm) 1.9392 @0 (only eval) | unknown |
| [`aloe-m1vox-ft`](#aloe-m1vox-ft) | finished | 22315, 22335 | 2026-10-16 16:47 .. 2026-10-16 18:30 PT | ppm32b@11488, asrppm-nonorm@11488 | 22315: 0 -> 111 (stopped, speed fix); 22335: fresh 0 -> 366 | word CE(norm) 1.9687 @0 -> 1.7934 @366 | export-244; export-305; export-366 |
| [`aloe-v2vox-ft`](#aloe-v2vox-ft) | finished | 22420 | 2026-10-16 20:12 .. 2026-10-16 20:42 PT | ppm32b@11488, asrppm-nonorm@11488 | 0 -> 253 | word CE(norm) 2.0371 @0 -> 1.8246 @253 | export-253 |
| [`aloe-v2vox-raw`](#aloe-v2vox-raw) | finished | 23060, 23079, 23081 | 2026-10-17 ~16:5x .. 2026-10-17 18:31 PT | ppm32b@11488, asrppm-nonorm@11488 | 0 -> 253 (23060 stopped at 6, 23079 restart aborted, 23081 full run f… | word CE(norm) 2.0372 @0 -> 1.8248 @253 | export-253 |
| [`aloe-v2vox-raw-w14503`](#aloe-v2vox-raw-w14503) | finished | 25962, 25963 | 2026-10-20 ~00:5x .. 2026-10-20 02:31 PT | ppm32b@14503, asrppm-nonorm@11488 | 0 -> 253 | word CE(norm) 2.0177 @0 -> 1.8138 @253 | export-42; export-253 / HF: ppm-aloe-v2vox-raw-w14503 |
| [`aloe-v2vox-asr-w15973`](#aloe-v2vox-asr-w15973) | finished | 26221, 26232 | 2026-10-20 ~19:0x .. 2026-10-20 19:32 PT | ppm32b@15973, asrppm-w15973@15973 | 0 -> 253 | word CE(norm) 2.0125 @0 -> 1.8075 @253 | export-42; export-253 / HF: ppm-aloe-v2vox-asr-w15973 |
| [`aloe-v2vox-asr-w15973-textB`](#aloe-v2vox-asr-w15973-textb) | finished | 26419, 26422 | 2026-10-20 23:25 .. 2026-10-21 01:07 PT | ppm32b@15973, asrppm-w15973@15973 | 0 -> 253 (26419 exit 1 at step-0 pack check) | word CE(norm) 2.2137 @0 -> 1.9100 @253 | export-42; export-253 / HF: ppm-aloe-v2vox-asr-w15973-textB |
| [`aloe-samsep-vox17127`](#aloe-samsep-vox17127) | finished | 29345 | 2026-10-24 07:24 .. 2026-10-24 07:53 PT | ppmvox-v3@17127 | 0 -> 249 | word CE(norm) 1.8484 @0 -> 1.8023 @249 | export-vox17127-249; dcp-249 (node-local) / HF: ppm-aloe-samsep-vox17127, ppm-aloe-samsep-vox17127-final, aloe-ft-vox17127-nopersona |
| [`aloespk-brianna-A`](#aloespk-brianna-a) | stopped-with-save | 29536, 29540, 29543, 29545 | 2026-10-24 ~08:50 .. 2026-10-24 09:23 PT | aloe-samsep-vox17127@249 | 0 -> 24 of 40 (stopped; candidate export-10) | word CE(norm) standard Aloe eval_speech 1.8023 @0 -> 1.8138 @10 -> 1.8323 @24 (speaker held-out eval not logged in this run) | export-10 (LoRA delta); export-20 / HF: ppm-aloespk-lora-fFP3835k-A, ppm-aloespk-lora-fFP3835k-A-final |
| [`aloespk-brianna-B`](#aloespk-brianna-b) | stopped-with-save | 29549 | 2026-10-24 ~09:27 .. 2026-10-24 09:45 PT | aloe-samsep-vox17127@249 | 0 -> 56 (past best; stopped) | word CE(norm) speaker held-out eval_spk_speech best 1.7873 @24 -> 1.8809 @56; standard 1.8023 @0, 1.8119 @20, 1.8302 @40 | export-40 / HF: ppm-aloespk-lora-fFP3835k-B, ppm-aloespk-lora-fFP3835k-B-final |
| [`aloespk-brianna-B2`](#aloespk-brianna-b2) | finished | 29558 | 2026-10-24 ~09:49 .. 2026-10-24 10:00 PT | aloe-samsep-vox17127@249 | 0 -> 24 | word CE(norm) standard eval_speech 1.8094 @16 (cb0 best), 1.8151 @24 (light best) | export-16; export-24 / HF: ppm-aloespk-lora-fFP3835k-B2, ppm-aloespk-lora-fFP3835k-B2-final |
| [`aloespk-rosie`](#aloespk-rosie) | finished | 29575, 29577 | 2026-10-24 18:29 .. 2026-10-24 18:45 PT | aloe-samsep-vox17127@249 | 0 -> 48 | word CE(norm) speaker held-out eval_spk_speech 1.6230 @6 -> best 1.5848 @42; standard eval_speech 1.8023 @0 -> 1.8292 @48 | export-24; export-48 (LoRA deltas) / HF: ppm-aloespk-lora-xyvR1rjH, ppm-aloespk-lora-xyvR1rjH-final |
| [`qqa-ft`](#qqa-ft) | finished | 21817, 21823, 21832 | 2026-10-15 23:20 .. 2026-10-16 ~01:xx PT | ppm32b@11263 | 0 -> 223 (21817 crashed at 82; 21823 full; 21832 rerun) | word CE(norm) B normalised word CE 1.9862 @0 -> 1.4268 @223 (21823); 1.4318 @223 (21832) | export-223; scale/export-14,28,59,117,223 / HF: ppm-qqa-w223 |
| [`ttconvo-ft`](#ttconvo-ft) | stopped-with-save | 26510 | 2026-10-21 04:17 .. 2026-10-21 05:02 PT | ppm811-epoch2@17012 | 0 -> 285 of 296 (STOP_AFTER_SAVE) | word CE(norm) eval_ttv3/word_ce_norm 1.9703 @0 -> 1.9565 @275 (last eval) | export-47; export-94; export-195 / HF: ppm-ttconvo-ft |
| [`si20k-vox-ppm`](#si20k-vox-ppm) | finished | 24857, 24916, 24920 | 2026-10-18 21:33 .. 2026-10-18 23:41 PT | ppm32b@12829, asrppm-nonorm@11488 | 0 -> 198 | word CE(norm) si20k eval_speech/word_ce_norm 2.213 @0 -> 1.4567 @198 | export-33; export-66; export-99 / HF: si20k-vox-ppm |
| [`si20k-vox-intl`](#si20k-vox-intl) | finished | 24858, 24879, 24903, 24917, 24921 | 2026-10-18 21:33 .. 2026-10-18 23:34 PT | intl14b@6242, asr14b@480 | 0 -> 198 | word CE(norm) si20k eval_speech/word_ce_norm 2.450 @0 -> 1.5449 @198 | export-33..165; export-198 / HF: si20k-vox-intl |
| [`si20k-txt-ppm`](#si20k-txt-ppm) | finished | 25045, 25049 | 2026-10-19 00:24 .. 2026-10-19 01:41 PT | ppm32b@12829, asrppm-nonorm@11488 | 0 -> 198 (25045 stopped at step 1 for bookkeeping, relaunched) | word CE(norm) si20k eval 2.376 @0 -> 1.4818 @198 (no user audio 1.9752) | export-33..165; export-198 / HF: si20k-txt-ppm |
| [`si20k-txt-intl`](#si20k-txt-intl) | finished | 25046, 25050 | 2026-10-19 00:24 .. 2026-10-19 01:41 PT | intl14b@6242, asr14b@480 | 0 -> 198 | word CE(norm) si20k eval 2.615 @0 -> 1.5639 @198 (no user audio 2.1303) | export-33..165; export-198 / HF: si20k-txt-intl |
| [`si20k-tin-ppm`](#si20k-tin-ppm) | finished | 25173, 25178 | 2026-10-19 02:16 .. 2026-10-19 03:15 PT | ppm32b@12829 | 0 -> 198 (25173 crashed before training) | word CE(norm) si20k eval 1.686 @0 -> 1.4148 @198 (A words blanked 1.9878) | export-198 / HF: si20k-tin-ppm |
| [`si20k-tin-intl`](#si20k-tin-intl) | finished | 25174 | 2026-10-19 02:16 .. 2026-10-19 03:18 PT | intl14b@6242 | 0 -> 198 | word CE(norm) si20k eval 1.828 @0 -> 1.4860 @198 (A words blanked 2.0708) | export-198 / HF: si20k-tin-intl |
| [`si20k-tdx-intl`](#si20k-tdx-intl) | finished | 25182, 25248 | 2026-10-19 02:36 .. 2026-10-19 03:59 PT | intl14b@6242 | 0 -> 198 | word CE(norm) si20k eval 1.5062 @198 (A words blanked 2.1147) | export-198 / HF: si20k-tdx-intl |
| [`si20k-vox-ppm11488`](#si20k-vox-ppm11488) | failed | 25245 | 2026-10-19 03:33 .. 2026-10-19 04:36 PT | ppm32b@11488, asrppm-nonorm@11488 | 0 -> 198 trained; final export lost (scratch full) -> last weights ex… | word CE(norm) si20k eval 2.221 @0 -> 1.4648 @198 (weights lost); 1.4677 @165 (last saved) | export-165 / HF: si20k-vox-ppm11488 |
| [`si20k-vox-ppm-nocorrupt`](#si20k-vox-ppm-nocorrupt) | stopped-with-save | 25282, 25283, 25293, 25312 | 2026-10-19 04:23 .. 2026-10-19 05:17 PT | ppm32b@12829, asrppm-nonorm@11488 | 0 -> 153 (stopped by the user) | word CE(norm) si20k eval 2.2134 @0 -> 1.4620 @150 (last eval) | export-33..132; export-198-stop153 / HF: si20k-vox-ppm-nocorrupt |
| [`xfer-tt-ss`](#xfer-tt-ss) | failed | 25393 | 2026-10-19 05:50 .. 2026-10-19 07:13 PT | ppm32b@12829, asrppm-nonorm@11488 | 0 -> 198 trained; ended without the final export | word CE(norm) si20k speech-speech eval 2.2131 @0 -> 1.4707 @198 (no user input 1.9932) | export-99 / HF: xfer-tt-ss |
| [`xfer-none-ss`](#xfer-none-ss) | finished | 25395, 25405, 25411 | 2026-10-19 05:52 .. 2026-10-19 07:20 PT | ppm32b@12829, asrppm-nonorm@11488 | 0 -> 198 | word CE(norm) si20k speech-speech eval 1.5410 @198 (no user input 2.0489) | export-198 / HF: xfer-none-ss |


### ppmvox-v1

**PPMVOX v1 (PPM + Voxtral user channel, 16 nodes)**  
group `ppmvox` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/ppmvox`

- **Jobs:** 28694, 28723, 28729, 28734
- **Dates:** 2026-10-24 03:07 .. 2026-10-24 04:34 PT (28694/28723/28729 aborted at preflight)
- **Base model:** unknown
- **Init:** pure PPM 32B dcp-16861 (resume_ppm: weights + AdamW) /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-16861; Voxtral adapter asrppm-w16861 w-840
- **Parents (resolved):** ppm32b@16861, asrppm-w16861@16861-w840, base:voxtral-mini-4b-realtime-2602
- **Produces:** ppmvox-v1@17035
- **Data:** vpod = SAM-separated podcast stems, 4144 rows both directions (8288 entries; trees 700k + intelligent; licence 'unknown, user ordered 10-24') + PPM pose micro + txtdup-full text-duplex rows (PPM_TEXT_OFFSET 1106560)
- **Recipe:** ppmvox/code trainer, ppmvox_16n.env on prod_ppm.env; 500 vox steps = 50 adapter-only warm-up + cycle pose,vpod,vpod,text; PPM LRs constant (trunk 5e-5 ...), vadapter/vgain 5e-4; world 128
- **Steps:** 16861 -> 17035 (vox 174 of planned 500)
- **Evals:** norm_word_ce = eval_vpod/word_ce_norm 1.9498 @16861 -> 1.8662 @16994 (last eval; v1 eval set, not comparable with v3+ eval set)
- **Checkpoints / exports:** w-16911; w-17018; w-17035 (FINAL bf16); dcp-17035
- **Local paths:** /scratch/amartyavarma/ppmvox/exports/ppmvox-v1/w-17035 (node-local, 16 hosts); /scratch/amartyavarma/ppmvox/ckpt/ppmvox-v1/dcp-17035
- **HF repos:** [InternalCan/ppmvox-16861](https://huggingface.co/InternalCan/ppmvox-16861) *(name match)*, [InternalCan/ppmvox-16861-e2etest](https://huggingface.co/InternalCan/ppmvox-16861-e2etest) *(name match)*
- **Notes:** Superseded at ~04:35 PT by v2/v3 (restart from dcp-16861 with shuffled / globally mixed vpod order, ppmvox_16n_v2.env:101). Eval set changed for v3 (v3 step-0 vpod 2.0231 vs v1 1.9498 on the same weights).
- **Citations:**
  - `/data/home/amartyavarma/ppm/ppmvox/PROMPT.md:5 -> goal (user 10-24): PPM extension with Voxtral`
  - `/data/home/amartyavarma/ppm/ppmvox/PROMPT.md:11 -> init pure PPM dcp-16861 (NOT 17850)`
  - `/data/home/amartyavarma/ppm/ppmvox/ppmvox_16n.env:3 -> PPM_RUN=ppmvox-v1`
  - `/data/home/amartyavarma/ppm/ppmvox/ppmvox_16n.env:7 -> PPMVOX_MAX_STEPS=500`
  - `/data/home/amartyavarma/ppm/ppmvox/ppmvox_16n.env:10 -> cycle pose,vpod,vpod,text`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-28734.log:2 -> adapter asrppm-w16861/w-840`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-28734.log:21 -> job 28734 start 10-24 03:07:12 PT, env ppmvox_16n.env`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-28734.log:152 -> vpod index 8288 entries / 4144 rows both directions`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-28734.log:425 -> EVAL step 16861 eval_vpod/word_ce_norm 1.9498`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-28734.log:945 -> EVAL step 16994 eval_vpod/word_ce_norm 1.8662`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-28734.log:988 -> export w-17035 FINAL`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-28734.log:990 -> STOPPED at step 17035, dcp-17035`
  - `/data/home/amartyavarma/ppm/ppmvox/DESIGN.md:159 -> optimizer/LR section (trunk 5e-5, vadapter 5e-4, constant)`

### ppmvox-v2

**PPMVOX v2 (prepared, never launched)**  
group `ppmvox` · status **abandoned** · dir `/data/home/amartyavarma/ppm/ppmvox`

- **Jobs:** unknown
- **Dates:** 2026-10-24 ~04:35 PT (env written)
- **Base model:** unknown
- **Init:** pure PPM dcp-16861 + asrppm-w16861 w-1346
- **Parents (resolved):** ppm32b@16861, asrppm-w16861@16861
- **Produces:** none recorded
- **Data:** vpod pool-shuffled order (vpod/index.jsonl rewritten by vpod_pool_shuffle.py)
- **Recipe:** ppmvox_16n_v2.env = v1 + shuffled vpod order + final matched adapter w-1346
- **Steps:** none run
- **Evals:** none recorded
- **Checkpoints / exports:** unknown
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Notes:** No ppmvox-v2 job log or runs dir exists (runs/ has v1, v3, v3x, v5, v5b only); v3 was launched instead. Row kept only so the version numbering is explained.
- **Citations:**
  - `/data/home/amartyavarma/ppm/ppmvox/ppmvox_16n_v2.env:3 -> PPM_RUN=ppmvox-v2`
  - `/data/home/amartyavarma/ppm/ppmvox/ppmvox_16n_v2.env:52 -> VOX_ADAPTER asrppm-w16861/w-1346`
  - `/data/home/amartyavarma/ppm/ppmvox/ppmvox_16n_v2.env:101 -> v2 decision b: pool-shuffled restart`
  - `/data/home/amartyavarma/ppm/ppmvox/ppmvox_16n_v3.env:101 -> v3 (user: 'please do a proper mix') replaced it`

### ppmvox-v3

**PPMVOX v3 (w-17127; global vpod mix)**  
group `ppmvox` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/ppmvox`

- **Jobs:** 28998, 29005
- **Dates:** 2026-10-24 05:16 .. 2026-10-24 07:19 PT (28998 aborted at preflight)
- **Base model:** unknown
- **Init:** pure PPM dcp-16861 (resume_ppm) + Voxtral adapter asrppm-w16861 w-1346 (sha 9f97877dec3f5956)
- **Parents (resolved):** ppm32b@16861, asrppm-w16861@16861, base:voxtral-mini-4b-realtime-2602
- **Produces:** ppmvox-v3@17127
- **Data:** vpod3 = SAM-separated podcast stems, 4134 rows in both directions (8268 entries; global mix over many podcasts) + PPM pose micro + txtdup-full text
- **Recipe:** ppmvox_16n_v3.env (= v2 + vpod3 index + vpod3_host.py daemons); 500 vox steps planned, 50 adapter-only warm-up, cycle pose,vpod,vpod,text; PPM constant LRs; world 128; saves off (weights export only)
- **Steps:** 16861 -> 17127 (vox 266 of 500; stopped on user order 'plateaued')
- **Evals:** norm_word_ce = eval_vpod/word_ce_norm 2.0231 @16861 -> 1.9217 @17071 (last logged eval); other = listening eval (v3 held-out set, clean intact): base 2.0231, w-16911 1.9985, w-17032 1.9242
- **Checkpoints / exports:** w-16911; w-17032; w-17127 (FINAL bf16)
- **Local paths:** /scratch/amartyavarma/ppmvox/exports/ppmvox-v3/w-17127 (node-local, 16 hosts); /data/home/amartyavarma/ppm/ppmvox/release-w17127/ppm-voxtral-w17127
- **HF repos:** [InternalCan/ppmvox-v3-16861](https://huggingface.co/InternalCan/ppmvox-v3-16861), [InternalCan/ppm-voxtral-w17127](https://huggingface.co/InternalCan/ppm-voxtral-w17127)
- **Notes:** Init of the Aloe FT vox17127 (29345), its persona twin (29444) and the Brianna/Rosie speaker LoRAs. Optimizer state not saved (FULL_SAVE=0), so a continuation re-inits AdamW.
- **Citations:**
  - `/data/home/amartyavarma/ppm/ppmvox/ppmvox_16n_v3.env:3 -> PPM_RUN=ppmvox-v3`
  - `/data/home/amartyavarma/ppm/ppmvox/ppmvox_16n_v3.env:52 -> VOX_ADAPTER w-1346 (sha line 53)`
  - `/data/home/amartyavarma/ppm/ppmvox/ppmvox_16n_v3.env:101 -> v3 = global mix restart from dcp-16861`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-29005.log:20 -> job 29005 start 10-24 05:16:53 PT env v3`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-29005.log:151 -> vpod3 index 8268 entries / 4134 rows`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-29005.log:424 -> EVAL step 16861 eval_vpod/word_ce_norm 2.0231`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-29005.log:1023 -> EVAL step 17071 eval_vpod/word_ce_norm 1.9217`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-29005.log:1080 -> export w-17127 FINAL`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-29005.log:1082 -> STOPPED at 17127, no full save`
  - `/data/home/amartyavarma/ppm/control/OPS_PLAYBOOK.md:390 -> user 10-24 07:18 'ppm voxtral seems to have plateaued - save and exit'`
  - `/data/home/amartyavarma/ppm/control/OPS_PLAYBOOK.md:396 -> listening-eval numbers (v3 held-out set)`
  - `/data/home/amartyavarma/ppm/ppmvox/release-w17127/logs/pipeline-29675.log:5 -> full backup InternalCan/ppmvox-v3-16861`
  - `/data/home/amartyavarma/ppm/ppmvox/release-w17127/logs/pipeline-29675.log:264 -> release InternalCan/ppm-voxtral-w17127 complete + verified`

### ppmvox-v3x

**PPMVOX v3x extension (prepared, cancelled before start)**  
group `ppmvox` · status **abandoned** · dir `/data/home/amartyavarma/ppm/ppmvox`

- **Jobs:** 29218
- **Dates:** 2026-10-24 (pending, scancelled ~07:18 PT)
- **Base model:** unknown
- **Init:** PPMVOX v3 export w-17361 (planned; v3 actually ended at 17127)
- **Parents (resolved):** ppmvox-v3@17127
- **Produces:** none recorded
- **Data:** as v3 (disjoint one-row-per-episode pool)
- **Recipe:** ppmvox_16n_v3x.env: INIT_MODE=export, 520 steps, LR0 for 20 steps, no warm-up
- **Steps:** none run
- **Evals:** none recorded
- **Checkpoints / exports:** unknown
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Notes:** runs/ppmvox-v3x-j29218 exists but is empty (never trained).
- **Citations:**
  - `/data/home/amartyavarma/ppm/ppmvox/ppmvox_16n_v3x.env:9 -> MAX_STEPS 520`
  - `/data/home/amartyavarma/ppm/ppmvox/ppmvox_16n_v3x.env:25 -> INIT_FROM w-17361`
  - `/data/home/amartyavarma/ppm/ppmvox/DESIGN.md:291 -> v3x extension, job 29218 afterany:29005`
  - `/data/home/amartyavarma/ppm/control/OPS_PLAYBOOK.md:390 -> main scancel pending v3x 29218 + 29228`

### ppmvox-v5

**PPMVOX v5 (8,1,1 dcp-17850 init, pose:vpod 1:1)**  
group `ppmvox` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/ppmvox`

- **Jobs:** 29813, 29830, 29853, 29859, 29863, 30024
- **Dates:** 2026-10-25 00:25 .. 2026-10-25 03:11 PT
- **Base model:** unknown
- **Init:** 8,1,1 dcp-17850 (resume_ppm) + Voxtral adapter asrppm-811-17850 w-720 (sha a4397ab0e4a0fa5d)
- **Parents (resolved):** ppm811-append3@17850, asrppm-811-17850@17850, base:voxtral-mini-4b-realtime-2602
- **Produces:** ppmvox-v5@18100
- **Data:** vpod3 (4134 rows both directions) + FRESH 8,1,1 pose stream (fresh811) + txtdup-full text (offset 1169024)
- **Recipe:** ppmvox_16n_v5.env = v3 recipe except init 17850 + matched adapter, cycle pose,vpod,pose,vpod,text, full saves on; 29813 ran the 50-step adapter warm-up (ppmvox_16n_v5wu.env) saved to ckpt-wu; 29863 continued; 30024 = converter writing bf16 w-18100 from dcp-18100
- **Steps:** 17850 -> 17900 (warm-up, 29813) -> 18100 (vox 250 of 500, SAVE_NOW then paused for v5b)
- **Evals:** norm_word_ce = eval_vpod/word_ce_norm 2.0342 @17850 -> 1.9744 @17900 -> 1.9177 @18084 (last eval)
- **Checkpoints / exports:** ckpt-wu step 17900; dcp-18100 (full); w-18100 (bf16); merged world-64 dcp-18100
- **Local paths:** /data/home/amartyavarma/ppm/ppmvox/ckpt-wu/ppmvox-v5; /scratch/amartyavarma/ppmvox/exports/ppmvox-v5/w-18100; /data/home/amartyavarma/ppm/ppmvox/v5b-merge/dcp-18100; /scratch/amartyavarma/ppmvox/ckpt/v5-18100-merged-keep/dcp-18100
- **HF repos:** [InternalCan/ppmvox-v5-17850](https://huggingface.co/InternalCan/ppmvox-v5-17850), [InternalCan/ppmvox-v5-18100-merged-state](https://huggingface.co/InternalCan/ppmvox-v5-18100-merged-state)
- **Notes:** Uses the 8,1,1 lineage (17850), not pure PPM. 29830 aborted preflight, 29853/29859 died on data stalls. w-18100 is the init of the pose-duplex FT (other group). Pose eval off in this run (PPMVOX_POSE_EVAL=0).
- **Citations:**
  - `/data/home/amartyavarma/ppm/ppmvox/ppmvox_16n_v5.env:3 -> v5 = v3 recipe except init 17850 + asrppm-811-17850, pose:vpod 1:1, fresh pose, full saves`
  - `/data/home/amartyavarma/ppm/ppmvox/ppmvox_16n_v5.env:24 -> INIT_FROM dcp-17850`
  - `/data/home/amartyavarma/ppm/ppmvox/ppmvox_16n_v5.env:74 -> VOX_ADAPTER asrppm-811-17850/w-720`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-29813.log:21 -> job 29813 start 10-25 00:25:16 PT env v5wu`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-29813.log:426 -> EVAL step 17850 eval_vpod/word_ce_norm 2.0342`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-29813.log:861 -> EVAL step 17900 eval_vpod 1.9744`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-29813.log:862 -> warm-up state saved (step 17900)`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-29863.log:21 -> job 29863 start 10-25 00:50:59 PT env v5`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-29863.log:999 -> EVAL step 18084 eval_vpod/word_ce_norm 1.9177`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-29863.log:1019 -> SAVE_NOW at 18100: dcp-18100`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-30024.log:130 -> w-18100 bf16 export (converter)`
  - `/data/home/amartyavarma/ppm/ppmvox/launch/V5B_PLAN_CUT.md:20 -> HF InternalCan/ppmvox-v5-17850/w-18100`
  - `/data/home/amartyavarma/ppm/control/OPS_PLAYBOOK.md:786 -> HF InternalCan/ppmvox-v5-18100-merged-state verified`
  - `/data/home/amartyavarma/ppm/control/OPS_PLAYBOOK.md:563 -> pose-duplex FT staged from v5 final export (InternalCan/ppmvox-v5-17850)`

### ppmvox-v5b

**PPMVOX v5b (8 nodes, Voxtral encoder unfrozen)**  
group `ppmvox` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/ppmvox`

- **Jobs:** 30104
- **Dates:** 2026-10-25 03:32 .. 2026-10-25 05:10 PT
- **Base model:** unknown
- **Init:** merged world-64 full save of v5 dcp-18100 (vox 250)
- **Parents (resolved):** ppmvox-v5@18100
- **Produces:** ppmvox-v5b@18163
- **Data:** as v5 (v5 per-step micros reproduced: each rank plays 2 v5 ranks); vpod 16480 train / 56 eval entries
- **Recipe:** ppmvox_8n_v5b.env = v5 + 2 slots per rank (GA x2, world 64) + Voxtral encoder trainable (voxenc LR 1e-5, fresh AdamW, 50-step warm-up, clip 1.0)
- **Steps:** 18100 -> 18163 (vox 313); STOP_AFTER_SAVE on user go to restart MoT
- **Evals:** norm_word_ce = eval_vpod/word_ce_norm 1.9183 @18100 -> best 1.9155 @18142 -> 1.9167 @18163 (final)
- **Checkpoints / exports:** dcp-18163 (full, incl voxenc AdamW); bf16 export w-18163 NOT written
- **Local paths:** /scratch/amartyavarma/ppmvox/ckpt/ppmvox-v5b/dcp-18163 (8 hosts)
- **HF repos:** [InternalCan/ppmvox-v5b-18163-state](https://huggingface.co/InternalCan/ppmvox-v5b-18163-state), [InternalCan/ppmvox-v5b](https://huggingface.co/InternalCan/ppmvox-v5b)
- **Notes:** Dead end per user (unfrozen encoder judged not working). One host's part of dcp-18163 was written via /data ckpt-hostsave and later deleted after HF verify (OPS_PLAYBOOK:786).
- **Citations:**
  - `/data/home/amartyavarma/ppm/ppmvox/ppmvox_8n_v5b.env:1 -> v5b = v5 recipe exactly + v5b block (world 64, 2 slots/rank, Voxtral unfrozen)`
  - `/data/home/amartyavarma/ppm/ppmvox/ppmvox_8n_v5b.env:135 -> PPMVOX_LR_VOXENC=1e-5`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-30104.log:12 -> job 30104 start 10-25 03:32:51 PT env v5b`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-30104.log:3 -> resumed from merged dcp-18100`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-30104.log:84 -> voxenc trainable 0.970B params`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-30104.log:98 -> EVAL step 18100 eval_vpod 1.9183`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-30104.log:335 -> EVAL step 18142 eval_vpod 1.9155`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-30104.log:357 -> EVAL step 18163 eval_vpod 1.9167`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-30104.log:381 -> final bf16 export w-18163 NOT written`
  - `/data/home/amartyavarma/ppm/ppmvox/logs/ppmvox16-30104.log:383 -> STOPPED at 18163, dcp-18163`
  - `/data/home/amartyavarma/ppm/control/OPS_PLAYBOOK.md:631 -> user: assume Voxtral unfreeze isn't working`
  - `/data/home/amartyavarma/ppm/control/OPS_PLAYBOOK.md:635 -> user go: stop v5b + restart MoT`
  - `/data/home/amartyavarma/ppm/control/OPS_PLAYBOOK.md:786 -> HF InternalCan/ppmvox-v5b-18163-state verified`
  - `/data/home/amartyavarma/ppm/ppmvox/launch/V5B_RUNBOOK.md:50 -> HF watcher repo InternalCan/ppmvox-v5b`

### asrppm-rmsnorm-11488

**asrppm (first ASR adapter on PPM dcp-11488, RMS-normed adapter)**  
group `asr-adapter` · status **abandoned** · dir `/data/home/amartyavarma/ppm/asr-adapter`

- **Jobs:** 21908, 21909, 21917, 21972
- **Dates:** 2026-10-16 01:54 .. 2026-10-16 03:20 PT
- **Base model:** unknown
- **Init:** frozen PPM dcp-11488 backbone (bb-11488); adapter init j19174 w-3695
- **Parents (resolved):** ppm32b@11488, base:voxtral-mini-4b-realtime-2602, qwen1ch-s4-asr2-ga1@3695
- **Produces:** asrppm-rmsnorm@11488
- **Data:** streamed MLS-eng train (stream_mls.py seed 20261007), 15006-position packs
- **Recipe:** train_asr_ppm.py, asr.sbatch; adapter only, LR 5e-4 AdamW, warmup 50, WSD decay to 0 over 12160..15200; world 16
- **Steps:** 0 -> 305 (21917) -> 496 (21972)
- **Evals:** norm_word_ce = transcript normalised word CE best 3.7845 @480 (WER 1.002): did not learn
- **Checkpoints / exports:** w-305; w-360; w-480; w-496
- **Local paths:** /data/home/amartyavarma/ppm/asr-adapter/ckpt-weka/asrppm
- **HF repos:** none named in run files
- **Notes:** Replaced by asrppm-nonorm (VADAPT_NORM=none) at 03:20 PT 10-16.
- **Citations:**
  - `/data/home/amartyavarma/ppm/asr-adapter/README.md:16 -> adapter init j19174 w-3695`
  - `/data/home/amartyavarma/ppm/asr-adapter/README.md:20 -> data MLS-eng stream`
  - `/data/home/amartyavarma/ppm/asr-adapter/README.md:22 -> LR 5e-4 AdamW`
  - `/data/home/amartyavarma/ppm/asr-adapter/logs/asrppm-21917.log:400 -> stopped 305, best norm CE 3.8385 @240`
  - `/data/home/amartyavarma/ppm/asr-adapter/logs/asrppm-21972.log:290 -> stopped 496, best norm CE 3.7845 @480`

### asrppm-nonorm

**asrppm-nonorm (ASR adapter w-1972 on PPM dcp-11488)**  
group `asr-adapter` · status **finished** · dir `/data/home/amartyavarma/ppm/asr-adapter`

- **Jobs:** 21986, 21990, 22237
- **Dates:** 2026-10-16 03:20 .. 2026-10-16 07:49 PT
- **Base model:** unknown
- **Init:** frozen PPM dcp-11488 backbone; adapter init j19174 w-3695; VADAPT_NORM=none
- **Parents (resolved):** ppm32b@11488, base:voxtral-mini-4b-realtime-2602, qwen1ch-s4-asr2-ga1@3695
- **Produces:** asrppm@11488
- **Data:** streamed MLS-eng train (seed 20261007)
- **Recipe:** train_asr_ppm.py; adapter-only, LR 5e-4, world 16 x GA 1; eval every 120 steps
- **Steps:** 0 -> 1972 (STOP_AFTER_SAVE; user declared finished at 1972)
- **Evals:** norm_word_ce = transcript normalised word CE best 0.2549 @1680 (WER 0.811); 0.2565 at the 1972 resume eval
- **Checkpoints / exports:** w-1680; w-1800; w-1920; w-1972; final-s1972
- **Local paths:** /data/home/amartyavarma/ppm/asr-adapter/ckpt-weka/asrppm-nonorm; /data/home/amartyavarma/ppm/asr-adapter/ckpt-final/asrppm-nonorm/final-s1972/adapter-bf16-w1972
- **HF repos:** none named in run files
- **Notes:** The 'ASR w-1972' adapter (sha 8919c69d29e4eb21) used by aloe-m1vox/v2vox/v2vox-raw/w14503 FTs, all si20k PPM runs and xfer runs. Backup on GCS only (pre HF rule).
- **Citations:**
  - `/data/home/amartyavarma/ppm/asr-adapter/logs/asrppm-21990.log:2109 -> STOPPED at 1972; best norm CE 0.2549 @1680 (WER 0.811)`
  - `/data/home/amartyavarma/ppm/asr-adapter/README.md:41 -> 22237 resume step-0 eval norm CE 0.2565, died before 1973`
  - `/data/home/amartyavarma/ppm/asr-adapter/README.md:43 -> user: run FINISHED at 1972; final-s1972 (+GCS)`

### asrppm-w15973

**asrppm-w15973 (ASR adapter on PPM w-15973)**  
group `asr-adapter` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/asr-adapter`

- **Jobs:** 26064
- **Dates:** 2026-10-20 15:18 .. 2026-10-20 18:32 PT
- **Base model:** unknown
- **Init:** frozen PPM w-15973 backbone (/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-15973 -> bb-15973); same recipe as nonorm
- **Parents (resolved):** ppm32b@15973, base:voxtral-mini-4b-realtime-2602
- **Produces:** asrppm@15973
- **Data:** streamed MLS-eng (stream-w15973, reference order)
- **Recipe:** train_asr_ppm.py (PPM_STEP knob), asr.sbatch, VADAPT_NORM=none; adapter-only LR 5e-4; world 16 on slinky-15+18
- **Steps:** 0 -> 1441 (STOP_AFTER_SAVE)
- **Evals:** norm_word_ce = transcript normalised word CE best 0.2629 @1320 (WER 0.823)
- **Checkpoints / exports:** w-1200; w-1320; w-1440; w-1441
- **Local paths:** /data/home/amartyavarma/ppm/asr-adapter/ckpt-weka/asrppm-w15973; /data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973/adapter-init/asrppm-w15973-w-1441
- **HF repos:** [InternalCan/asr-adapter-ppm](https://huggingface.co/InternalCan/asr-adapter-ppm)
- **Notes:** w-1441 = adapter of the Aloe reference FTs 26232 / 26422 / 26467 and the samsep 16861 FT 28249. Canonical id asrppm@15973 = export w-1441.
- **Citations:**
  - `/data/home/amartyavarma/ppm/asr-adapter/README.md:45 -> asrppm-w15973 same recipe on PPM w-15973`
  - `/data/home/amartyavarma/ppm/asr-adapter/logs/asrppm-26064.log:3 -> backbone w-15973 frozen`
  - `/data/home/amartyavarma/ppm/asr-adapter/logs/asrppm-26064.log:1438 -> EVAL step 1320 norm CE 0.2629`
  - `/data/home/amartyavarma/ppm/asr-adapter/logs/asrppm-26064.log:1566 -> STOPPED at 1441, export w-1441`
  - `/data/home/amartyavarma/ppm/asr-adapter/w15973/hf_backup/hf_asrppm-asrppm-w15973-w-1080.log:1 -> HF InternalCan/asr-adapter-ppm/asrppm-w15973/`

### asrppm-w16861

**asrppm-w16861 (ASR adapter matched to pure PPM dcp-16861; 'w-480')**  
group `asr-adapter` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/asr-adapter`

- **Jobs:** 28311, 28462
- **Dates:** 2026-10-24 00:14 .. 2026-10-24 04:02 PT
- **Base model:** unknown
- **Init:** frozen PPM dcp-16861 backbone (bb-16861 from hf://InternalCan/ppm-v1-b/dcp-16861); fresh adapter, VADAPT_NORM=none
- **Parents (resolved):** ppm32b@16861, base:voxtral-mini-4b-realtime-2602, qwen1ch-s4-asr2-ga1@3695
- **Produces:** asrppm@16861-w480, asrppm@16861-w840, asrppm@16861
- **Data:** streamed MLS-eng (stream-w16861)
- **Recipe:** train_asr_ppm.py (w16861/ GA2 variant); 28311 world 8 x GA 2 (logical 16) on slinky-18; 28462 world 14 on slinky-18+4; MAX_STEPS 1500; LR 5e-4
- **Steps:** 0 -> 340 (28311) -> 1346 (28462, STOP_AFTER_SAVE; target 1500 never reached)
- **Evals:** norm_word_ce = transcript normalised word CE best 0.2662 @1200 (WER 0.979); w-480: 0.2899 (WER 0.572); w-840: 0.2871
- **Checkpoints / exports:** w-480; w-840; w-1080; w-1200; w-1320; w-1346
- **Local paths:** /data/home/amartyavarma/ppm/asr-adapter/ckpt-weka/asrppm-w16861; /data/home/amartyavarma/ppm/aloe-ft-32-samsep/adapter-init/asrppm-w16861-w-480; /data/home/amartyavarma/ppm/ppmvox/adapters/asrppm-w16861/w-1346
- **HF repos:** [InternalCan/asrppm-w16861](https://huggingface.co/InternalCan/asrppm-w16861)
- **Notes:** w-480 is the adapter inside the a16861w480 Aloe FT (28511) and the 10-24 live demo; w-840 fed PPMVOX v1; w-1346 fed PPMVOX v3. w-480 was later pruned from ckpt-weka (copy kept under aloe-ft-32-samsep/adapter-init).
- **Citations:**
  - `/data/home/amartyavarma/ppm/asr-adapter/logs/asrppm-28311.log:3 -> backbone 16861 from hf InternalCan/ppm-v1-b/dcp-16861`
  - `/data/home/amartyavarma/ppm/asr-adapter/logs/asrppm-28311.log:396 -> stopped at 340`
  - `/data/home/amartyavarma/ppm/asr-adapter/logs/asrppm-28462.log:226 -> EVAL step 480 norm CE 0.2899, WER 0.572`
  - `/data/home/amartyavarma/ppm/asr-adapter/logs/asrppm-28462.log:595 -> EVAL step 840 norm CE 0.2871`
  - `/data/home/amartyavarma/ppm/asr-adapter/logs/asrppm-28462.log:964 -> EVAL step 1200 norm CE 0.2662`
  - `/data/home/amartyavarma/ppm/asr-adapter/logs/asrppm-28462.log:1117 -> STOPPED at 1346, best 0.2662 @1200`
  - `/data/home/amartyavarma/ppm/asr-adapter/w16861/hf_backup/hf_asrppm-w16861-w-480.log:1 -> HF InternalCan/asrppm-w16861/w-480/`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/A16861_RERUN_CANCELLED.txt:1 -> user went with w-480; w-1500 rerun dropped`

### asrppm-811-17850

**asrppm-811-17850 (8,1,1 ASR adapter on dcp-17850)**  
group `asr-adapter` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/asr-adapter`

- **Jobs:** 29671, 29676
- **Dates:** 2026-10-24 22:09 .. 2026-10-25 00:26 PT
- **Base model:** unknown
- **Init:** frozen 8,1,1 dcp-17850 backbone (bb-17850); fresh adapter, VADAPT_NORM=none
- **Parents (resolved):** ppm811-append3@17850, base:voxtral-mini-4b-realtime-2602, qwen1ch-s4-asr2-ga1@3695
- **Produces:** asrppm811@17850
- **Data:** streamed MLS-eng (stream-811)
- **Recipe:** train_asr_ppm.py, world 14 (slinky-18 8 + slinky-4 6), MAX_STEPS 1500, LR 5e-4
- **Steps:** 0 -> 835 (STOP_AFTER_SAVE; target 1500)
- **Evals:** norm_word_ce = transcript normalised word CE best 0.2878 @720 (WER 0.743); other = text-duplex held-out B norm word CE 0.7979 (frozen backbone, KL 0)
- **Checkpoints / exports:** w-480; w-600; w-720; w-835
- **Local paths:** /data/home/amartyavarma/ppm/asr-adapter/ckpt-weka/asrppm-811-17850
- **HF repos:** [InternalCan/asrppm-811-17850](https://huggingface.co/InternalCan/asrppm-811-17850)
- **Notes:** w-720 (sha a4397ab0e4a0fa5d) is the adapter of PPMVOX v5 / v5b. Canonical id asrppm811@17850 = w-720 (I could not find a pre-agreed id for the 8,1,1 adapter).
- **Citations:**
  - `/data/home/amartyavarma/ppm/asr-adapter/logs/asrppm-29671.log:1 -> 29671 gated, cancelled`
  - `/data/home/amartyavarma/ppm/asr-adapter/logs/asrppm-29676.log:1 -> job 29676 start 10-24 22:24:57 PT, PPM_STEP=17850`
  - `/data/home/amartyavarma/ppm/asr-adapter/logs/asrppm-29676.log:3663 -> EVAL step 720 norm CE 0.2878`
  - `/data/home/amartyavarma/ppm/asr-adapter/logs/asrppm-29676.log:3782 -> STOPPED at 835, best 0.2878 @720`
  - `/data/home/amartyavarma/ppm/asr-adapter/w811-17850/CHAIN.md:3 -> hf_watch -> private InternalCan/asrppm-811-17850`
  - `memory:only-demo-aloe-mothers.md:11 -> the live demo was taken down 10-24 for the 8,1,1 ASR adapter`

### aloe-ft-21980

**Aloe FT (first PPM Aloe FT, partner words+Mimi on A)**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-ft`

- **Jobs:** 21980
- **Dates:** 2026-10-16 03:15 .. 2026-10-16 04:12 PT
- **Base model:** unknown
- **Init:** PPM dcp-11488 converted: /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488
- **Parents (resolved):** ppm32b@11488
- **Produces:** aloe-ft-21980@366
- **Data:** Aloe handover none-crossbleed s2h-m1 stores: 11,700 train rows x 2 orientations = 23,400 micros, eval 240 rows x 2
- **Recipe:** qqa-ft trainer (aloe.env, run C mapping): 366 steps, warmup 12, linear decay to 0; trunk 3e-6, spec 6e-5, proj 3e-5, audio 6e-5, tower/light 1.5e-5; world 64
- **Steps:** 0 -> 366
- **Evals:** norm_word_ce = eval_speech/word_ce_norm 1.8303 @0 -> 1.7944 @366
- **Checkpoints / exports:** export-244; export-305; export-366
- **Local paths:** /scratch/amartyavarma/aloe-ft/exports/export-366 (node-local)
- **HF repos:** none named in run files
- **Notes:** Eval values are on A=words+Mimi rows, not comparable with the Voxtral-A FTs. Backup was GCS (gcs_local_exports.sbatch), no HF repo named.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-ft/logs/aloeft-21980.log:1 -> job 21980 start 10-16 03:15:57 PT, init local-11488`
  - `/data/home/amartyavarma/ppm/aloe-ft/logs/aloeft-21980.log:52 -> data plan 11,700 rows x 2 orientations`
  - `/data/home/amartyavarma/ppm/aloe-ft/logs/aloeft-21980.log:57 -> EVAL step 0 1.8303`
  - `/data/home/amartyavarma/ppm/aloe-ft/logs/aloeft-21980.log:642 -> EVAL step 366 1.7944`
  - `/data/home/amartyavarma/ppm/aloe-ft/logs/aloeft-21980.log:644 -> export-366`
  - `/data/home/amartyavarma/ppm/aloe-ft/aloe.env:9 -> QQA_STEPS=366`
  - `/data/home/amartyavarma/ppm/aloe-m1vox-ft/aloe_m1vox.env:1 -> m1vox = 21980's env verbatim + Voxtral-on-A block (so 21980 had partner words+Mimi on A)`
  - `memory:aloe-handover-only.md:17 -> none-crossbleed s2h-m1 stores, 11,700 train / 240 eval`

### aloe-vox-ft

**Aloe vox FT (pose FT recipe + Voxtral on A)**  
group `aloe-speech-ft` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/aloe-vox-ft`

- **Jobs:** 22275, 22284
- **Dates:** 2026-10-16 15:41 .. 2026-10-16 16:23 PT
- **Base model:** unknown
- **Init:** /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-12465 + ASR adapter asrppm-nonorm w-1972
- **Parents (resolved):** ppm32b@12465, asrppm-nonorm@11488
- **Produces:** aloe-vox-ft@72
- **Data:** aloe-pose-ft data: 27,517 qualifying B-track items (pose FT stores)
- **Recipe:** aloe_vox.env = aloe-pose-ft/aloe_pose.env (22091) + [VOX] (A = Voxtral latents through ASR adapter); planned 430 steps, warmup 14, decay to 0
- **Steps:** 0 -> 72 (STOP_AFTER_SAVE; planned 430)
- **Evals:** norm_word_ce = 1.9360 @0 -> 1.8697 @72
- **Checkpoints / exports:** export-72
- **Local paths:** /scratch/amartyavarma/aloe-vox-ft/exports/export-72 (node-local)
- **HF repos:** none named in run files
- **Notes:** 22275 stopped after the step-0 eval (speed fix, SPEED_FIX.md). Pose state of this run (pose on or off) not verified; the voxall env says the no-pose run is 22291.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-vox-ft/aloe_vox.env:1 -> = aloe_pose.env (22091) except [VOX]`
  - `/data/home/amartyavarma/ppm/aloe-vox-ft/aloe_vox.env:12 -> QQA_STEPS=430`
  - `/data/home/amartyavarma/ppm/aloe-vox-ft/logs/aloevoxft-22284.log:1 -> init local-12465`
  - `/data/home/amartyavarma/ppm/aloe-vox-ft/logs/aloevoxft-22284.log:218 -> vadapter <- asrppm-nonorm w-1972`
  - `/data/home/amartyavarma/ppm/aloe-vox-ft/logs/aloevoxft-22284.log:492 -> EVAL step 72 1.8697`
  - `/data/home/amartyavarma/ppm/aloe-vox-ft/logs/aloevoxft-22284.log:494 -> STOP_AFTER_SAVE at 72`

### aloe-voxall-ft

**Aloe voxall FT (all handover conversations, Voxtral A)**  
group `aloe-speech-ft` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/aloe-voxall-ft`

- **Jobs:** 22303
- **Dates:** 2026-10-16 16:31 .. 2026-10-16 16:36 PT
- **Base model:** unknown
- **Init:** /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-12465 + asrppm-nonorm w-1972
- **Parents (resolved):** ppm32b@12465, asrppm-nonorm@11488
- **Produces:** none recorded
- **Data:** ALL Aloe handover conversations, every speaker track a B candidate minus bleed into B; no pose
- **Recipe:** aloe_voxall.env = aloe_vox.env (22284) + [VOXALL]; planned 498 steps, warmup 16
- **Steps:** 0 -> 11 (STOP_AFTER_SAVE; planned 498)
- **Evals:** norm_word_ce = 1.9392 @0 (only eval)
- **Checkpoints / exports:** unknown
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Notes:** Stopped almost immediately (likely superseded by the handover-only rule, memory aloe-handover-only.md); weights-only export at 11 not tracked further.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-voxall-ft/aloe_voxall.env:1 -> data = all handover conversations`
  - `/data/home/amartyavarma/ppm/aloe-voxall-ft/aloe_voxall.env:17 -> QQA_STEPS=498`
  - `/data/home/amartyavarma/ppm/aloe-voxall-ft/logs/aloevoxallft-22303.log:259 -> EVAL step 0 1.9392`
  - `/data/home/amartyavarma/ppm/aloe-voxall-ft/logs/aloevoxallft-22303.log:466 -> STOP_AFTER_SAVE at step 11`

### aloe-m1vox-ft

**Aloe m1vox FT (first Voxtral-on-A Aloe FT, job 22335)**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-m1vox-ft`

- **Jobs:** 22315, 22335
- **Dates:** 2026-10-16 16:47 .. 2026-10-16 18:30 PT
- **Base model:** unknown
- **Init:** /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 + ASR adapter asrppm-nonorm w-1972 (x learnable scalar s)
- **Parents (resolved):** ppm32b@11488, asrppm-nonorm@11488
- **Produces:** aloe-m1vox-ft@366
- **Data:** Aloe handover none-crossbleed s2h-m1 stores (Model-1) + partner A FLACs for Voxtral
- **Recipe:** aloe_m1vox.env = 21980's aloe.env verbatim + [VOX] (A = Voxtral latents via adapter x s) + [M1VOX] data; 366 steps; compile_decoder on
- **Steps:** 22315: 0 -> 111 (stopped, speed fix); 22335: fresh 0 -> 366
- **Evals:** norm_word_ce = 1.9687 @0 -> 1.7934 @366
- **Checkpoints / exports:** export-244; export-305; export-366
- **Local paths:** /data/home/amartyavarma/ppm/aloe-m1vox-ft/export-366
- **HF repos:** none named in run files
- **Notes:** Template for all later Voxtral-A Aloe FTs. Backup GCS (gcs_local_exports.sbatch).
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-m1vox-ft/aloe_m1vox.env:1 -> = 21980 env + Voxtral block`
  - `/data/home/amartyavarma/ppm/aloe-m1vox-ft/logs/aloem1voxft-22315.log:562 -> 22315 STOP_AFTER_SAVE at 111`
  - `/data/home/amartyavarma/ppm/aloe-m1vox-ft/logs/aloem1voxft-22335.log:1 -> 22335 start 10-16 17:18 PT, init local-11488`
  - `/data/home/amartyavarma/ppm/aloe-m1vox-ft/logs/aloem1voxft-22335.log:60 -> VOX_ADAPTER asrppm-nonorm final-s1972`
  - `/data/home/amartyavarma/ppm/aloe-m1vox-ft/logs/aloem1voxft-22335.log:249 -> EVAL step 0 1.9687`
  - `/data/home/amartyavarma/ppm/aloe-m1vox-ft/logs/aloem1voxft-22335.log:836 -> EVAL step 366 1.7934`
  - `/data/home/amartyavarma/ppm/aloe-m1vox-ft/logs/aloem1voxft-22335.log:838 -> export-366`
  - `memory:aloe-ft-voxtral-a.md:11 -> every Aloe FT from 10-16 copies aloe-m1vox-ft (job 22335)`

### aloe-v2vox-ft

**Aloe v2vox FT (v2 recipe, Voxtral on A, loudness-normalised A)**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-v2vox-ft`

- **Jobs:** 22420
- **Dates:** 2026-10-16 20:12 .. 2026-10-16 20:42 PT
- **Base model:** unknown
- **Init:** /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 + asrppm-nonorm w-1972
- **Parents (resolved):** ppm32b@11488, asrppm-nonorm@11488
- **Produces:** aloe-v2vox-ft@253
- **Data:** v2 plan (first chunks, no-text prompt, B voice clip 30%, light_frac 0.1) on the s2h-m1 none stores + v2-plan A tracks
- **Recipe:** aloe_v2vox.env = aloe_m1vox.env (22335) + aloe-v2-ft changes (253 steps, warmup 8, DEC_K 1, exports 42..211, 253)
- **Steps:** 0 -> 253
- **Evals:** norm_word_ce = 2.0371 @0 -> 1.8246 @253
- **Checkpoints / exports:** export-253
- **Local paths:** /data/home/amartyavarma/ppm/aloe-v2vox-ft/export-253
- **HF repos:** none named in run files
- **Notes:** Superseded by aloe-v2vox-raw (raw A gain).
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-v2vox-ft/aloe_v2vox.env:1 -> = m1vox env + v2 changes`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-ft/logs/aloev2vox-22420.log:1 -> job 22420 start 10-16 20:12 PT`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-ft/logs/aloev2vox-22420.log:64 -> VOX_ADAPTER asrppm-nonorm w-1972`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-ft/logs/aloev2vox-22420.log:312 -> EVAL step 0 2.0371`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-ft/logs/aloev2vox-22420.log:781 -> EVAL step 253 1.8246`

### aloe-v2vox-raw

**Aloe v2vox raw-A FT (export-253; the '11k' live demo model)**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-v2vox-raw`

- **Jobs:** 23060, 23079, 23081
- **Dates:** 2026-10-17 ~16:5x .. 2026-10-17 18:31 PT
- **Base model:** unknown
- **Init:** /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 + asrppm-nonorm w-1972
- **Parents (resolved):** ppm32b@11488, asrppm-nonorm@11488
- **Produces:** aloe-v2vox-raw@253
- **Data:** as aloe-v2vox-ft, but A into Voxtral at raw recorded level (VOX_A_GAIN=raw, ALOE_RAWGAIN)
- **Recipe:** aloe_v2vox_raw.env = aloe_v2vox.env (22420) except raw A gain, logical-world-64 accumulation on 3 nodes (world 24), no rowcache; 253 steps
- **Steps:** 0 -> 253 (23060 stopped at 6, 23079 restart aborted, 23081 full run from 0)
- **Evals:** norm_word_ce = 2.0372 @0 -> 1.8248 @253
- **Checkpoints / exports:** export-253
- **Local paths:** /data/home/amartyavarma/ppm/aloe-v2vox-raw/export-253
- **HF repos:** [InternalCan/ppm-aloe-v2vox-raw](https://huggingface.co/InternalCan/ppm-aloe-v2vox-raw) *(name match)*
- **Notes:** Recipe reference (job 23081) for si20k-vox-ppm and the w14503 / w15973 re-runs. Backup GCS (gs://.../ckpt-backup/aloe-v2vox-raw/).
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-v2vox-raw/aloe_v2vox_raw.env:1 -> = 22420 env except raw A gain etc.`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-raw/LAUNCHED_JOB:1 -> job 23081, restart from step 0 off slinky-4; 23060 stopped at 6`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-raw/logs/aloev2vraw-23081.log:44 -> init local-11488`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-raw/logs/aloev2vraw-23081.log:166 -> EVAL step 0 2.0372`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-raw/logs/aloev2vraw-23081.log:515 -> EVAL step 253 1.8248`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-raw/logs/aloev2vraw-23081.log:517 -> export-253`
  - `memory:only-demo-aloe-mothers.md:27 -> live demo served aloe-v2vox-raw export-253 before 10-20; '11k endpoint' afterwards (line 30)`

### aloe-v2vox-raw-w14503

**Aloe v2vox raw FT from PPM w-14503**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-v2vox-raw-w14503`

- **Jobs:** 25962, 25963
- **Dates:** 2026-10-20 ~00:5x .. 2026-10-20 02:31 PT
- **Base model:** unknown
- **Init:** /data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-14503 + asrppm-nonorm w-1972
- **Parents (resolved):** ppm32b@14503, asrppm-nonorm@11488
- **Produces:** aloe-v2vox-raw-w14503@253
- **Data:** as aloe-v2vox-raw
- **Recipe:** aloe_v2vox_raw.env (= 23081) with init swapped to w-14503; 253 steps, world 24 (logical 64)
- **Steps:** 0 -> 253
- **Evals:** norm_word_ce = 2.0177 @0 -> 1.8138 @253
- **Checkpoints / exports:** export-42; export-253
- **Local paths:** /data/home/amartyavarma/ppm/aloe-v2vox-raw-w14503/export-253
- **HF repos:** [InternalCan/ppm-aloe-v2vox-raw-w14503](https://huggingface.co/InternalCan/ppm-aloe-v2vox-raw-w14503)
- **Notes:** 25962 killed (exit 143) before training; 25963 the run.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-v2vox-raw-w14503/LAUNCHED_JOB:1 -> job 25963, init w-14503`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-raw-w14503/logs/aloev2vraw-25963.log:167 -> EVAL step 0 2.0177`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-raw-w14503/logs/aloev2vraw-25963.log:516 -> EVAL step 253 1.8138`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-raw-w14503/logs/aloev2vraw-25963.log:518 -> export-253`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-raw-w14503/logs/BACKUP_LOG.txt:1 -> HF InternalCan/ppm-aloe-v2vox-raw-w14503`
  - `memory:only-demo-aloe-mothers.md:25 -> live demo model from 10-20 = w-14503-init FT export-253`

### aloe-v2vox-asr-w15973

**Aloe reference FT 26232 (PPM w-15973 + asrppm-w15973)**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973`

- **Jobs:** 26221, 26232
- **Dates:** 2026-10-20 ~19:0x .. 2026-10-20 19:32 PT
- **Base model:** unknown
- **Init:** /data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-15973 + adapter asrppm-w15973 w-1441
- **Parents (resolved):** ppm32b@15973, asrppm-w15973@15973
- **Produces:** aloe-v2vox-asr-w15973@253
- **Data:** as aloe-v2vox-raw (s2h-m1 none stores, v2 first-chunk plan, raw A)
- **Recipe:** aloe_v2vox_raw.env (23081 recipe) with init w-15973 and the matched w-1441 adapter; 253 steps on 8 nodes (world 64)
- **Steps:** 0 -> 253
- **Evals:** norm_word_ce = 2.0125 @0 -> 1.8075 @253
- **Checkpoints / exports:** export-42; export-253
- **Local paths:** /scratch/amartyavarma/aloe-v2vox-asr-w15973/exports/export-253 (node-local)
- **HF repos:** [InternalCan/ppm-aloe-v2vox-asr-w15973](https://huggingface.co/InternalCan/ppm-aloe-v2vox-asr-w15973)
- **Deployed:** Aloe speech w-15973 site (ppm/aloe-m1vox-live-w15973, front :8925, one model) (paused)
- **Notes:** 26221 aborted (LAUNCHED_JOB.26221-aborted-scratch). Reference for the textB, zoom-cotrain and all samsep FTs (cfg diffs vs 26232).
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973/LAUNCHED_JOB:1 -> job 26232, init w-15973, adapter asrppm-w15973-w-1441`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973/logs/aloev2vasr-26232.log:256 -> EVAL step 0 2.0125`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973/logs/aloev2vasr-26232.log:725 -> EVAL step 253 1.8075`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973/logs/aloev2vasr-26232.log:727 -> export-253`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973/logs/BACKUP_LOG.txt:1 -> HF InternalCan/ppm-aloe-v2vox-asr-w15973`
  - `memory:ppm-vs-811-naming.md:9 -> Aloe reference FTs 26232/26467 used PPM w-15973 + asrppm-w15973`

### aloe-v2vox-asr-w15973-textB

**Aloe textB FT (user speech in, assistant text only)**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973-textB`

- **Jobs:** 26419, 26422
- **Dates:** 2026-10-20 23:25 .. 2026-10-21 01:07 PT
- **Base model:** unknown
- **Init:** /data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-15973 + asrppm-w15973 w-1441
- **Parents (resolved):** ppm32b@15973, asrppm-w15973@15973
- **Produces:** aloe-v2vox-asr-w15973-textB@253
- **Data:** as 26232
- **Recipe:** aloe_textB.env = 26232 env verbatim except ALOE_B_TEXT_ONLY=1 (B text lane only, no B Mimi, no cb0/light loss); 3 nodes
- **Steps:** 0 -> 253 (26419 exit 1 at step-0 pack check)
- **Evals:** norm_word_ce = 2.2137 @0 -> 1.9100 @253
- **Checkpoints / exports:** export-42; export-253
- **Local paths:** /scratch/amartyavarma/aloe-v2vox-asr-w15973-textB/exports/export-253 (node-local)
- **HF repos:** [InternalCan/ppm-aloe-v2vox-asr-w15973-textB](https://huggingface.co/InternalCan/ppm-aloe-v2vox-asr-w15973-textB)
- **Notes:** Deployed as the text-out + SAM CD3 TTS demo in /data/home/amartyavarma/ppm/aloe-textb-tts (demo dir only, no training there).
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973-textB/aloe_textB.env:1 -> user: speech in, text only out; = 26232 except ALOE_B_TEXT_ONLY`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973-textB/logs/aloev2vtextB-26422.log:176 -> EVAL step 0 2.2137`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973-textB/logs/aloev2vtextB-26422.log:526 -> EVAL step 253 1.9100`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973-textB/logs/aloev2vtextB-26422.log:528 -> export-253`
  - `/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973-textB/logs/BACKUP_LOG.txt:1 -> HF InternalCan/ppm-aloe-v2vox-asr-w15973-textB`

### aloe-samsep-vox17127

**Aloe FT vox17127 (no persona) on PPMVOX v3 w-17127**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/vox` · also audited in part_posefts.json

- **Jobs:** 29345
- **Dates:** 2026-10-24 07:24 .. 2026-10-24 07:53 PT
- **Base model:** unknown
- **Init:** PPMVOX v3 w-17127 (/scratch/amartyavarma/aloevox-init/w-17127) with its own trained adapter (ALOE_INIT_VOX=1)
- **Parents (resolved):** ppmvox-v3@17127
- **Produces:** aloe-samsep-vox17127@249
- **Data:** s2h-m1 rows with SAM-separated Mimi codes (same as 28511)
- **Recipe:** vox/run_vox.env: = 28511 except init / adapter / hosts / names; 249 steps; same per-module LRs; world 64
- **Steps:** 0 -> 249
- **Evals:** norm_word_ce = 1.8484 @0 -> 1.8023 @249
- **Checkpoints / exports:** export-vox17127-249; dcp-249 (node-local)
- **Local paths:** /data/home/amartyavarma/ppm/aloe-ft-32-samsep/vox/export-vox17127-249; /data/home/amartyavarma/ppm/aloe-ft-32-samsep/release/aloe-ft-vox17127-nopersona
- **HF repos:** [InternalCan/ppm-aloe-samsep-vox17127](https://huggingface.co/InternalCan/ppm-aloe-samsep-vox17127), [InternalCan/ppm-aloe-samsep-vox17127-final](https://huggingface.co/InternalCan/ppm-aloe-samsep-vox17127-final), [InternalCan/aloe-ft-vox17127-nopersona](https://huggingface.co/InternalCan/aloe-ft-vox17127-nopersona), [InternalCan/ppm-aloe-samsep-vox17127-saves](https://huggingface.co/InternalCan/ppm-aloe-samsep-vox17127-saves) *(name match)*
- **Deployed:** Speech twin of the pose demo (ppm/aloe-m1vox-live server_twin.py, front :8935) (paused); Aloe speech demo 10-24 clone (ppm/aloe-m1vox-live-1024, front :8905) - model 'aloe' (default) (paused); Original Aloe speech live demo (ppm/aloe-m1vox-live, 4-model picker: Brianna / Rosie / Aloe no-persona / Aloe + persona, 10 backends) (down)
- **Notes:** Init of the Brianna and Rosie speaker LoRAs; served in the 10-24 demo picker as 'Aloe FT (no persona)' and in the 10-27 clone (aloe-m1vox-live-1024). \|\| Second independent audit (part_posefts.json): Speech-only. Parent run id 'ppmvox-v3' is a guess for the PPMVOX v3 run whose stop export is w-17127.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-29345.log:8 -> ALOE_INIT_VOX=1`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-29345.log:70 -> QQA_INIT w-17127`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-29345.log:84 -> QQA_STEPS=249`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-29345.log:319 -> EVAL step 0 1.8484`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-29345.log:706 -> EVAL step 249 1.8023`
  - `/data/home/amartyavarma/ppm/control/OPS_PLAYBOOK.md:397 -> diff vs 28511 = init/adapter/hosts/names only`
  - `/data/home/amartyavarma/ppm/control/OPS_PLAYBOOK.md:410 -> DONE, export-vox17127-249 COMPLETE on /data (78.2 GB)`
  - `memory:nopersona-inference-defaults.md:7 -> pushed as private HF InternalCan/aloe-ft-vox17127-nopersona`
  - `memory:only-demo-aloe-mothers.md:19 -> demo hot-swapped to vox17127 export-249 (job 29345) 10-24 07:59 PT`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/vox/run_vox.env:1 -> = run_w64.env (28511) with ONLY init changed: PPMVOX v3 stop export w-17127`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/vox/LAUNCHED_JOB:1 -> job 29345 launched 2026-10-24 07:22:17 PT`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-29345.log:162 -> cfg init /scratch/amartyavarma/aloevox-init/w-17127, 249 steps`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-29345.log:319 -> step-0 word_ce_norm 1.8484`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-29345.log:706 -> step-249 word_ce_norm 1.8023`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/hf_watch-29345-slinky-1.log:94 -> export-249 upload start (repos -final and intermediates named in hf_watch logs)`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/release/aloe-ft-vox17127-nopersona/README.md:11 -> release 'Aloe FT (no persona): vox17127, export-249'`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/release/aloe-ft-vox17127-nopersona/QUICKSTART.md:16 -> HF InternalCan/aloe-ft-vox17127-nopersona`

### aloespk-brianna-A

**Brianna voice FT run A (LoRA r4, replay 0) -> A-10**  
group `aloe-speech-ft` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora`

- **Jobs:** 29536, 29540, 29543, 29545
- **Dates:** 2026-10-24 ~08:50 .. 2026-10-24 09:23 PT
- **Base model:** unknown
- **Init:** Aloe FT vox17127 export-249 (/scratch/amartyavarma/aloespk-init-w8/export-249)
- **Parents (resolved):** aloe-samsep-vox17127@249
- **Produces:** aloespk-brianna-A@10
- **Data:** speaker fFP3835k (Brianna): 12 calls / 19 rows / 2.31 h; train 16 rows, 2 held-out calls; she is always B; voice clip from her other calls
- **Recipe:** run_brianna_A.env: frozen trunk + LoRA r4/a8 (LR 1e-4) on q,k,v,o,gate,up,down; tower/light/audio/proj trained; replay 0; 20 epochs = 40 steps planned, warm-up 2, decay to 0; world 8 on slinky-18
- **Steps:** 0 -> 24 of 40 (stopped; candidate export-10)
- **Evals:** norm_word_ce = standard Aloe eval_speech 1.8023 @0 -> 1.8138 @10 -> 1.8323 @24 (speaker held-out eval not logged in this run)
- **Checkpoints / exports:** export-10 (LoRA delta); export-20
- **Local paths:** /scratch/amartyavarma/aloespk-lora/exports-fFP3835k-A/export-10 (slinky-18); /scratch/amartyavarma/aloe-demo-brianna-A10/merged-A10 (slinky-10, merged onto vox17127)
- **HF repos:** [InternalCan/ppm-aloespk-lora-fFP3835k-A](https://huggingface.co/InternalCan/ppm-aloespk-lora-fFP3835k-A), [InternalCan/ppm-aloespk-lora-fFP3835k-A-final](https://huggingface.co/InternalCan/ppm-aloespk-lora-fFP3835k-A-final), [InternalCan/ppm-aloespk-lora-fFP3835k-A-saves](https://huggingface.co/InternalCan/ppm-aloespk-lora-fFP3835k-A-saves) *(name match)*
- **Deployed:** Aloe speech demo 10-24 clone - model 'brianna' (removed from picker) (down)
- **Notes:** 29536/29540/29543 were failed starts (no training past step 0). The standard eval rises as it specialises; no held-out Brianna CE is logged for A.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/run_brianna_A.env:1 -> run A: replay 0, 20 epochs (40 steps)`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/run_brianna_A.env:39 -> ALOE_LORA_RANK=4`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/STATUS.txt:13 -> Brianna plan 12 calls / 19 rows / 2.31 h`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/STATUS.txt:17 -> A stopped at 24, candidate export-10 on HF`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/logs/aloespklora-29545.log:527 -> EVAL step 10 1.8138`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/logs/aloespklora-29545.log:528 -> export-10`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/logs/aloespklora-29545.log:550 -> EVAL step 24 1.8323`
  - `/data/home/amartyavarma/ppm/aloe-m1vox-live/a16861/RELAUNCH_ALL.md:7 -> Brianna merged-A10 path, re-make via vox17127/brianna_local_merge.sbatch`
  - `memory:only-demo-aloe-mothers.md:43 -> demo served Brianna voice FT A-10 (LoRA r4 merged onto vox17127)`

### aloespk-brianna-B

**Brianna voice FT run B (replay 0.5)**  
group `aloe-speech-ft` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora`

- **Jobs:** 29549
- **Dates:** 2026-10-24 ~09:27 .. 2026-10-24 09:45 PT
- **Base model:** unknown
- **Init:** Aloe FT vox17127 export-249
- **Parents (resolved):** aloe-samsep-vox17127@249
- **Produces:** aloespk-brianna-B@40
- **Data:** Brianna rows + 50% replay of standard Aloe micros
- **Recipe:** run_brianna_B.env (replay 0.5), LoRA r4 as A
- **Steps:** 0 -> 56 (past best; stopped)
- **Evals:** norm_word_ce = speaker held-out eval_spk_speech best 1.7873 @24 -> 1.8809 @56; standard 1.8023 @0, 1.8119 @20, 1.8302 @40
- **Checkpoints / exports:** export-40
- **Local paths:** /scratch/amartyavarma/aloespk-lora/exports-fFP3835k-B/export-40
- **HF repos:** [InternalCan/ppm-aloespk-lora-fFP3835k-B](https://huggingface.co/InternalCan/ppm-aloespk-lora-fFP3835k-B), [InternalCan/ppm-aloespk-lora-fFP3835k-B-final](https://huggingface.co/InternalCan/ppm-aloespk-lora-fFP3835k-B-final), [InternalCan/ppm-aloespk-lora-fFP3835k-B-saves](https://huggingface.co/InternalCan/ppm-aloespk-lora-fFP3835k-B-saves) *(name match)*
- **Notes:** Overfit after ~24 steps; B2 is the identical re-run stopped at 24.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/logs/aloespklora-29549.log:557 -> EVAL step 24 eval_spk_speech 1.7873`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/logs/aloespklora-29549.log:601 -> EVAL step 56 eval_spk_speech 1.8809`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/logs/aloespklora-29549.log:581 -> export-40`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/STATUS.txt:17 -> B stopped at 56 (past best)`

### aloespk-brianna-B2

**Brianna voice FT run B2 (replay 0.5, stop 24)**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora`

- **Jobs:** 29558
- **Dates:** 2026-10-24 ~09:49 .. 2026-10-24 10:00 PT
- **Base model:** unknown
- **Init:** Aloe FT vox17127 export-249
- **Parents (resolved):** aloe-samsep-vox17127@249
- **Produces:** aloespk-brianna-B2@16, aloespk-brianna-B2@24
- **Data:** Brianna rows + 50% replay
- **Recipe:** run_brianna_B2.env = B, stop at 24
- **Steps:** 0 -> 24
- **Evals:** norm_word_ce = standard eval_speech 1.8094 @16 (cb0 best), 1.8151 @24 (light best)
- **Checkpoints / exports:** export-16; export-24
- **Local paths:** /scratch/amartyavarma/aloespk-lora/exports-fFP3835k-B2 (slinky-18)
- **HF repos:** [InternalCan/ppm-aloespk-lora-fFP3835k-B2](https://huggingface.co/InternalCan/ppm-aloespk-lora-fFP3835k-B2), [InternalCan/ppm-aloespk-lora-fFP3835k-B2-final](https://huggingface.co/InternalCan/ppm-aloespk-lora-fFP3835k-B2-final) *(not in model listing: dataset/bucket or deleted)*
- **Notes:** Not the deployed Brianna model (A-10 was).
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/logs/aloespklora-29558.log:532 -> EVAL step 16 1.8094`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/logs/aloespklora-29558.log:543 -> EVAL step 24 1.8151`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/STATUS.txt:17 -> B2 export-16 cb0 best (std 1.8094) + export-24 light best`

### aloespk-rosie

**Rosie voice FT (LoRA r4, 2 epochs) -> merged-48**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/rosie`

- **Jobs:** 29575, 29577
- **Dates:** 2026-10-24 18:29 .. 2026-10-24 18:45 PT
- **Base model:** unknown
- **Init:** Aloe FT vox17127 export-249 (/scratch/amartyavarma/aloerosie-init/export-249)
- **Parents (resolved):** aloe-samsep-vox17127@249
- **Produces:** aloespk-rosie@48
- **Data:** speaker xyvR1rjH (Rosie), always B, held-out calls; voice clip from another call
- **Recipe:** run_rosie.env = run_brianna_A.env recipe (replay 0, LoRA r4/a8 LR 1e-4, tower/dec/audio/proj trained, adapter + vgain frozen); exactly 2 epochs = 48 steps; world 6 on slinky-4
- **Steps:** 0 -> 48
- **Evals:** norm_word_ce = speaker held-out eval_spk_speech 1.6230 @6 -> best 1.5848 @42; standard eval_speech 1.8023 @0 -> 1.8292 @48
- **Checkpoints / exports:** export-24; export-48 (LoRA deltas)
- **Local paths:** /scratch/amartyavarma/aloespk-lora/exports-xyvR1rjH/export-48; /scratch/amartyavarma/aloe-demo-rosie-48/merged-48 (slinky-18)
- **HF repos:** [InternalCan/ppm-aloespk-lora-xyvR1rjH](https://huggingface.co/InternalCan/ppm-aloespk-lora-xyvR1rjH), [InternalCan/ppm-aloespk-lora-xyvR1rjH-final](https://huggingface.co/InternalCan/ppm-aloespk-lora-xyvR1rjH-final) *(not in model listing: dataset/bucket or deleted)*, [InternalCan/ppm-aloespk-lora-xyvR1rjH-saves](https://huggingface.co/InternalCan/ppm-aloespk-lora-xyvR1rjH-saves) *(name match)*
- **Deployed:** Aloe speech demo 10-24 clone - model 'rosie' (paused)
- **Notes:** 29575 terminated before training. Env header says 'user go 10-25' but the job log is stamped 10-24 18:29 PT.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/rosie/run_rosie.env:1 -> Rosie LoRA FT recipe, 2 epochs, world 6`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/rosie/run_rosie.env:23 -> QQA_STEPS=48`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/logs/aloespklora-29577.log:499 -> EVAL step 0 1.8023`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/logs/aloespklora-29577.log:554 -> EVAL step 42 eval_spk_speech 1.5848`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/logs/aloespklora-29577.log:561 -> EVAL step 48 eval_speech 1.8292`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/logs/aloespklora-29577.log:565 -> export-48`
  - `/data/home/amartyavarma/ppm/aloe-m1vox-live/a16861/RELAUNCH_ALL.md:8 -> Rosie merged-48 path (re-make: rosie/xfer_delta.sh + rosie/merge.sbatch)`

### qqa-ft

**PPM quick-QA speech-duplex FT (run C recipe)**  
group `other` · status **finished** · dir `/data/home/amartyavarma/ppm/qqa-ft`

- **Jobs:** 21817, 21823, 21832
- **Dates:** 2026-10-15 23:20 .. 2026-10-16 ~01:xx PT
- **Base model:** unknown
- **Init:** PPM node-local dcp-11263 -> /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11263 (fresh AdamW)
- **Parents (resolved):** ppm32b@11263
- **Produces:** qqa-ft@223
- **Data:** run C's quick-QA speech store: 95,192 conversations, same order/packs, 96 held-out eval conversations; A = user lane + Mimi, B = assistant lane + Mimi
- **Recipe:** qqa.env: 223 packed steps (1 epoch), trunk 3e-6 (no LoRA), decoder 1.5e-5 peak, decay to 0; world 24 on slinky-7/15/18; 21832 = scaling re-run with exports 14/28/59/117/223
- **Steps:** 0 -> 223 (21817 crashed at 82; 21823 full; 21832 rerun)
- **Evals:** norm_word_ce = B normalised word CE 1.9862 @0 -> 1.4268 @223 (21823); 1.4318 @223 (21832); other = QA acc_greedy 0.2861 (w223) vs init 0.0962; scaling rerun 0.2981 @95,192 conversations
- **Checkpoints / exports:** export-223; scale/export-14,28,59,117,223
- **Local paths:** /data/home/amartyavarma/ppm/qqa-ft/export-223; /data/home/amartyavarma/ppm/qqa-ft/scale/export-223
- **HF repos:** [InternalCan/ppm-qqa-w223](https://huggingface.co/InternalCan/ppm-qqa-w223)
- **Notes:** Speech-duplex format (user words + Mimi on A), no Voxtral. Its qqa.env is the recipe ancestor of every PPM Aloe FT (memory aloe-handover-only.md:15).
- **Citations:**
  - `/data/home/amartyavarma/ppm/qqa-ft/STATUS.md:5 -> job 21817, init PPM step 11263`
  - `/data/home/amartyavarma/ppm/qqa-ft/STATUS.md:15 -> local-11263 copy`
  - `/data/home/amartyavarma/ppm/qqa-ft/STATUS.md:21 -> 95,192 conversations, 96 held-out`
  - `/data/home/amartyavarma/ppm/qqa-ft/STATUS.md:37 -> trunk 3e-6`
  - `/data/home/amartyavarma/ppm/qqa-ft/STATUS.md:62 -> 21817 crashed at step 82`
  - `/data/home/amartyavarma/ppm/qqa-ft/STATUS.md:68 -> 21823 evals 1.9862 -> 1.4268 @223`
  - `/data/home/amartyavarma/ppm/qqa-ft/STATUS.md:70 -> export-223 on /data (GCS backup)`
  - `/data/home/amartyavarma/ppm/qqa-ft/STATUS.md:78 -> QA acc_greedy 0.2861`
  - `/data/home/amartyavarma/ppm/qqa-ft/STATUS.md:85 -> scaling rerun 21832`
  - `/data/home/amartyavarma/ppm/qqa-ft/STATUS.md:91 -> 21832 223: 1.4318`
  - `memory:speech-duplex-means-stage3.md:11 -> 'speech duplex FT' = stage-3 format (user text + Mimi)`

### ttconvo-ft

**ttconvo-ft (speech-only FT of 8,1,1 dcp-17012 on tt-v3)**  
group `other` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/ttconvo-ft`

- **Jobs:** 26510
- **Dates:** 2026-10-21 04:17 .. 2026-10-21 05:02 PT
- **Base model:** unknown
- **Init:** 8,1,1 dcp-17012 (/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17012)
- **Parents (resolved):** ppm811-epoch2@17012
- **Produces:** ttconvo-ft@285
- **Data:** tt-v3 TTS conversations, only the on-disk 26% subset (37.9k of 144k rows); voice clip p 0.8, persona p 0.2; eval store tt-v3-rows/evalstore
- **Recipe:** ft.env = prod_811.env + [FT] block: GA speech only, trunk 3e-6, spec 6e-5, proj 3e-5, audio 6e-5, tower 1.5e-5, dec 1.5e-5; 296 steps, warmup 10, WSD decay to 0; world 128
- **Steps:** 0 -> 285 of 296 (STOP_AFTER_SAVE)
- **Evals:** norm_word_ce = eval_ttv3/word_ce_norm 1.9703 @0 -> 1.9565 @275 (last eval); other = eval_speech/word_ce_norm 1.9486 @275
- **Checkpoints / exports:** export-47; export-94; export-195; export-285; dcp-285 (node-local)
- **Local paths:** /scratch/amartyavarma/ttconvo-ft/ckpt/ttconvo-ft/dcp-285
- **HF repos:** [InternalCan/ppm-ttconvo-ft](https://huggingface.co/InternalCan/ppm-ttconvo-ft)
- **Notes:** 8,1,1 lineage (not pure PPM). Partner words+Mimi on A (PPM speech rows), no Voxtral.
- **Citations:**
  - `/data/home/amartyavarma/ppm/ttconvo-ft/ft.env:245 -> speech-only FT of PPM 8,1,1 dcp-17012 on tt-v3`
  - `/data/home/amartyavarma/ppm/ttconvo-ft/ft.env:251 -> PPM_LR_TRUNK=3e-6`
  - `/data/home/amartyavarma/ppm/ttconvo-ft/ft.env:266 -> PPM_MAX_STEPS=296`
  - `/data/home/amartyavarma/ppm/ttconvo-ft/ft.env:291 -> PPM_FT_INIT dcp-17012`
  - `/data/home/amartyavarma/ppm/ttconvo-ft/PAGE_STATUS.txt:1 -> only 26% of tt-v3 train on disk`
  - `/data/home/amartyavarma/ppm/ttconvo-ft/logs/ft-26510.log:146 -> EVAL step 0 eval_ttv3 1.9703`
  - `/data/home/amartyavarma/ppm/ttconvo-ft/logs/ft-26510.log:827 -> EVAL step 275 eval_ttv3 1.9565`
  - `/data/home/amartyavarma/ppm/ttconvo-ft/logs/ft-26510.log:839 -> STOP_AFTER_SAVE at 285, dcp-285`
  - `/data/home/amartyavarma/ppm/ttconvo-ft/hf_backup/HF_BACKUP-export-285.json:2 -> repo InternalCan/ppm-ttconvo-ft`

### si20k-vox-ppm

**si20k-vox-ppm (PPM w-12829 + ASR w-1972, speech in/out, + podcast 1:1)**  
group `si20k` · status **finished** · dir `/data/home/amartyavarma/ppm/si20k-vox-ppm`

- **Jobs:** 24857, 24916, 24920
- **Dates:** 2026-10-18 21:33 .. 2026-10-18 23:41 PT
- **Base model:** unknown
- **Init:** PPM dcp-12829 -> /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-12829 + ASR adapter asrppm-nonorm w-1972
- **Parents (resolved):** ppm32b@12829, asrppm-nonorm@11488
- **Produces:** si20k-vox-ppm@198
- **Data:** spoken-instruct 20k tt-v3 TTS set (si20k-vox stores, 12,625 train / 192 eval strict keep_spans rows), A = Voxtral latents of raw user audio (qqa-corrupt-v2 + gauss:6 mic gain), B = words + Mimi; co-trained 1:1 with unseen podcast rows
- **Recipe:** code/train_si20k_vox.py generated from 23081's trainer; si20k_vox_ppm.env = 23081 env + data/podcast blocks; 198 steps, warmup 8, decay to 0; trunk 3e-6 ... vadapter 5e-4; 64 rows/step
- **Steps:** 0 -> 198
- **Evals:** norm_word_ce = si20k eval_speech/word_ce_norm 2.213 @0 -> 1.4567 @198; other = podcast eval 2.076 -> 1.985; low-mic eval 1.459
- **Checkpoints / exports:** export-33; export-66; export-99; export-132; export-165; export-198
- **Local paths:** /data/home/amartyavarma/ppm/si20k-vox-ppm/export-198
- **HF repos:** [InternalCan/si20k-vox-ppm](https://huggingface.co/InternalCan/si20k-vox-ppm)
- **Notes:** 24857/24916 were earlier launches (stopped/relaunched for CPU starvation, README:130-134); 24920 is the full run. Deployment dir si20k-ppm-live (not a training run).
- **Citations:**
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm/README.md:1 -> si20k-vox-ppm definition`
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm/README.md:7 -> recipe copied from aloe-v2vox-raw job 23081`
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm/README.md:14 -> Init ppm-v1 dcp-12829`
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm/README.md:17 -> adapter ASR w-1972`
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm/logs/si20kvox-ppm-24920.log:43 -> QQA_INIT local-12829`
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm/logs/si20kvox-ppm-24920.log:892 -> EVAL step 198 eval_speech/word_ce_norm 1.4567`
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm/README.md:138 -> HF InternalCan/si20k-vox-ppm/export-198-dcp; 2.213 -> 1.457`
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm/README.md:139 -> podcast eval 2.076 -> 1.985`
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm/README.md:141 -> loaders should use THETA_FP32.json`
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm11488/README.md:7 -> 12,625 / 192 strict rows, plan 01cacb274286ddb6`

### si20k-vox-intl

**si20k-vox-intl (INTL 14B w-6242 + asr14b w-480, speech in/out)**  
group `si20k` · status **finished** · dir `/data/home/amartyavarma/ppm/si20k-vox-intl`

- **Jobs:** 24858, 24879, 24903, 24917, 24921
- **Dates:** 2026-10-18 21:33 .. 2026-10-18 23:34 PT
- **Base model:** unknown
- **Init:** INTL 14B next-arch w-6242 (/data/home/amartyavarma/ppm/next-arch/ckpt-weka/ppm-next/w-6242) + 14B ASR adapter asr14b w-480 (sha b6ca8779f8d0b1bd)
- **Parents (resolved):** intl14b@6242, asr14b@480
- **Produces:** si20k-vox-intl@198
- **Data:** as si20k-vox-ppm (si20k tt-v3 strict rows + unseen podcast rows 1:1)
- **Recipe:** trainer generated from aloe-nextvox-6242 job 24193; 198 steps, warmup 8, decay to 0; 64 rows/step
- **Steps:** 0 -> 198
- **Evals:** norm_word_ce = si20k eval_speech/word_ce_norm 2.450 @0 -> 1.5449 @198; other = podcast eval 2.144 -> 1.992
- **Checkpoints / exports:** export-33..165; export-198
- **Local paths:** /data/home/amartyavarma/ppm/si20k-vox-intl/export-198
- **HF repos:** [InternalCan/si20k-vox-intl](https://huggingface.co/InternalCan/si20k-vox-intl)
- **Notes:** asr14b@6242 = the 14B adapter w-480 trained on w-6242 (another agent's group; id guessed).
- **Citations:**
  - `/data/home/amartyavarma/ppm/si20k-vox-intl/README.md:7 -> recipe copied from aloe-nextvox-6242 job 24193`
  - `/data/home/amartyavarma/ppm/si20k-vox-intl/README.md:13 -> Init w-6242`
  - `/data/home/amartyavarma/ppm/si20k-vox-intl/README.md:14 -> adapter asr14b w-480`
  - `/data/home/amartyavarma/ppm/si20k-vox-intl/logs/si20kvox-intl-24921.log:57 -> QQA_INIT w-6242`
  - `/data/home/amartyavarma/ppm/si20k-vox-intl/logs/si20kvox-intl-24921.log:88 -> VOX_ADAPTER adapter-asr14b-w480`
  - `/data/home/amartyavarma/ppm/si20k-vox-intl/logs/si20kvox-intl-24921.log:907 -> EVAL step 198 1.5449`
  - `/data/home/amartyavarma/ppm/si20k-vox-intl/README.md:136 -> HF InternalCan/si20k-vox-intl/export-198-dcp; 2.450 -> 1.545`
  - `/data/home/amartyavarma/ppm/si20k-vox-intl/LAUNCHED_JOB:1 -> 24858 / 24879 launches`

### si20k-txt-ppm

**si20k-txt-ppm (speech in, text out; PPM 12829)**  
group `si20k` · status **finished** · dir `/data/home/amartyavarma/ppm/si20k-txt-ppm`

- **Jobs:** 25045, 25049
- **Dates:** 2026-10-19 00:24 .. 2026-10-19 01:41 PT
- **Base model:** unknown
- **Init:** PPM dcp-12829 (local-12829) + ASR w-1972
- **Parents (resolved):** ppm32b@12829, asrppm-nonorm@11488
- **Produces:** si20k-txt-ppm@198
- **Data:** si20k strict rows (12,625 / 192) + podcast 1:1; B text only (no audio summed in or predicted)
- **Recipe:** si20k-vox recipe with B audio removed; 198 steps
- **Steps:** 0 -> 198 (25045 stopped at step 1 for bookkeeping, relaunched)
- **Evals:** norm_word_ce = si20k eval 2.376 @0 -> 1.4818 @198 (no user audio 1.9752)
- **Checkpoints / exports:** export-33..165; export-198
- **Local paths:** /data/home/amartyavarma/ppm/si20k-txt-ppm/export-198
- **HF repos:** [InternalCan/si20k-txt-ppm](https://huggingface.co/InternalCan/si20k-txt-ppm)
- **Notes:** Deployment dir si20k-txt-ppm-live (not training).
- **Citations:**
  - `/data/home/amartyavarma/ppm/si20k-txt-ppm/README.md:3 -> user: streaming text in/out, no audio back into the backbone`
  - `/data/home/amartyavarma/ppm/si20k-txt-ppm/README.md:14 -> init dcp-12829, adapter w-1972`
  - `/data/home/amartyavarma/ppm/si20k-txt-ppm/README.md:29 -> 25045 stopped at step 1, relaunched`
  - `/data/home/amartyavarma/ppm/si20k-txt-ppm/logs/si20ktxt-ppm-25049.log:795 -> EVAL step 198 1.4818`
  - `/data/home/amartyavarma/ppm/si20k-txt-ppm/logs/si20ktxt-ppm-25049.log:796 -> no-A 1.9752`
  - `/data/home/amartyavarma/ppm/si20k-txt-ppm/README.md:34 -> HF InternalCan/si20k-txt-ppm/export-198-dcp; 2.376 -> 1.482`

### si20k-txt-intl

**si20k-txt-intl (speech in, text out; INTL 14B w-6242)**  
group `si20k` · status **finished** · dir `/data/home/amartyavarma/ppm/si20k-txt-intl`

- **Jobs:** 25046, 25050
- **Dates:** 2026-10-19 00:24 .. 2026-10-19 01:41 PT
- **Base model:** unknown
- **Init:** INTL w-6242 + asr14b w-480
- **Parents (resolved):** intl14b@6242, asr14b@480
- **Produces:** si20k-txt-intl@198
- **Data:** as si20k-txt-ppm
- **Recipe:** si20k-vox-intl recipe with B audio removed; 198 steps
- **Steps:** 0 -> 198
- **Evals:** norm_word_ce = si20k eval 2.615 @0 -> 1.5639 @198 (no user audio 2.1303)
- **Checkpoints / exports:** export-33..165; export-198
- **Local paths:** /data/home/amartyavarma/ppm/si20k-txt-intl/export-198
- **HF repos:** [InternalCan/si20k-txt-intl](https://huggingface.co/InternalCan/si20k-txt-intl)
- **Notes:** Deployment dir si20k-txt-intl-live.
- **Citations:**
  - `/data/home/amartyavarma/ppm/si20k-txt-intl/README.md:14 -> init w-6242 + asr14b w-480`
  - `/data/home/amartyavarma/ppm/si20k-txt-intl/logs/si20ktxt-intl-25050.log:57 -> QQA_INIT w-6242`
  - `/data/home/amartyavarma/ppm/si20k-txt-intl/logs/si20ktxt-intl-25050.log:820 -> EVAL step 198 1.5639`
  - `/data/home/amartyavarma/ppm/si20k-txt-intl/README.md:33 -> HF InternalCan/si20k-txt-intl/export-198-dcp; 2.615 -> 1.564`

### si20k-tin-ppm

**si20k-tin-ppm (text in, text + speech out; PPM 12829)**  
group `si20k` · status **finished** · dir `/data/home/amartyavarma/ppm/si20k-tin-ppm`

- **Jobs:** 25173, 25178
- **Dates:** 2026-10-19 02:16 .. 2026-10-19 03:15 PT
- **Base model:** unknown
- **Init:** PPM dcp-12829 (local-12829); no Voxtral / adapter
- **Parents (resolved):** ppm32b@12829
- **Produces:** si20k-tin-ppm@198
- **Data:** si20k strict rows with A = user words at TTS timing (no A audio) + podcast 1:1
- **Recipe:** code/train_si20k_tin.py (generated); 198 steps, warmup 8, decay to 0; B = words + Mimi + decoder K1
- **Steps:** 0 -> 198 (25173 crashed before training)
- **Evals:** norm_word_ce = si20k eval 1.686 @0 -> 1.4148 @198 (A words blanked 1.9878)
- **Checkpoints / exports:** export-198
- **Local paths:** /data/home/amartyavarma/ppm/si20k-tin-ppm/export-198
- **HF repos:** [InternalCan/si20k-tin-ppm](https://huggingface.co/InternalCan/si20k-tin-ppm)
- **Notes:** Deployment dirs si20k-tin-ppm-live / si20k-tin-live.
- **Citations:**
  - `/data/home/amartyavarma/ppm/si20k-tin-ppm/README.md:3 -> user: text in, text and speech out`
  - `/data/home/amartyavarma/ppm/si20k-tin-ppm/README.md:16 -> init dcp-12829, 198 steps`
  - `/data/home/amartyavarma/ppm/si20k-tin-ppm/README.md:26 -> step 0 1.686`
  - `/data/home/amartyavarma/ppm/si20k-tin-ppm/logs/si20ktin-ppm-25178.log:668 -> EVAL step 198 1.4148`
  - `/data/home/amartyavarma/ppm/si20k-tin-ppm/logs/si20ktin-ppm-25178.log:669 -> no-A 1.9878`
  - `/data/home/amartyavarma/ppm/si20k-tin-ppm/README.md:27 -> FINAL EXPORT export-198`

### si20k-tin-intl

**si20k-tin-intl (text in, text + speech out; INTL 14B w-6242)**  
group `si20k` · status **finished** · dir `/data/home/amartyavarma/ppm/si20k-tin-intl`

- **Jobs:** 25174
- **Dates:** 2026-10-19 02:16 .. 2026-10-19 03:18 PT
- **Base model:** unknown
- **Init:** INTL w-6242; no adapter
- **Parents (resolved):** intl14b@6242
- **Produces:** si20k-tin-intl@198
- **Data:** as si20k-tin-ppm
- **Recipe:** tin trainer on the 14B; 198 steps
- **Steps:** 0 -> 198
- **Evals:** norm_word_ce = si20k eval 1.828 @0 -> 1.4860 @198 (A words blanked 2.0708)
- **Checkpoints / exports:** export-198
- **Local paths:** /data/home/amartyavarma/ppm/si20k-tin-intl/export-198
- **HF repos:** [InternalCan/si20k-tin-intl](https://huggingface.co/InternalCan/si20k-tin-intl)
- **Notes:** Deployment dir si20k-tin-intl-live.
- **Citations:**
  - `/data/home/amartyavarma/ppm/si20k-tin-intl/logs/si20ktin-intl-25174.log:55 -> QQA_INIT w-6242`
  - `/data/home/amartyavarma/ppm/si20k-tin-intl/README.md:24 -> step 0 1.828`
  - `/data/home/amartyavarma/ppm/si20k-tin-intl/logs/si20ktin-intl-25174.log:688 -> EVAL step 198 1.4860`
  - `/data/home/amartyavarma/ppm/si20k-tin-intl/README.md:25 -> FINAL EXPORT export-198`

### si20k-tdx-intl

**si20k-tdx-intl (14B text duplex, no audio)**  
group `si20k` · status **finished** · dir `/data/home/amartyavarma/ppm/si20k-tdx-intl`

- **Jobs:** 25182, 25248
- **Dates:** 2026-10-19 02:36 .. 2026-10-19 03:59 PT
- **Base model:** unknown
- **Init:** INTL w-6242
- **Parents (resolved):** intl14b@6242
- **Produces:** si20k-tdx-intl@198
- **Data:** si20k rows, user words at TTS timing, assistant text only, no audio anywhere
- **Recipe:** tdx trainer (diff vs txt trainer in tests/); same schedule as si20k-vox-intl (198 steps)
- **Steps:** 0 -> 198
- **Evals:** norm_word_ce = si20k eval 1.5062 @198 (A words blanked 2.1147)
- **Checkpoints / exports:** export-198
- **Local paths:** /data/home/amartyavarma/ppm/si20k-tdx-intl/export-198
- **HF repos:** [InternalCan/si20k-tdx-intl](https://huggingface.co/InternalCan/si20k-tdx-intl)
- **Notes:** 25182 = first chained submission; 25248 the run. Step-0 value not extracted.
- **Citations:**
  - `/data/home/amartyavarma/ppm/si20k-tdx-intl/README.md:3 -> user: 14B only, text duplex, no audio`
  - `/data/home/amartyavarma/ppm/si20k-tdx-intl/logs/si20ktdx-intl-25248.log:55 -> QQA_INIT w-6242`
  - `/data/home/amartyavarma/ppm/si20k-tdx-intl/logs/si20ktdx-intl-25248.log:733 -> EVAL step 198 1.5062`
  - `/data/home/amartyavarma/ppm/si20k-tdx-intl/README.md:21 -> FINAL EXPORT export-198`
  - `/data/home/amartyavarma/ppm/si20k-tdx-intl/README.md:13 -> HF InternalCan/si20k-tdx-intl`

### si20k-vox-ppm11488

**si20k-vox-ppm11488 (speech in/out from PPM 11488, matched to ASR w-1972)**  
group `si20k` · status **failed** · dir `/data/home/amartyavarma/ppm/si20k-vox-ppm11488`

- **Jobs:** 25245
- **Dates:** 2026-10-19 03:33 .. 2026-10-19 04:36 PT
- **Base model:** unknown
- **Init:** /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 + ASR w-1972
- **Parents (resolved):** ppm32b@11488, asrppm-nonorm@11488
- **Produces:** si20k-vox-ppm11488@165
- **Data:** as si20k-vox-ppm (12,625 / 192 rows, plan 01cacb274286ddb6, same podcast rows)
- **Recipe:** = si20k-vox-ppm (j24920) with only QQA_INIT changed
- **Steps:** 0 -> 198 trained; final export lost (scratch full) -> last weights export-165
- **Evals:** norm_word_ce = si20k eval 2.221 @0 -> 1.4648 @198 (weights lost); 1.4677 @165 (last saved)
- **Checkpoints / exports:** export-165
- **Local paths:** unknown
- **HF repos:** [InternalCan/si20k-vox-ppm11488](https://huggingface.co/InternalCan/si20k-vox-ppm11488)
- **Notes:** Training completed but the step-198 weights do not exist; export-165 is the last artifact.
- **Citations:**
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm11488/README.md:6 -> recipe identical to j24920`
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm11488/README.md:9 -> only change QQA_INIT local-11488`
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm11488/README.md:19 -> step 0 2.221`
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm11488/logs/si20kvox11488-25245.log:746 -> EVAL step 198 1.4648`
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm11488/logs/si20kvox11488-25245.log:708 -> EVAL step 165 1.4677`
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm11488/README.md:22 -> crash at final export, step-198 weights lost`
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm11488/README.md:23 -> export-165 HF-verified InternalCan/si20k-vox-ppm11488/export-165`

### si20k-vox-ppm-nocorrupt

**si20k-vox-ppm-nocorrupt (no audio corruption diagnostic)**  
group `si20k` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/si20k-vox-ppm-nocorrupt`

- **Jobs:** 25282, 25283, 25293, 25312
- **Dates:** 2026-10-19 04:23 .. 2026-10-19 05:17 PT
- **Base model:** unknown
- **Init:** PPM dcp-12829 (local-12829) + ASR w-1972
- **Parents (resolved):** ppm32b@12829, asrppm-nonorm@11488
- **Produces:** si20k-vox-ppm-nocorrupt@153
- **Data:** as si20k-vox-ppm, SI20K_A_CORRUPT=none (clean user audio and podcast A audio)
- **Recipe:** = j24920 except corruption off; 198 steps planned
- **Steps:** 0 -> 153 (stopped by the user)
- **Evals:** norm_word_ce = si20k eval 2.2134 @0 -> 1.4620 @150 (last eval)
- **Checkpoints / exports:** export-33..132; export-198-stop153
- **Local paths:** /data/home/amartyavarma/ppm/si20k-vox-ppm-nocorrupt/export-198-stop153
- **HF repos:** [InternalCan/si20k-vox-ppm-nocorrupt](https://huggingface.co/InternalCan/si20k-vox-ppm-nocorrupt)
- **Notes:** 25282 = chain job; 25283 / 25293 aborted at preflight.
- **Citations:**
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm-nocorrupt/README.md:3 -> recipe identical to j24920 except SI20K_A_CORRUPT=none`
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm-nocorrupt/README.md:16 -> step 0 2.2134`
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm-nocorrupt/README.md:17 -> stopped by the user at 153, export-198-stop153`
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm-nocorrupt/logs/si20kvoxnoc-25312.log:692 -> EVAL step 150 1.4620`
  - `/data/home/amartyavarma/ppm/si20k-vox-ppm-nocorrupt/hf_backup/HF_BACKUP-export-132.json:1 -> HF backups (InternalCan/si20k-vox-ppm-nocorrupt)`

### xfer-tt-ss

**xfer-tt-ss (transfer exp A: text-text 0-98, then speech-speech)**  
group `si20k` · status **failed** · dir `/data/home/amartyavarma/ppm/xfer-tt-ss`

- **Jobs:** 25393
- **Dates:** 2026-10-19 05:50 .. 2026-10-19 07:13 PT
- **Base model:** unknown
- **Init:** PPM dcp-12829 (local-12829) + ASR w-1972
- **Parents (resolved):** ppm32b@12829, asrppm-nonorm@11488
- **Produces:** xfer-tt-ss@99
- **Data:** si20k rows: text-text modality for steps 0-98, speech-in/speech-out 99-197; podcast 1:1
- **Recipe:** xfer_tt_ss.env = si20k-vox-ppm (j24920) verbatim except dirs / stores; code/train_si20k_xfer.py; 198 steps
- **Steps:** 0 -> 198 trained; ended without the final export
- **Evals:** norm_word_ce = si20k speech-speech eval 2.2131 @0 -> 1.4707 @198 (no user input 1.9932)
- **Checkpoints / exports:** export-99
- **Local paths:** unknown
- **HF repos:** [InternalCan/xfer-tt-ss](https://huggingface.co/InternalCan/xfer-tt-ss)
- **Notes:** Step-198 weights not exported; only export-99 (end of text-text phase) survives. /data/home/amartyavarma/ppm/xfer is a data-transfer utility dir, not a run.
- **Citations:**
  - `/data/home/amartyavarma/ppm/xfer-tt-ss/xfer_tt_ss.env:1 -> transfer experiment A, PPM dcp-12829 + ASR w-1972`
  - `/data/home/amartyavarma/ppm/xfer-tt-ss/logs/xferttss-25393.log:43 -> QQA_INIT local-12829`
  - `/data/home/amartyavarma/ppm/xfer-tt-ss/logs/xferttss-25393.log:366 -> EVAL step 0 2.2131`
  - `/data/home/amartyavarma/ppm/xfer-tt-ss/logs/xferttss-25393.log:804 -> EVAL step 198 1.4707`
  - `/data/home/amartyavarma/ppm/xfer-tt-ss/FT_ENDED_NO_FINAL:1 -> ended without the final export`
  - `/data/home/amartyavarma/ppm/xfer-tt-ss/README.md:3 -> export-99 HF-verified InternalCan/xfer-tt-ss/export-99`

### xfer-none-ss

**xfer-none-ss (transfer control B: no si20k 0-98, then speech-speech)**  
group `si20k` · status **finished** · dir `/data/home/amartyavarma/ppm/xfer-none-ss`

- **Jobs:** 25395, 25405, 25411
- **Dates:** 2026-10-19 05:52 .. 2026-10-19 07:20 PT
- **Base model:** unknown
- **Init:** PPM dcp-12829 (local-12829) + ASR w-1972
- **Parents (resolved):** ppm32b@12829, asrppm-nonorm@11488
- **Produces:** xfer-none-ss@198
- **Data:** podcast rows only for steps 0-98 (si20k slot zero-weight), then si20k speech-in/speech-out 99-197
- **Recipe:** code/train_si20k_xfer.py, XFER_P1=none, XFER_SWITCH=99; 198 steps, warmup 8, decay to 0
- **Steps:** 0 -> 198
- **Evals:** norm_word_ce = si20k speech-speech eval 1.5410 @198 (no user input 2.0489)
- **Checkpoints / exports:** export-198
- **Local paths:** /data/home/amartyavarma/ppm/xfer-none-ss/export-198
- **HF repos:** [InternalCan/xfer-none-ss](https://huggingface.co/InternalCan/xfer-none-ss)
- **Notes:** 25395 = chain job, 25405 aborted at preflight, 25411 the run.
- **Citations:**
  - `/data/home/amartyavarma/ppm/xfer-none-ss/README.md:3 -> control B definition`
  - `/data/home/amartyavarma/ppm/xfer-none-ss/README.md:7 -> Init local-12829 + ASR w-1972`
  - `/data/home/amartyavarma/ppm/xfer-none-ss/logs/xfernoness-25411.log:43 -> QQA_INIT local-12829`
  - `/data/home/amartyavarma/ppm/xfer-none-ss/logs/xfernoness-25411.log:751 -> EVAL step 198 1.5410`
  - `/data/home/amartyavarma/ppm/xfer-none-ss/README.md:16 -> FINAL EXPORT export-198`

## 32B pose FTs (Xabi v2pure, monoplex, pose-duplex, Aloe pose) and SAM 8B pose FTs

| run | status | jobs | dates | init (parent node@step) | steps | headline evals | outputs |
|---|---|---|---|---|---|---|---|
| [`aloe-nopose-ft`](#aloe-nopose-ft) | finished | 22286, 22289, 22291, 22292 (GCS backup) | 2026-10-16 15:53 .. 2026-10-16 16:59 PT (22291) | ppm32b@11488 | 0 -> 430 (target 430) | word CE(norm) 1.7748 @430 (step0 1.8127) | export-72; export-143; export-215 |
| [`aloe-uncond-ft`](#aloe-uncond-ft) | finished | 22317, 22321 (GCS backup) | 2026-10-16 17:02 .. 2026-10-16 18:35 PT | ppm32b@11488 | 0 -> 430 (target 430) | word CE(norm) 1.8733 @430 (step0 1.9574); pose cb0 1.1115 @430; pose light 3.0443 @430 | export-72; export-143; export-215 |
| [`aloe-v2-ft`](#aloe-v2-ft) | finished | 22382 (store staging failed), 22383, 22401 (GCS backup) | 2026-10-16 19:30 .. 2026-10-16 20:01 PT | ppm32b@11488 | 0 -> 253 (target 253) | word CE(norm) 1.8281 @253 (step0 1.8730) | export-253 |
| [`aloe-zoom-cotrain`](#aloe-zoom-cotrain) | finished | 26467, 26469 (HF local exports), 26470 (gather), 26408 (zoom scan) | 2026-10-21 02:08 .. 2026-10-21 02:39 PT | ppm32b@15973, asrppm-w15973@15973 | 0 -> 253 (target 253) | word CE(norm) 1.8125 @253 (Aloe eval; step0 2.0125) | export-253 (node-local; /data consolidated refused) / HF: ppm-aloe-zoom-cotrain |
| [`aloe-samsep-16861`](#aloe-samsep-16861) | abandoned | 28249, 28250 (gather, cancelled), 28263 (watchdog, cancelled), 28268/28269 (uploads) | 2026-10-23 22:36 .. 2026-10-23 23:42 PT | ppm32b@16861, asrppm-w15973@15973 | 0 -> 90 (target 249; stopped) | word CE(norm) 1.8257 @83 (last eval; step0 2.0088) | export-41; export-83; export-stop90 / HF: ppm-aloe-samsep-16861-final, ppm-aloe-samsep-16861 |
| [`aloe-samsep-a16861w480`](#aloe-samsep-a16861w480) | finished | 28511, 28512 (watchdog), 28514-28521 (HF takeovers), 28522 (release) | 2026-10-24 02:25 .. 2026-10-24 02:55 PT | ppm32b@16861, asrppm-w16861@16861-w480 | 0 -> 249 (target 249) | word CE(norm) 1.8074 @249 (step0 2.0098) | export-41; export-166; export-208 / HF: ppm-aloe-samsep-a16861w480-final, ppm-aloe-samsep-a16861w480 |
| [`aloe-samsep-vox17127-persona`](#aloe-samsep-vox17127-persona) | finished | 29444, 29421 (stage), 29451 (watchdog), 29452-29459 (HF takeovers), 29460 (release) | 2026-10-24 07:54 .. 2026-10-24 08:24 PT | ppmvox-v3@17127 | 0 -> 249 (target 249) | word CE(norm) 1.8024 @249 (step0 1.8484) | export-249; dcp-249 (full save, 8 hosts) / HF: ppm-aloe-samsep-vox17127-persona-final |
| [`xabi-v2pure-7932`](#xabi-v2pure-7932) | finished | 26209, 26241 (HF final export), 26242 (rm saves) | 2026-10-20 18:13 .. 2026-10-20 19:04 PT | ppm32b@7932 | 0 -> 200 (target 200) | word CE(norm) 2.1198 @200 (step0 2.2654); pose cb0 1.6735 @200 (final = best); pose light 3.8056 @200 | export-200 / HF: ppm-aloe-v2pure-7932-final |
| [`xabi-v2pure-11488`](#xabi-v2pure-11488) | finished | 23051 | 2026-10-17 10:28 .. 2026-10-17 16:47 PT | ppm32b@11488 | 0 -> 200 (target 200) | word CE(norm) 2.0968 @200 (step0 2.2348); pose cb0 1.6334 @200; pose light 3.7324 @200 | export-200 |
| [`xabi-v2pure-13533`](#xabi-v2pure-13533) | finished | 26185 (aborted node checks), 26186 (aborted /scratch check, no steps), 26187, 26237 (HF final export), 26238 (rm saves) | 2026-10-20 17:54 .. 2026-10-20 18:46 PT | ppm32b@13533 | 0 -> 200 (target 200) | word CE(norm) 2.0860 @200 (step0 2.2338); pose cb0 1.6190 @200; pose light 3.7091 @200 | export-200; intermediate exports 33/67/100/133/167 / HF: ppm-aloe-v2pure-13533-final, ppm-aloe-v2pure-13533 |
| [`xabi-v2pure-16610`](#xabi-v2pure-16610) | finished | 26088 (terminated before training, exit 143), 26089, 26081 (init export dcp-16610 -> local-16610) | 2026-10-20 16:34 .. 2026-10-20 17:27 PT | ppm32b@16610 | 0 -> 200 (target 200) | word CE(norm) 2.0749 @200 (step0 2.2139); pose cb0 1.6026 @200; pose light 3.6834 @200 | export-200; intermediate exports 33/67/100/133/167 / HF: ppm-aloe-v2pure-16610 |
| [`xabi-v2pure-17352`](#xabi-v2pure-17352) | finished | 27211 (killed by user ~step 35), 27549, 27550 (xfree), 27551 (HF backup), 27552 (rm saves) | 2026-10-22 ~11:45 (27211) ; 2026-10-22 17:20 .. 2026-10-22… | ppm811-v4@17352 | 0 -> 200 (target 200) | word CE(norm) 2.0786 @200 (step0 2.2235); pose cb0 1.5849 @200; pose light 3.6530 @200 | export-200; intermediate exports 33/67/100/133/167 / HF: ppm-aloe-v2pure-17352-final, ppm-aloe-v2pure-17352 |
| [`xabi-v2pure-17641`](#xabi-v2pure-17641) | finished | 27622, 27623 (xfree), 27624 (HF backup), 27625 (rm saves), 27548/27619 (init convert) (+1) | 2026-10-22 17:25 .. 2026-10-22 18:16 PT | ppm811-append2@17641 | 0 -> 200 (target 200) | word CE(norm) 2.0781 @200 (step0 2.2232); pose cb0 1.5811 @200; pose light 3.6438 @200 | export-200; intermediate exports 33/67/100/133/167 / HF: ppm-aloe-v2pure-17641-final, ppm-aloe-v2pure-17641 |
| [`xabi-v2pure-17850`](#xabi-v2pure-17850) | finished | 27757, 27756 (init convert), 27758 (xfree), 27759 (HF final), 27760 (cleanup) | 2026-10-22 22:42 .. 2026-10-22 23:34 PT | ppm811-append3@17850 | 0 -> 200 (target 200) | word CE(norm) 2.0786 @200 (step0 2.2213); pose cb0 1.5788 @200 (best of series); pose light 3.6384 @200 | export-200; intermediate exports 33/67/100/133/167 / HF: ppm-aloe-v2pure-main-17850-final, ppm-aloe-v2pure-main-17850 |
| [`xabi-v2pure-clip80`](#xabi-v2pure-clip80) | stopped-with-save | 24198, 24215 (HF backup controller), 24200 (release nodes) | 2026-10-18 17:06 .. 2026-10-18 18:22 PT | ppm32b@11488 | 0 -> 109 (target 200; stopped) | word CE(norm) 2.1008 @100 (last eval; step0 2.2290); pose cb0 1.6387 @100; pose light 3.7431 @100 | export-33; export-67; export-100 / HF: ppm-aloe-v2pure-clip80 |
| [`aloe-pose-ft`](#aloe-pose-ft) | finished | 22091, 22081-22088 (store builds), 22090 (stage), 22096 (GCS backup) | 2026-10-16 05:02 .. 2026-10-16 06:31 PT | ppm32b@11488 | 0 -> 430 (target 430) | word CE(norm) 1.7789 @430 (step0 1.8576); pose cb0 1.1111 @430; pose light 3.0438 @430 | export-72; export-143; export-215 |
| [`mono8020`](#mono8020) | finished | 24229 (aborted preflight, /scratch), 24272, 24241/24273 (release nodes), 24245 (HF backup controller) | 2026-10-18 17:28 .. 2026-10-18 18:48 PT | ppm32b@11488 | 0 -> 357 (target 357) | word CE(norm) 1.9156 @357 (step0 2.0112; 1.9574 under 23051's rule = 22317 identity); pose cb0 1.1127 @357; pose light 3.0469 @357 | export-60; export-119; export-178 / HF: ppm-aloe-pose-mono-8020 |
| [`mono8020-sam`](#mono8020-sam) | abandoned | 30621 (SAM store build, in ppm/aloe-pose-ft-sam), 30624/30641 (CPU checks) | prepared 2026-10-26 ~05:08 .. ~06:05 PT; never trained | ppm32b@11488 | none (planned 357) | unknown | unknown |
| [`mono17850`](#mono17850) | finished | 31198 (failed in 1 s: quoting bug), 31240, 31467 (HF push) | 2026-10-27 16:29 .. 2026-10-27 17:49 PT | ppm811-append3@17850 | 0 -> 357 (target 357) | word CE(norm) 1.8976 @357 (step0 1.9864); pose cb0 1.0700 @357; pose light 2.9462 @357 | export-357 / HF: ppm-pose6-mono17850-final |
| [`mono17850sam`](#mono17850sam) | finished | 31443, 31587 (HF push) | 2026-10-27 17:15 .. 2026-10-27 18:35 PT | ppm811-append3@17850 | 0 -> 357 (target 357) | word CE(norm) 1.8994 @357 (step0 1.9848); pose cb0 1.0700 @357; pose light 2.9462 @357 | export-357 / HF: ppm-pose6-mono17850sam-final |
| [`aloe-pose-rosie`](#aloe-pose-rosie) | finished | 30836 (exit 143), 30837 (exit 143), 30839 (exit 1), 30844 (exit 1), 30847 | 2026-10-27 ~02:00 .. 2026-10-27 02:40 PT (30847: 02:30:49 .… | mono8020@357 | 0 -> 18 (target 18) | word CE(norm) 1.8028 @18 (step0 1.8412; Rosie eval, 4 clips); pose cb0 1.1879 @18; pose light 3.6276 @18 | export-final (step 18) |
| [`poseduplex-v5`](#poseduplex-v5) | finished | 30054 (terminated), 30072 (exit 2), 30074, 30215 (HF dcp upload) | 2026-10-25 03:16 .. 2026-10-25 04:36 PT | ppmvox-v5@18100 | 0 -> 498 (target 498) | word CE(norm) 1.7876 @498 (step0 1.8715); pose cb0 1.0695 @498 (1.0695 also @475); pose light 2.9394 @498 | export-83; export-165; export-248 / HF: ppm-aloe-poseduplex-v5-final, ppm-aloe-poseduplex-v5 |
| [`aloe-v2init-8869`](#aloe-v2init-8869) | finished | 23507 (cancelled), 23508 | 2026-10-18 14:03 .. 2026-10-18 14:30 PT | ppm32b@8869 | 0 -> 253 (target 253) | word CE(norm) 1.8442 @253 (step0 1.8888) | export-42; export-84; export-126 / HF: ppm-aloe-v2init-8869 |
| [`aloe-v2init-10351`](#aloe-v2init-10351) | finished | 22562 | 2026-10-16 22:06 .. 2026-10-16 22:42 PT | ppm32b@10351 | 0 -> 253 (target 253) | word CE(norm) 1.8351 @253 (step0 1.8797) | export-253 / HF: ppm-aloe-v2init-10351 |
| [`aloe-v2init-12529`](#aloe-v2init-12529) | finished | 22564 (cancelled), 22575 | 2026-10-16 22:10 .. 2026-10-16 22:40 PT | ppm32b@12529 | 0 -> 253 (target 253) | word CE(norm) 1.8230 @253 (step0 1.8665) | export-253 / HF: ppm-aloe-v2init-12529 |
| [`sam8b-aloe-ft`](#sam8b-aloe-ft) | finished | 19440 | 2026-10-12 19:53 .. 2026-10-12 21:13 PT | ext:sam8b-pose@61258 | 0 -> 1132 (target 1132) | pose cb0 1.7752 heldout @1132 (eval split 1.7681; step0 heldout 1.9103); pose light 3.8048 heldout @1132 (eval 3.7509) | export-step-1132 |
| [`sam8b-aloe-active`](#sam8b-aloe-active) | finished | 19623, 19620 (init download), 19661/19662/19689 (eval-only) | 2026-10-12 22:24 .. 2026-10-12 23:06 PT | ext:sam8b-pose@61258 | 0 -> 636 (target 636) | pose cb0 2.1434 heldout @636 (eval 2.1292); pose light 4.1227 heldout @636 (eval 4.0904) | export-step-636 / HF: sam-pose-aloe-active-step636 |
| [`sam8b-aloe-posev2`](#sam8b-aloe-posev2) | finished | 19934 (aborted 4-node), 19929 (init), 19949 | 2026-10-13 05:41 .. 2026-10-13 06:22 PT | ext:sam8b-pose@61258 | 0 -> 636 (target 636) | pose cb0 1.8465 heldout @636 (eval 1.8117); pose light 3.8981 heldout @636 (eval 3.8517) | export-step-636 |
| [`sam8b-aloe-posev3`](#sam8b-aloe-posev3) | finished | 20003, 20004 (prune) | 2026-10-13 09:32 .. 2026-10-13 10:14 PT | ext:sam8b-pose@61258 | 0 -> 636 (target 636) | pose cb0 1.6305 heldout @636 (eval 1.6213); pose light 3.6700 heldout @636 (eval 3.6476) | export-step-636 / HF: sam-pose-aloe-posev3-step636 |
| [`sam8b-aloe-v2`](#sam8b-aloe-v2) | finished | 20630, 20631 (prune), 20697 (HF upload), 20700 (samples) | 2026-10-14 11:12 .. 2026-10-14 13:48 PT | ext:sam8b-pose@61258 | 0 -> 636 (target 636) | pose cb0 1.6305 heldout @636 (eval 1.6218); pose light 3.6700 heldout @636 (eval 3.6476) | export-step-636 / HF: sam-pose-aloe-v2-step636 |
| [`sam8b-aloe-v2pure`](#sam8b-aloe-v2pure) | finished | 20759, 20760 (prune), 20758 (weka clean) | 2026-10-14 14:13 .. 2026-10-14 14:55 PT | ext:sam8b-pose@61258 | 0 -> 636 (target 636) | pose cb0 1.6305 heldout @636 (eval 1.6219); pose light 3.6700 heldout @636 (eval 3.6476) | export-step-636 |
| [`sam8b-v2pure-cmp`](#sam8b-v2pure-cmp) | finished | 26121 (aborted), 26141, 26142 (HF) | 2026-10-20 17:00 .. 2026-10-20 17:29 PT | ext:sam8b-pose@61258 | 0 -> 200 (target 200) | pose cb0 1.6541 heldout @200 (step0 1.7254); pose light 3.6958 heldout @200 | export-step-200 / HF: sam-pose-aloe-v2pure-cmp-step200 |
| [`sam8b-v2pure-cmp-lr3`](#sam8b-v2pure-cmp-lr3) | finished | 26144 | 2026-10-20 17:30 .. 2026-10-20 18:02 PT | ext:sam8b-pose@61258 | 0 -> 200 (target 200) | pose cb0 1.6611 heldout @200 (step0 1.7254); pose light 3.7042 heldout @200 | export-step-200 |


### aloe-nopose-ft

**aloe-nopose-ft (job 22291): no-pose ablation of 22091**  
group `aloe-pose-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-nopose-ft` · also audited in part_speech.json

- **Jobs:** 22286, 22289, 22291, 22292 (GCS backup)
- **Dates:** 2026-10-16 15:53 .. 2026-10-16 16:59 PT (22291)
- **Base model:** unknown
- **Init:** PPM main step 11488, /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488
- **Parents (resolved):** ppm32b@11488
- **Produces:** aloe-nopose-ft@430
- **Data:** 22091's plan 9ae948a59f971fc5 items with no pose arrays (POSE_B 0); prompt persona both + clip 1/3
- **Recipe:** aloe-pose-ft/aloe_pose.env (22091) verbatim except [NOPOSE] (no pose anywhere); 430 steps, warmup 14, world 64
- **Steps:** 0 -> 430 (target 430)
- **Evals:** norm_word_ce = 1.7748 @430 (step0 1.8127)
- **Checkpoints / exports:** export-72; export-143; export-215; export-287; export-358; export-430
- **Local paths:** /scratch/amartyavarma/aloe-nopose-ft/exports (node-local)
- **HF repos:** none named in run files
- **Notes:** Speech-only ablation (no pose evals). Earlier attempts 22286/22289 have logs; their outcome was not checked. \|\| Second independent audit (part_speech.json): 22286 aborted (store staging), 22289 cancelled; 22291 the real run. A channel = partner words+Mimi (inherited from 21980 recipe; not re-verified).
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-nopose-ft/aloe_nopose.env:1 -> = 22091 verbatim except [NOPOSE] (no pose anywhere)`
  - `/data/home/amartyavarma/ppm/aloe-nopose-ft/logs/aloenoposeft-22291.log:1 -> job 22291 world 64 2026-10-16 15:53:32 PT`
  - `/data/home/amartyavarma/ppm/aloe-nopose-ft/logs/aloenoposeft-22291.log:71 -> cfg init local-11488, pose_b false`
  - `/data/home/amartyavarma/ppm/aloe-nopose-ft/logs/aloenoposeft-22291.log:90 -> step-0 word_ce_norm 1.8127`
  - `/data/home/amartyavarma/ppm/aloe-nopose-ft/logs/aloenoposeft-22291.log:742 -> step-430 word_ce_norm 1.7748`
  - `/data/home/amartyavarma/ppm/aloe-nopose-ft/gcs_local_exports.sbatch:5 -> GCS backup prefix aloe-nopose-ft-local`
  - `/data/home/amartyavarma/ppm/aloe-nopose-ft/aloe_nopose.env:1 -> = aloe_pose.env (22091) except no-pose lines`
  - `/data/home/amartyavarma/ppm/aloe-nopose-ft/logs/aloenoposeft-22286.log:34 -> init local-11488`
  - `/data/home/amartyavarma/ppm/aloe-nopose-ft/logs/aloenoposeft-22291.log:90 -> EVAL step 0 1.8127`
  - `/data/home/amartyavarma/ppm/aloe-nopose-ft/logs/aloenoposeft-22291.log:742 -> EVAL step 430 1.7748`
  - `/data/home/amartyavarma/ppm/aloe-nopose-ft/logs/aloenoposeft-22291.log:744 -> export-430`

### aloe-uncond-ft

**aloe-uncond-ft (job 22317): pose FT with channel A empty**  
group `aloe-pose-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-uncond-ft` · also audited in part_speech.json

- **Jobs:** 22317, 22321 (GCS backup)
- **Dates:** 2026-10-16 17:02 .. 2026-10-16 18:35 PT
- **Base model:** unknown
- **Init:** PPM main step 11488, /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488
- **Parents (resolved):** ppm32b@11488
- **Produces:** aloe-uncond-ft@430
- **Data:** 22091's plan 9ae948a59f971fc5 (27,517 B-track items, eval 531), POSE_B 1; prompt persona both + clip 1/3
- **Recipe:** aloe-pose-ft/aloe_pose.env (22091) verbatim except [UNCOND]: channel A empty, per-stream loss multipliers, pose head not built; 430 steps, warmup 14, world 64
- **Steps:** 0 -> 430 (target 430)
- **Evals:** norm_word_ce = 1.8733 @430 (step0 1.9574); pose_cb0_ce = 1.1115 @430; pose_light_ce = 3.0443 @430
- **Checkpoints / exports:** export-72; export-143; export-215; export-287; export-358; export-430
- **Local paths:** /scratch/amartyavarma/aloe-uncond-ft/exports (node-local)
- **HF repos:** [InternalCan/ppm-aloe-uncond-ft](https://huggingface.co/InternalCan/ppm-aloe-uncond-ft) *(name match)*
- **Notes:** Parent recipe of the v2pure family (23051 = 22317 with v2pure data). GCS export-430 upload on slinky-16 rc 1 in gcs_local log (logs/gcs_local-22321.log:723); retry logs present, final backup state not deep-checked. \|\| Second independent audit (part_speech.json): Ablation; not used downstream as far as the files show.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-uncond-ft/aloe_uncond.env:1 -> = 22091 verbatim except channel A empty, loss multipliers, no pose head`
  - `/data/home/amartyavarma/ppm/aloe-uncond-ft/logs/aloeuncondft-22317.log:3 -> job 22317 world 64 2026-10-16 17:02:45 PT`
  - `/data/home/amartyavarma/ppm/aloe-uncond-ft/logs/aloeuncondft-22317.log:85 -> cfg init local-11488, steps 430, pose_b true, a_empty true`
  - `/data/home/amartyavarma/ppm/aloe-uncond-ft/logs/aloeuncondft-22317.log:99 -> plan 9ae948a59f971fc5 POSE_B 1`
  - `/data/home/amartyavarma/ppm/aloe-uncond-ft/logs/aloeuncondft-22317.log:104 -> step-0 word_ce_norm 1.9574, pose cb0 1.2319, light 3.4934`
  - `/data/home/amartyavarma/ppm/aloe-uncond-ft/logs/aloeuncondft-22317.log:756 -> step-430 word_ce_norm 1.8733, pose cb0 1.1115, light 3.0443`
  - `/data/home/amartyavarma/ppm/aloe-uncond-ft/gcs_local_exports.sbatch:5 -> GCS backup prefix aloe-uncond-ft-local`
  - `/data/home/amartyavarma/ppm/aloe-uncond-ft/aloe_uncond.env:1 -> channel A empty variant of 22091`
  - `/data/home/amartyavarma/ppm/aloe-uncond-ft/logs/aloeuncondft-22317.log:37 -> init local-11488`
  - `/data/home/amartyavarma/ppm/aloe-uncond-ft/logs/aloeuncondft-22317.log:104 -> EVAL step 0 1.9574`
  - `/data/home/amartyavarma/ppm/aloe-uncond-ft/logs/aloeuncondft-22317.log:756 -> EVAL step 430 1.8733`
  - `/data/home/amartyavarma/ppm/aloe-uncond-ft/logs/aloeuncondft-22317.log:758 -> export-430`

### aloe-v2-ft

**aloe-v2-ft (job 22383): Aloe v2 speech FT (first chunks, words+Mimi on A)**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-v2-ft` · also audited in part_speech.json

- **Jobs:** 22382 (store staging failed), 22383, 22401 (GCS backup)
- **Dates:** 2026-10-16 19:30 .. 2026-10-16 20:01 PT
- **Base model:** unknown
- **Init:** PPM main step 11488, /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488
- **Parents (resolved):** ppm32b@11488
- **Produces:** aloe-v2-ft@253
- **Data:** handover none-crossbleed s2h-m1 stores, v2 = first chunks only: 8,067 train rows x 2 orientations = 16,134 micros (~940 h), eval 162 rows x 2 = 324 micros; voice-clip-only prompt (voice_mod 3), no persona; partner words+Mimi on A
- **Recipe:** aloe_v2.env (qqa-ft knobs): 253 steps, warmup 8, eval 25; LR trunk 3e-6 spec 6e-5 proj 3e-5 audio 6e-5 tower 1.5e-5 dec 1.5e-5; 1 row/rank/step world 64; ~5.4 s/step
- **Steps:** 0 -> 253 (target 253)
- **Evals:** norm_word_ce = 1.8281 @253 (step0 1.8730)
- **Checkpoints / exports:** export-253
- **Local paths:** /data/home/amartyavarma/ppm/aloe-v2-ft/export-253
- **HF repos:** none named in run files
- **Notes:** Speech-only, no pose. Reference recipe of the aloe-v2init series (as job 22383). \|\| Second independent audit (part_speech.json): The run the user objected to (words+Mimi on A); redone as aloe-v2vox-ft.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-v2-ft/aloe_v2.env:9 -> QQA_STEPS=253`
  - `/data/home/amartyavarma/ppm/aloe-v2-ft/aloe_v2.env:10 -> QQA_WARMUP=8`
  - `/data/home/amartyavarma/ppm/aloe-v2-ft/aloe_v2.env:21 -> QQA_LR_TRUNK=3e-6`
  - `/data/home/amartyavarma/ppm/aloe-v2-ft/logs/aloev2-22383.log:47 -> job 22383 on 8 nodes world 64`
  - `/data/home/amartyavarma/ppm/aloe-v2-ft/logs/aloev2-22383.log:100 -> init local-11488`
  - `/data/home/amartyavarma/ppm/aloe-v2-ft/logs/aloev2-22383.log:131 -> v2 first chunks only 8,067 / 162 rows`
  - `/data/home/amartyavarma/ppm/aloe-v2-ft/logs/aloev2-22383.log:138 -> step-0 word_ce_norm 1.8730`
  - `/data/home/amartyavarma/ppm/aloe-v2-ft/logs/aloev2-22383.log:606 -> step-253 word_ce_norm 1.8281`
  - `/data/home/amartyavarma/ppm/aloe-v2-ft/gcs_local_exports.sbatch:5 -> GCS backup gs://podcast-raw-audio-data-mass/amartyavarma/ckpt-backup/aloe-v2-ft-local/`
  - `memory:aloe-handover-only.md:17 -> v2 recipe: first chunks 8,067 train + 162 eval rows, ~940 h`
  - `memory:aloe-ft-voxtral-a.md:18 -> v2 launched with words+Mimi on A (user then demanded Voxtral)`
  - `/data/home/amartyavarma/ppm/aloe-v2-ft/logs/aloev2-22382.log:1 -> 22382 aborted store staging`
  - `/data/home/amartyavarma/ppm/aloe-v2-ft/logs/aloev2-22383.log:78 -> init local-11488`
  - `/data/home/amartyavarma/ppm/aloe-v2-ft/logs/aloev2-22383.log:138 -> EVAL step 0 1.8730`
  - `/data/home/amartyavarma/ppm/aloe-v2-ft/logs/aloev2-22383.log:606 -> EVAL step 253 1.8281`
  - `/data/home/amartyavarma/ppm/aloe-v2-ft/logs/aloev2-22383.log:608 -> export-253`
  - `memory:aloe-ft-voxtral-a.md:18 -> v2 launched with words+Mimi on A; user: 'DO THE SAME RUN WITH VOXTRAL'`
  - `memory:aloe-handover-only.md:17 -> v2 recipe = first chunks, 8,067 train + 162 eval rows`

### aloe-zoom-cotrain

**aloe-zoom-cotrain (job 26467): Aloe Voxtral speech FT co-trained 1:1 with zoom-strangers-5k text**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-zoom-cotrain` · also audited in part_speech.json

- **Jobs:** 26467, 26469 (HF local exports), 26470 (gather), 26408 (zoom scan)
- **Dates:** 2026-10-21 02:08 .. 2026-10-21 02:39 PT
- **Base model:** unknown
- **Init:** PPM w-15973 (/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-15973) + ASR adapter asrppm-w15973-w-1441 (= 26232's pair)
- **Parents (resolved):** ppm32b@15973, asrppm-w15973@15973
- **Produces:** aloe-zoom-cotrain@253
- **Data:** Aloe = exactly 26232's: handover none-crossbleed s2h-m1 stores, first chunks, 8,067 rows x 2 orientations = 16,134 micros (~940 h), eval 162 x 2; Voxtral A; B clip 30%, no persona. Zoom = InternalCan/zoom-strangers-5k on-disk copy, text only, 4,900 train / 100 eval conversations (594 h rendered), B text CE
- **Recipe:** copy of 26232 (aloe-v2vox-asr-w15973) + rank groups (8 aloe hosts / 8 zoom hosts, world 128) + 1:1 loss; 253 steps; trainer code/train_cotrain.py
- **Steps:** 0 -> 253 (target 253)
- **Evals:** norm_word_ce = 1.8125 @253 (Aloe eval; step0 2.0125); other = eval_zoom/word_ce_norm 1.2693 @253 (step0 1.7440)
- **Checkpoints / exports:** export-253 (node-local; /data consolidated refused)
- **Local paths:** unknown
- **HF repos:** [InternalCan/ppm-aloe-zoom-cotrain](https://huggingface.co/InternalCan/ppm-aloe-zoom-cotrain)
- **Notes:** Speech + text only (no pose). README.md:1 still says 'not launched' but the log shows the run completed. Reference run 26232 (aloe-v2vox-asr-w15973) is outside this part's scope. \|\| Second independent audit (part_speech.json): Speaker LoRA logs map the 26467 export to /data/home/amartyavarma/ppm/aloe-zoom-cotrain/export-253 (aloespklora-29545.log:264).
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-zoom-cotrain/README.md:6 -> copy of latest Aloe FT job 26232 + rank groups, 1:1 loss, no-op passes`
  - `/data/home/amartyavarma/ppm/aloe-zoom-cotrain/README.md:25 -> Aloe data exactly 26232's (first chunks 8,067 rows, ~940 h, Voxtral A, 30% clip)`
  - `/data/home/amartyavarma/ppm/aloe-zoom-cotrain/README.md:34 -> zoom: 4,900 train / 100 eval conversations, text only`
  - `/data/home/amartyavarma/ppm/aloe-zoom-cotrain/LAUNCHED_JOB:1 -> job 26467 launched 2026-10-21 02:08:37 PT`
  - `/data/home/amartyavarma/ppm/aloe-zoom-cotrain/logs/aloezoomco-26467.log:1 -> 16 hosts world 128`
  - `/data/home/amartyavarma/ppm/aloe-zoom-cotrain/logs/aloezoomco-26467.log:106 -> init w-15973 + adapter asrppm-w15973-w-1441, 253 steps`
  - `/data/home/amartyavarma/ppm/aloe-zoom-cotrain/logs/aloezoomco-26467.log:416 -> step-0 word_ce_norm 2.0125 (zoom 1.7440)`
  - `/data/home/amartyavarma/ppm/aloe-zoom-cotrain/logs/aloezoomco-26467.log:867 -> step-253 word_ce_norm 1.8125, eval_zoom/word_ce_norm 1.2693`
  - `/data/home/amartyavarma/ppm/aloe-zoom-cotrain/logs/aloezoomco-26467.log:869 -> /data export-253 REFUSED (floor)`
  - `/data/home/amartyavarma/ppm/aloe-zoom-cotrain/logs/BACKUP_LOG.txt:143 -> export-253 HF upload VERIFY PASS to private InternalCan/ppm-aloe-zoom-cotrain`
  - `memory:ppm-vs-811-naming.md:9 -> Aloe reference FTs 26232/26467 used PPM w-15973 + asrppm-w15973`
  - `/data/home/amartyavarma/ppm/aloe-zoom-cotrain/aloe_zoom_cotrain.env:1 -> user: co-train aloe and zoom-strangers-5k 1:1; = 26232 env + COTRAIN`
  - `/data/home/amartyavarma/ppm/aloe-zoom-cotrain/LAUNCHED_JOB:1 -> job 26467 10-21 02:08:37 PT`
  - `/data/home/amartyavarma/ppm/aloe-zoom-cotrain/logs/aloezoomco-26467.log:51 -> init w-15973`
  - `/data/home/amartyavarma/ppm/aloe-zoom-cotrain/logs/aloezoomco-26467.log:416 -> EVAL step 0 2.0125`
  - `/data/home/amartyavarma/ppm/aloe-zoom-cotrain/logs/aloezoomco-26467.log:867 -> EVAL step 253 1.8125`
  - `/data/home/amartyavarma/ppm/aloe-zoom-cotrain/logs/BACKUP_LOG.txt:1 -> HF InternalCan/ppm-aloe-zoom-cotrain`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/PROMPT.md:17 -> the 'Aloe + zoom co-train export-253' served the live demo before 10-23`

### aloe-samsep-16861

**aloe-ft-32-samsep 16861 (job 28249): Aloe SAM-sep speech FT, stopped at 90**  
group `aloe-speech-ft` · status **abandoned** · dir `/data/home/amartyavarma/ppm/aloe-ft-32-samsep` · also audited in part_speech.json

- **Jobs:** 28249, 28250 (gather, cancelled), 28263 (watchdog, cancelled), 28268/28269 (uploads)
- **Dates:** 2026-10-23 22:36 .. 2026-10-23 23:42 PT
- **Base model:** unknown
- **Init:** pure PPM dcp-16861, node-local /scratch/amartyavarma/aloe-samsep-16861-init; ASR adapter asrppm-w15973-w-1441
- **Parents (resolved):** ppm32b@16861, asrppm-w15973@15973
- **Produces:** aloe-samsep-16861@90
- **Data:** s2h-m1-samsep stores (Aloe handover none-crossbleed bucket, first chunks, Mimi swapped for SAM-separated codes); plan d1a3e3721328da0a: 15,898 train micros in 249 steps x 64 logical ranks; eval 316; Voxtral latents of partner on A; prompt B clip 30%, no persona
- **Recipe:** aloe-zoom-cotrain env (26467) = 26232's aloe_v2vox_raw.env minus co-train; init swapped; 249 steps, warmup 8, LR trunk 3e-6; world 14 (slinky-18 x8 + slinky-4 x6) logical 64 with AdamW host offload
- **Steps:** 0 -> 90 (target 249; stopped)
- **Evals:** norm_word_ce = 1.8257 @83 (last eval; step0 2.0088)
- **Checkpoints / exports:** export-41; export-83; export-stop90
- **Local paths:** unknown
- **HF repos:** [InternalCan/ppm-aloe-samsep-16861-final](https://huggingface.co/InternalCan/ppm-aloe-samsep-16861-final), [InternalCan/ppm-aloe-samsep-16861](https://huggingface.co/InternalCan/ppm-aloe-samsep-16861)
- **Notes:** Speech-only (no pose). Superseded by 28511 (same recipe, matched adapter asrppm-w16861 w-480). \|\| Second independent audit (part_speech.json): Local copies of export-stop90 deleted 10-23 23:57 PT; HF is the only copy (STOPPED_1024.md:8).
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/run.env:1 -> user: fine-tune latest 32B PPM, no cotrain; handover rows with SAM codes`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/run.env:11 -> QQA_INIT=/scratch/amartyavarma/aloe-samsep-16861-init`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/RESUME_1023.md:4 -> base corrected to pure PPM dcp-16861`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-28249.log:117 -> cfg: init + adapter asrppm-w15973-w-1441, 249 steps, persona none, voice_frac 0.3`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-28249.log:216 -> plan d1a3e3721328da0a 15,898 micros 249 x 64 logical on 14 GPUs`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-28249.log:254 -> step-0 word_ce_norm 2.0088`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-28249.log:366 -> step-83 word_ce_norm 1.8257`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/STOPPED_1024.md:1 -> stopped 10-23 23:41 PT: rerun with an adapter matched to 16861`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/STOPPED_1024.md:2 -> stopped at step 90/249, resume state not saved`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/STOPPED_1024.md:5 -> export-stop90 on HF -final VERIFY PASS`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/STOPPED_1024.md:6 -> export-41/83 on HF InternalCan/ppm-aloe-samsep-16861`
  - `memory:ppm-vs-811-naming.md:11 -> on 10-23 the 17850 init was wrongly taken as PPM; user corrected`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/LAUNCHED_JOB:1 -> job 28249 slinky-18 + slinky-4, run.env`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-28249.log:90 -> VOX_ADAPTER asrppm-w15973-w-1441`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-28249.log:254 -> EVAL step 0 2.0088`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/STOPPED_1024.md:1 -> stopped 10-23 23:41 PT for a matched adapter`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/STOPPED_1024.md:3 -> last eval step 83 1.8257`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/STOPPED_1024.md:5 -> export-stop90 only on HF InternalCan/ppm-aloe-samsep-16861-final`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/PROMPT.md:7 -> data = SAM-separated Mimi codes on the same rows`

### aloe-samsep-a16861w480

**aloe-ft-32-samsep a16861w480 (job 28511): SAM-sep speech FT, PPM 16861 + matched adapter**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/w64` · also audited in part_speech.json

- **Jobs:** 28511, 28512 (watchdog), 28514-28521 (HF takeovers), 28522 (release)
- **Dates:** 2026-10-24 02:25 .. 2026-10-24 02:55 PT
- **Base model:** unknown
- **Init:** pure PPM dcp-16861 (/scratch/amartyavarma/aloe-samsep-16861-init) + ASR adapter /data/home/amartyavarma/ppm/aloe-ft-32-samsep/adapter-init/asrppm-w16861-w-480
- **Parents (resolved):** ppm32b@16861, asrppm-w16861@16861-w480
- **Produces:** aloe-samsep-a16861w480@249
- **Data:** same as 28249 (s2h-m1-samsep stores, plan d1a3e3721328da0a, 15,898 micros, eval 316, B clip 30%, no persona, Voxtral A)
- **Recipe:** 28249 run.env + matched-adapter swap (run_w64.env); world 64 on 8 MoT hosts (slinky-11..17,19); 249 steps, warmup 8, mean 5.94 s/step
- **Steps:** 0 -> 249 (target 249)
- **Evals:** norm_word_ce = 1.8074 @249 (step0 2.0098)
- **Checkpoints / exports:** export-41; export-166; export-208; export-249
- **Local paths:** unknown
- **HF repos:** [InternalCan/ppm-aloe-samsep-a16861w480-final](https://huggingface.co/InternalCan/ppm-aloe-samsep-a16861w480-final), [InternalCan/ppm-aloe-samsep-a16861w480](https://huggingface.co/InternalCan/ppm-aloe-samsep-a16861w480)
- **Notes:** Speech-only. 'asrppm@16861' = ASR adapter trained on ppm32b w-16861 (adapter step w-480). \|\| Second independent audit (part_speech.json): Live demo model 10-24 03:12 to 07:59 PT, then fallback.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-28511.log:128 -> cfg init 16861 + adapter asrppm-w16861-w-480, 249 steps`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-28511.log:282 -> plan d1a3e3721328da0a, stores s2h-m1-samsep`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-28511.log:426 -> step-0 word_ce_norm 2.0098`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-28511.log:818 -> step-249 word_ce_norm 1.8074`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/w64/DONE_1024.md:1 -> job 28511 10-24 02:25-02:55 PT, 249 steps, exit 0`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/w64/DONE_1024.md:3 -> final export on HF -final/export-249 VERIFY PASS`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/w64/DONE_1024.md:6 -> intermediates export-41/166/208 on HF`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/w64/DONE_1024.md:10 -> node-local export-249 deleted; copies = HF + live demo copy on slinky-10`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/A16861_RERUN_CANCELLED.txt:1 -> w-1500 matched-adapter rerun dropped; user went with w-480`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-28511.log:67 -> QQA_INIT 16861 init`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-28511.log:75 -> QQA_LR_TRUNK=3e-6`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-28511.log:81 -> QQA_STEPS=249`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-28511.log:92 -> VOX_ADAPTER asrppm-w16861-w-480 (sha line 93)`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-28511.log:426 -> EVAL step 0 2.0098`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-28511.log:818 -> EVAL step 249 1.8074`
  - `/data/home/amartyavarma/ppm/control/OPS_PLAYBOOK.md:308 -> final export verified on HF InternalCan/ppm-aloe-samsep-a16861w480-final/export-249`
  - `memory:only-demo-aloe-mothers.md:21 -> live demo 10-24 03:12 PT served a16861w480 export-249`

### aloe-samsep-vox17127-persona

**aloe-ft-32-samsep vox17127-persona (job 29444): 29345 + persona text on B (p 0.5)**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona` · also audited in part_speech.json

- **Jobs:** 29444, 29421 (stage), 29451 (watchdog), 29452-29459 (HF takeovers), 29460 (release)
- **Dates:** 2026-10-24 07:54 .. 2026-10-24 08:24 PT
- **Base model:** unknown
- **Init:** PPMVOX v3 w-17127 with ALOE_INIT_VOX=1 (as 29345)
- **Parents (resolved):** ppmvox-v3@17127
- **Produces:** aloe-samsep-vox17127-persona@249
- **Data:** 29345's data + persona prose block on B: p_target 0.5 (realised 0.5055, eligible 0.8202, p_eff 0.6096); B clip 30% unchanged
- **Recipe:** vox/run_vox.env (29345) with ONLY the persona block on B (data_aloe_persona.py, train_samsep_persona.py); 249 steps, same LRs/schedule; world 64 on 8 hosts; end full save dcp-249
- **Steps:** 0 -> 249 (target 249)
- **Evals:** norm_word_ce = 1.8024 @249 (step0 1.8484)
- **Checkpoints / exports:** export-249; dcp-249 (full save, 8 hosts)
- **Local paths:** /data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona/export-vox17127-persona-249
- **HF repos:** [InternalCan/ppm-aloe-samsep-vox17127-persona-final](https://huggingface.co/InternalCan/ppm-aloe-samsep-vox17127-persona-final), [InternalCan/ppm-aloe-samsep-vox17127-persona](https://huggingface.co/InternalCan/ppm-aloe-samsep-vox17127-persona) *(name match)*, [InternalCan/ppm-aloe-samsep-vox17127-persona-saves](https://huggingface.co/InternalCan/ppm-aloe-samsep-vox17127-persona-saves) *(name match)*
- **Deployed:** Aloe speech demo 10-24 clone - model 'persona' (paused)
- **Notes:** Speech-only. Persona draw p 0.5 predates/differs from the 80/20 rule (user ask via main 10-24). \|\| Second independent audit (part_speech.json): Served as 'Aloe FT + persona' in the 10-24 demo picker (memory only-demo-aloe-mothers.md:13).
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona/run_persona.env:1 -> = 29345 with ONLY the persona text block on B`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona/logs/aloepersona-29444.log:1 -> job 29444 on 8 hosts`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona/logs/aloepersona-29444.log:180 -> persona cfg p_target 0.5 side B`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona/logs/aloepersona-29444.log:397 -> step-0 word_ce_norm 1.8484`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona/logs/aloepersona-29444.log:783 -> step-249 word_ce_norm 1.8024`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona/STATUS.txt:4 -> p=0.5 CPU test realised fractions`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona/STATUS.txt:6 -> DONE: final 1.8024; dcp-249 full save; export-249 on /data + HF -final VERIFY PASS`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona/STATUS.txt:7 -> export loaded by the live demo (aloedemo persona backends)`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona/logs/aloepersona-29444.log:397 -> EVAL step 0 1.8484`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona/logs/aloepersona-29444.log:783 -> EVAL step 249 1.8024`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona/logs/aloepersona-29444.log:787 -> export-249`
  - `/data/home/amartyavarma/ppm/control/OPS_PLAYBOOK.md:408 -> user spec: 50% persona rows, prose only`
  - `/data/home/amartyavarma/ppm/control/OPS_PLAYBOOK.md:413 -> persona share 0.5055, step-0 eval bit-identical to 29345`
  - `/data/home/amartyavarma/ppm/control/OPS_PLAYBOOK.md:423 -> export COMPLETE /data path`

### xabi-v2pure-7932

**Xabi v2pure scaling point 7932 (aloe-v2pure-7932)**  
group `xabi-v2pure` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-v2pure-7932`

- **Jobs:** 26209, 26241 (HF final export), 26242 (rm saves)
- **Dates:** 2026-10-20 18:13 .. 2026-10-20 19:04 PT
- **Base model:** unknown
- **Init:** PPM 32B dcp-7932 restored from private HF InternalCan/ppm-v1, converted with ppm/code/ckpt_local_export.py; node-local /scratch/amartyavarma/ppm-restore/init-7932-A
- **Parents (resolved):** ppm32b@7932
- **Produces:** xabi-v2pure-7932@200
- **Data:** Aloe pose release v2 'v2pure' (Xabi single-speaker pose+speech), stores ppm/aloe-v2pure-ft/stores p0..p7; plan 8672b1eb38dae118: 367,495 train clips (2047.4 h) -> 12,791 rows -> 200 steps x 64 rows; eval 3,481 held-out clips; prompt = 23051's rule: voice clip on 1/3 (voice_mod 3), no persona (stores carry none)
- **Recipe:** job 26089 verbatim except init (run.env); trainer code/train_aloe_pose.py; channel A empty, B = speech+words+pose; LR trunk 3e-6 / spec 6e-5 / proj 3e-5 / audio 6e-5 / towers 1.5e-5 / decp_bb 1e-5, decp_bb_alpha 1.0, warmup 7, WSD linear decay to 0 at 200, dec_k 8; world 64 (8 nodes)
- **Steps:** 0 -> 200 (target 200)
- **Evals:** norm_word_ce = 2.1198 @200 (step0 2.2654); pose_cb0_ce = 1.6735 @200 (final = best); pose_light_ce = 3.8056 @200
- **Checkpoints / exports:** export-200
- **Local paths:** /scratch/amartyavarma/aloe-v2pure-7932/exports (node-local shards, 8 hosts)
- **HF repos:** [InternalCan/ppm-aloe-v2pure-7932-final](https://huggingface.co/InternalCan/ppm-aloe-v2pure-7932-final)
- **Notes:** Scaling-series point (pretraining loss -> FT loss). Intermediate export steps 33/67/100/133/167 are configured (cfg line 89) but their HF backup was not verified here; only the -final repo is named in BACKUP_LOG.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-v2pure-7932/run.env:1 -> = 26089 verbatim except init = PPM step 7932`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-7932/run.env:2 -> init restored from HF InternalCan/ppm-v1 dcp-7932, node-local init-7932-A`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-7932/RUNNING_JOB.txt:1 -> job 26209 started 2026-10-20 18:13:41 PT`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-7932/RUNNING_JOB.txt:2 -> ended 19:04:45 PT rc 0`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-7932/logs/aloev2p7932-26209.log:89 -> resolved cfg (init path, LRs, decp_bb_alpha 1.0, voice_mod 3)`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-7932/logs/aloev2p7932-26209.log:99 -> plan 8672b1eb38dae118, 367,495 clips 2047.4 h, 200 x 64 rows`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-7932/logs/aloev2p7932-26209.log:105 -> step-0 eval word_ce_norm 2.2654, pose cb0 1.8496, light 4.2563`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-7932/logs/aloev2p7932-26209.log:516 -> step-200 eval word_ce_norm 2.1198, pose cb0 1.6735, pose light 3.8056`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-7932/BACKUP_JOB:1 -> HF repo InternalCan/ppm-aloe-v2pure-7932-final; /data export-200 refused by 300 GB floor, node-local export-200 only copy`
  - `memory:xabi-ft-scaling-laws.md:22 -> series table 7932: 1.6735 / 3.806`

### xabi-v2pure-11488

**aloe-v2pure-ft (job 23051): first Xabi v2pure FT, scaling point 11488**  
group `xabi-v2pure` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-v2pure-ft`

- **Jobs:** 23051
- **Dates:** 2026-10-17 10:28 .. 2026-10-17 16:47 PT
- **Base model:** unknown
- **Init:** PPM main step 11488, /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488
- **Parents (resolved):** ppm32b@11488
- **Produces:** xabi-v2pure-11488@200
- **Data:** Aloe pose release v2 (v2pure) stores built here (code/build_v2pure_rows.py from aloe-pose/sam-cache-active-v2pure = the 8B SAM v2pure FT's cache); 367,495 train clips (92.13M frames, 2,047 h, 12,384 recordings) -> 12,791 rows -> 200 steps x 64 rows; eval 3,481 held-out clips (113 recordings = 8B's held-out set); prompt voice clip 1/3, persona_sides both but stores have no persona
- **Recipe:** job 22317 (aloe-uncond-ft) verbatim except data (aloe_v2pure.env [V2PURE]); world 24 on slinky-4/7/15; warmup 7, linear decay to 0 at 200, eval every 25, exports 33/67/100/133/167 node-local + 200; ~99 s/step
- **Steps:** 0 -> 200 (target 200)
- **Evals:** norm_word_ce = 2.0968 @200 (step0 2.2348); pose_cb0_ce = 1.6334 @200; pose_light_ce = 3.7324 @200
- **Checkpoints / exports:** export-200
- **Local paths:** /data/home/amartyavarma/ppm/aloe-v2pure-ft/export-200; /scratch/amartyavarma/aloe-v2pure-ft/exports/export-<step> (node-local)
- **HF repos:** [InternalCan/ppm-aloe-v2pure-ft](https://huggingface.co/InternalCan/ppm-aloe-v2pure-ft) *(name match)*
- **Notes:** This dir IS the 11488 point of the scaling series (memory lists 11488 but no aloe-v2pure-11488 dir exists). 26089 was cloned from this run at world 64. HF backup of its exports not found in the dir (unknown).
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-v2pure-ft/README.md:3 -> init PPM main step 11488 local-11488; recipe = 22317 verbatim, data changes`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-ft/README.md:7 -> stores built from aloe-pose/sam-cache-active-v2pure`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-ft/README.md:14 -> nodes slinky-4/7/15, world 24`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-ft/README.md:18 -> 367,495 clips, 2,047 h, 12,791 rows, 200 x 64; eval 3,481 clips`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-ft/README.md:21 -> warmup 7, decay to 0 at 200, eval every 25, export steps`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-ft/README.md:39 -> node-local exports + /data export-<steps>`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-ft/RUNNING_JOB.txt:1 -> started 2026-10-17 10:28:38 PT`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-ft/RUNNING_JOB.txt:2 -> ended 16:47:38 PT rc 0`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-ft/logs/aloev2pureft-23051.log:76 -> init local-11488`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-ft/logs/aloev2pureft-23051.log:110 -> step-0 eval word_ce_norm 2.2348, pose cb0 1.8619, light 4.3650`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-ft/logs/aloev2pureft-23051.log:520 -> step-200 word_ce_norm 2.0968, pose cb0 1.6334, light 3.7324`
  - `memory:xabi-ft-scaling-laws.md:23 -> series table 11488: 1.6334 / 3.732`

### xabi-v2pure-13533

**Xabi v2pure scaling point 13533 (aloe-v2pure-13533)**  
group `xabi-v2pure` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-v2pure-13533`

- **Jobs:** 26185 (aborted node checks), 26186 (aborted /scratch check, no steps), 26187, 26237 (HF final export), 26238 (rm saves)
- **Dates:** 2026-10-20 17:54 .. 2026-10-20 18:46 PT
- **Base model:** unknown
- **Init:** PPM 32B dcp-13533 restored from private HF InternalCan/ppm-v1, converted with ckpt_local_export.py; node-local /scratch/amartyavarma/ppm-restore/init-13533-B
- **Parents (resolved):** ppm32b@13533
- **Produces:** xabi-v2pure-13533@200
- **Data:** same as 26089: v2pure plan 8672b1eb38dae118, 367,495 clips (2047.4 h) -> 200 x 64 rows; eval 3,481 clips; voice clip 1/3, no persona
- **Recipe:** job 26089 verbatim except init; world 64 (8 nodes); LR trunk 3e-6, warmup 7, decay to 0 at 200, decp_bb alpha 1.0 LR 1e-5
- **Steps:** 0 -> 200 (target 200)
- **Evals:** norm_word_ce = 2.0860 @200 (step0 2.2338); pose_cb0_ce = 1.6190 @200; pose_light_ce = 3.7091 @200
- **Checkpoints / exports:** export-200; intermediate exports 33/67/100/133/167
- **Local paths:** /scratch/amartyavarma/aloe-v2pure-13533/exports (node-local)
- **HF repos:** [InternalCan/ppm-aloe-v2pure-13533-final](https://huggingface.co/InternalCan/ppm-aloe-v2pure-13533-final), [InternalCan/ppm-aloe-v2pure-13533](https://huggingface.co/InternalCan/ppm-aloe-v2pure-13533)
- **Notes:** Jobs 26185/26186 aborted at pre-train node checks (no steps); 26187 is the real run.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-v2pure-13533/run.env:1 -> = 26089 verbatim except init = PPM step 13533`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-13533/run.env:2 -> restored from HF InternalCan/ppm-v1 dcp-13533; init-13533-B`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-13533/RUNNING_JOB.txt:1 -> job 26187 started 2026-10-20 17:54:20 PT`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-13533/RUNNING_JOB.txt:2 -> ended 18:46:47 PT rc 0`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-13533/logs/aloev2p13533-26186.log:1 -> 26186 header; its tail shows ABORT /scratch 160 GB < 165 GB`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-13533/logs/aloev2p13533-26187.log:89 -> init init-13533-B`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-13533/logs/aloev2p13533-26187.log:105 -> step-0 word_ce_norm 2.2338, pose cb0 2.3960, light 5.0983`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-13533/logs/aloev2p13533-26187.log:516 -> step-200 word_ce_norm 2.0860, pose cb0 1.6190, light 3.7091`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-13533/BACKUP_JOB:1 -> HF repo InternalCan/ppm-aloe-v2pure-13533-final, node-local export-200 only copy`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-13533/logs/BACKUP_LOG.txt:17 -> export-33 HF upload VERIFY PASS to intermediates repo InternalCan/ppm-aloe-v2pure-13533`
  - `memory:xabi-ft-scaling-laws.md:24 -> series table 13533: 1.6190 / 3.709`

### xabi-v2pure-16610

**Xabi v2pure point 16610 = REFERENCE recipe job 26089 (aloe-v2pure-16610)**  
group `xabi-v2pure` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-v2pure-16610`

- **Jobs:** 26088 (terminated before training, exit 143), 26089, 26081 (init export dcp-16610 -> local-16610)
- **Dates:** 2026-10-20 16:34 .. 2026-10-20 17:27 PT
- **Base model:** unknown
- **Init:** PPM 32B step 16610: /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-16610 (exported from /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-16610 by job 26081)
- **Parents (resolved):** ppm32b@16610
- **Produces:** xabi-v2pure-16610@200
- **Data:** v2pure stores ppm/aloe-v2pure-ft/stores p0..p7; plan 8672b1eb38dae118: 367,495 clips (2047.4 h) -> 12,791 rows -> 200 x 64 rows; eval 3,481 clips; voice clip 1/3 (voice_mod 3), persona_sides both (none in stores)
- **Recipe:** aloe-v2pure-ft (23051) verbatim except init and 8 nodes (world 64); cfg: steps 200, warmup 7, WSD linear to 0, lr trunk 3e-6 spec 6e-5 proj 3e-5 audio 6e-5 tower_s/tower_p 1.5e-5 decp_bb 1e-5, decp_bb_alpha 1.0, dec_k 8 dec_lr 1.5e-5 dec_p_lr 2e-5, rows_per_step 64, a_empty true, pose_b true, export_steps 33/67/100/133/167
- **Steps:** 0 -> 200 (target 200)
- **Evals:** norm_word_ce = 2.0749 @200 (step0 2.2139); pose_cb0_ce = 1.6026 @200; pose_light_ce = 3.6834 @200
- **Checkpoints / exports:** export-200; intermediate exports 33/67/100/133/167
- **Local paths:** /data/home/amartyavarma/ppm/aloe-v2pure-16610/export-200; /scratch/amartyavarma/aloe-v2pure-16610/exports (node-local)
- **HF repos:** [InternalCan/ppm-aloe-v2pure-16610](https://huggingface.co/InternalCan/ppm-aloe-v2pure-16610)
- **Notes:** 26089 is the byte-identical reference every later v2pure point clones. ppm/samdata-ft holds only the dcp-16610 export tooling for this run and the samdata_ft.html scaling page (no training run there).
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-v2pure-16610/run.env:1 -> = aloe-v2pure-ft (23051) verbatim except init local-16610, 8 nodes`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-16610/RUNNING_JOB.txt:1 -> job 26089 started 2026-10-20 16:34:37 PT`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-16610/RUNNING_JOB.txt:2 -> ended 17:27:25 PT rc 0`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-16610/logs/aloev2p16610-26089.log:81 -> full resolved cfg (init, LRs, schedule, data cfg, stores)`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-16610/logs/aloev2p16610-26089.log:110 -> plan 8672b1eb38dae118 data line`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-16610/logs/aloev2p16610-26089.log:116 -> step-0 word_ce_norm 2.2139, pose cb0 2.7795, light 5.4741`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-16610/logs/aloev2p16610-26089.log:527 -> step-200 word_ce_norm 2.0749, pose cb0 1.6026, pose light 3.6834`
  - `/data/home/amartyavarma/ppm/samdata-ft/logs/export16610-26081.log:1 -> export dcp-16610 -> ckpt-infer/ppm-v1/local-16610 (10-20 16:24 PT)`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-16610/logs/BACKUP_LOG.txt:1 -> HF InternalCan/ppm-aloe-v2pure-16610 named in backup log`
  - `memory:xabi-ft-scaling-laws.md:13 -> series uses job 26089's recipe verbatim, only init changes`
  - `memory:xabi-ft-scaling-laws.md:25 -> 16610: 1.6026 / 3.683`

### xabi-v2pure-17352

**Xabi v2pure point 17352 (8,1,1 init; aloe-v2pure-17352)**  
group `xabi-v2pure` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-v2pure-17352`

- **Jobs:** 27211 (killed by user ~step 35), 27549, 27550 (xfree), 27551 (HF backup), 27552 (rm saves)
- **Dates:** 2026-10-22 ~11:45 (27211) ; 2026-10-22 17:20 .. 2026-10-22 18:13 PT (27549)
- **Base model:** unknown
- **Init:** PPM 8,1,1 step 17352: /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17352 (decp_bb = frozen 16861 weights, trained as 26089)
- **Parents (resolved):** ppm811-v4@17352
- **Produces:** xabi-v2pure-17352@200
- **Data:** same as 26089 (v2pure plan 8672b1eb38dae118, 200 x 64 rows, eval 3,481 clips, clip 1/3, no persona)
- **Recipe:** 26089 run.env verbatim except init, names/dirs, [SPACE] ALOE_SAVE_EVERY_MIN 60 -> 100000 (no hourly resumable save; operational); final_save 0; world 64
- **Steps:** 0 -> 200 (target 200)
- **Evals:** norm_word_ce = 2.0786 @200 (step0 2.2235); pose_cb0_ce = 1.5849 @200; pose_light_ce = 3.6530 @200
- **Checkpoints / exports:** export-200; intermediate exports 33/67/100/133/167
- **Local paths:** /scratch/amartyavarma/aloe-v2pure-17352/exports (node-local)
- **HF repos:** [InternalCan/ppm-aloe-v2pure-17352-final](https://huggingface.co/InternalCan/ppm-aloe-v2pure-17352-final), [InternalCan/ppm-aloe-v2pure-17352](https://huggingface.co/InternalCan/ppm-aloe-v2pure-17352)
- **Notes:** Killed first attempt 27211 reached eval step 33 (logs/aloev2p17352-27211.log:213, word_ce_norm 2.1154) before the user's kill; the row's results are from 27549.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-v2pure-17352/run.env:1 -> extend scaling series exactly as 26089`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-17352/run.env:3 -> init PPM 8,1,1 step 17352, decp_bb frozen 16861 weights`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-17352/run.env:5 -> ALOE_SAVE_EVERY_MIN 60 -> 100000`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-17352/KILLED_1022.txt:1 -> 27211 cancelled 10-22 12:19 PT at ~step 35 on user 'kill it'`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-17352/RUNNING_JOB.txt:1 -> 27549 started 2026-10-22 17:20:40 PT`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-17352/RUNNING_JOB.txt:2 -> ended 18:13:09 PT rc 0`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-17352/BACKUP_JOB:1 -> final_save 0, xfree/backup/rm jobs, HF repos -final and intermediates`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-17352/logs/aloev2p17352-27549.log:82 -> init local-17352`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-17352/logs/aloev2p17352-27549.log:100 -> step-0 word_ce_norm 2.2235, pose cb0 2.8610, light 5.6819`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-17352/logs/aloev2p17352-27549.log:511 -> step-200 word_ce_norm 2.0786, pose cb0 1.5849, light 3.6530`
  - `memory:xabi-ft-scaling-laws.md:41 -> 17352: 1.5849 / 3.6530`

### xabi-v2pure-17641

**Xabi v2pure point 17641 (end of 8,1,1 delivery-2; aloe-v2pure-17641)**  
group `xabi-v2pure` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-v2pure-17641`

- **Jobs:** 27622, 27623 (xfree), 27624 (HF backup), 27625 (rm saves), 27548/27619 (init convert), 27618 (ftb_bonly)
- **Dates:** 2026-10-22 17:25 .. 2026-10-22 18:16 PT
- **Base model:** unknown
- **Init:** PPM 8,1,1 step 17641 (end of delivery-2 append run job 27204): /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17641
- **Parents (resolved):** ppm811-append2@17641
- **Produces:** xabi-v2pure-17641@200
- **Data:** same as 26089 (v2pure plan 8672b1eb38dae118, 200 x 64 rows, eval 3,481 clips, clip 1/3, no persona)
- **Recipe:** 26089 run.env verbatim except init, names, ALOE_SAVE_EVERY_MIN 100000; world 64 spread over the 16 PPM hosts with per-host rank counts (run.sh / tools/ftb_chain.sh); intermediates to HF during run (tools/xfree_run.sh)
- **Steps:** 0 -> 200 (target 200)
- **Evals:** norm_word_ce = 2.0781 @200 (step0 2.2232); pose_cb0_ce = 1.5811 @200; pose_light_ce = 3.6438 @200
- **Checkpoints / exports:** export-200; intermediate exports 33/67/100/133/167
- **Local paths:** /scratch/amartyavarma/aloe-v2pure-17641/exports (node-local)
- **HF repos:** [InternalCan/ppm-aloe-v2pure-17641-final](https://huggingface.co/InternalCan/ppm-aloe-v2pure-17641-final), [InternalCan/ppm-aloe-v2pure-17641](https://huggingface.co/InternalCan/ppm-aloe-v2pure-17641)
- **Notes:** Init conversion/chain jobs (ftb_convert-27548/27619, ftb_bonly-27618) are in logs/; their exact role not deep-checked.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-v2pure-17641/run.env:1 -> point 6 = END of 8,1,1 delivery-2 append run job 27204`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-17641/run.env:3 -> init local-17641, decp_bb frozen 16861 weights trained as 26089`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-17641/run.env:5 -> world 64 = 26089's micro layout over 16 PPM hosts`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-17641/RUNNING_JOB.txt:1 -> 27622 started 2026-10-22 17:25:33 PT`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-17641/RUNNING_JOB.txt:2 -> ended 18:16:36 PT rc 0`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-17641/BACKUP_JOB:1 -> final_save 0, HF repos`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-17641/logs/aloev2p17641-27622.log:82 -> init local-17641`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-17641/logs/aloev2p17641-27622.log:100 -> step-0 word_ce_norm 2.2232, pose cb0 2.9154, light 5.7289`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-17641/logs/aloev2p17641-27622.log:511 -> step-200 word_ce_norm 2.0781, pose cb0 1.5811, light 3.6438`
  - `memory:xabi-ft-scaling-laws.md:42 -> 17641: 1.5811 / 3.6438`

### xabi-v2pure-17850

**Xabi v2pure point 17850 (aloe-v2pure-main-17850, launched as 'MONOFT')**  
group `xabi-v2pure` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-v2pure-main-17850`

- **Jobs:** 27757, 27756 (init convert), 27758 (xfree), 27759 (HF final), 27760 (cleanup)
- **Dates:** 2026-10-22 22:42 .. 2026-10-22 23:34 PT
- **Base model:** unknown
- **Init:** PPM 8,1,1 delivery-3 append run (job 27653) paused save dcp-17850, converted node-local per FT host: /scratch/amartyavarma/aloe-v2pure-main-17850-init
- **Parents (resolved):** ppm811-append3@17850
- **Produces:** xabi-v2pure-17850@200
- **Data:** same as 26089 (v2pure plan 8672b1eb38dae118, 200 x 64 rows, eval 3,481 clips, clip 1/3, no persona)
- **Recipe:** aloe-v2pure-17641/run.env (27622) verbatim except init and names/dirs (CFG_DIFF.txt); prep/launch tooling ppm/monoft (pause_main.sh + launch.sh, template/); final resumable save OFF; world 64 = 8 hosts
- **Steps:** 0 -> 200 (target 200)
- **Evals:** norm_word_ce = 2.0786 @200 (step0 2.2213); pose_cb0_ce = 1.5788 @200 (best of series); pose_light_ce = 3.6384 @200
- **Checkpoints / exports:** export-200; intermediate exports 33/67/100/133/167
- **Local paths:** /data/home/amartyavarma/ppm/aloe-v2pure-main-17850/export-200; /scratch/amartyavarma/aloe-v2pure-main-17850/exports (node-local, 11 GB/host)
- **HF repos:** [InternalCan/ppm-aloe-v2pure-main-17850-final](https://huggingface.co/InternalCan/ppm-aloe-v2pure-main-17850-final), [InternalCan/ppm-aloe-v2pure-main-17850](https://huggingface.co/InternalCan/ppm-aloe-v2pure-main-17850)
- **Deployed:** Playground (genweb) - monoplex slot 'base' (default) (live)
- **Notes:** Despite the 'monoplex' ask this is a Xabi/v2pure FT (memory monoplex-means-aloe-mono-rows). ppm/monoft holds only tooling for it (no separate run). Served in the genweb playground since 10-27 14:44 PT.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-v2pure-main-17850/run.env:1 -> MONOFT, user 10-22 ~21:55 'fine-tune on the monoplex data on 8 nodes'`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-main-17850/run.env:2 -> = 17641 run.env verbatim except init dcp-17850 of job 27653, node-local init`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-main-17850/RUNNING_JOB.txt:1 -> 27757 started 2026-10-22 22:42:43 PT`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-main-17850/RUNNING_JOB.txt:2 -> ended 23:34:44 PT rc 0`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-main-17850/BACKUP_JOB:1 -> chain jobs + HF repos`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-main-17850/logs/aloev2pm17850-27757.log:82 -> init /scratch/amartyavarma/aloe-v2pure-main-17850-init`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-main-17850/logs/aloev2pm17850-27757.log:100 -> step-0 word_ce_norm 2.2213, pose cb0 2.9608, light 5.8260`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-main-17850/logs/aloev2pm17850-27757.log:511 -> step-200 word_ce_norm 2.0786, pose cb0 1.5788, light 3.6384`
  - `/data/home/amartyavarma/ppm/monoft/STATUS.md:7 -> result pose cb0 1.5788, light 3.6384`
  - `/data/home/amartyavarma/ppm/monoft/STATUS.md:10 -> final export-200 on private HF -final, VERIFY PASS 8/8`
  - `/data/home/amartyavarma/ppm/monoft/STATUS.md:15 -> main run 27653 paused at 17850 for it`
  - `memory:monoplex-means-aloe-mono-rows.md:13 -> mis-brief: user asked monoplex data, job 27757 trained on Xabi data`
  - `memory:playground-mono-32b-v2pure.md:11 -> 10-27 14:35 PT user put this export-200 in the playground monoplex slot`

### xabi-v2pure-clip80

**aloe-v2pure-clip80 (job 24198): 23051 with 80% voice-clip / no-persona prompt**  
group `xabi-v2pure` · status **stopped-with-save** · dir `/data/home/amartyavarma/ppm/aloe-v2pure-clip80`

- **Jobs:** 24198, 24215 (HF backup controller), 24200 (release nodes)
- **Dates:** 2026-10-18 17:06 .. 2026-10-18 18:22 PT
- **Base model:** unknown
- **Init:** PPM main step 11488, /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488
- **Parents (resolved):** ppm32b@11488
- **Produces:** xabi-v2pure-clip80@100
- **Data:** 23051's v2pure stores read in place (same 12,791 rows, 200 x 64 planned); ONE change = prompt rule: voice clip drawn on 80% (realised valid 74.2% train), persona none
- **Recipe:** 23051 verbatim except prompt rule (aloe_clip80.env [CLIP80]); world 24 on slinky-0/1/2; 34-37 s/step
- **Steps:** 0 -> 109 (target 200; stopped)
- **Evals:** norm_word_ce = 2.1008 @100 (last eval; step0 2.2290); pose_cb0_ce = 1.6387 @100; pose_light_ce = 3.7431 @100
- **Checkpoints / exports:** export-33; export-67; export-100; full save step-00109 (node-local)
- **Local paths:** /scratch/amartyavarma/aloe-v2pure-clip80/saves/step-00109 (slinky-0/1/2); /scratch/amartyavarma/aloe-v2pure-clip80/exports (slinky-0/1/2)
- **HF repos:** [InternalCan/ppm-aloe-v2pure-clip80](https://huggingface.co/InternalCan/ppm-aloe-v2pure-clip80)
- **Notes:** Never resumed; nodes reclaimed by the TTS fleet. Not comparable to the scaling series (different prompt rule).
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-v2pure-clip80/README.md:1 -> 23051 re-run with ONE change, the system-prompt rule`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-clip80/README.md:54 -> job 24198 slinky-0/1/2 world 24 launched 10-18 17:06 PT`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-clip80/README.md:57 -> step-0 word_ce_norm 2.2290`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-clip80/README.md:63 -> STOPPED at step 109 (external STOP_AFTER_SAVE), full save step-00109 kept`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-clip80/README.md:67 -> HF InternalCan/ppm-aloe-v2pure-clip80 export-33/67/100`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-clip80/RUNNING_JOB.txt:2 -> ended 2026-10-18 18:22:58 PT rc 0`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-clip80/logs/aloeclip80-24198.log:129 -> step-0 word_ce_norm 2.2290, pose cb0 1.8766, light 4.3114`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-clip80/logs/aloeclip80-24198.log:314 -> step-100 word_ce_norm 2.1008, pose cb0 1.6387, light 3.7431`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-clip80/exports/BACKUP_LOG.txt:17 -> (lines 17-20) export-133/167/200 and consolidated export-200 MISSING (job ended)`

### aloe-pose-ft

**aloe-pose-ft (job 22091): first PPM Aloe pose FT (handover rows, pose on B)**  
group `aloe-pose-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-pose-ft`

- **Jobs:** 22091, 22081-22088 (store builds), 22090 (stage), 22096 (GCS backup)
- **Dates:** 2026-10-16 05:02 .. 2026-10-16 06:31 PT
- **Base model:** unknown
- **Init:** PPM main step 11488, /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488
- **Parents (resolved):** ppm32b@11488
- **Produces:** aloe-pose-ft@430
- **Data:** Aloe handover pose stores ppm/aloe-pose-ft/stores p0..p7 (17,872 rows); plan 9ae948a59f971fc5: 17,527 train rows -> 27,517 qualifying B-track micros (pose ok, no bleed into B; ~3,033 h), eval 345 rows -> 531 B micros; prompt persona_sides both + voice clip 1/3 (persona on ~every item)
- **Recipe:** = aloe-ft/aloe.env (job 21980) + [POSE] lines (PPM pose loss weights); 430 steps = 1 epoch at 1 row/rank/step, world 64; warmup 14, linear decay to 0 at 430; LR trunk 3e-6; exports 72/143/215/287/358 + 430
- **Steps:** 0 -> 430 (target 430)
- **Evals:** norm_word_ce = 1.7789 @430 (step0 1.8576); pose_cb0_ce = 1.1111 @430; pose_light_ce = 3.0438 @430
- **Checkpoints / exports:** export-72; export-143; export-215; export-287; export-358; export-430
- **Local paths:** /scratch/amartyavarma/aloe-pose-ft/exports (node-local)
- **HF repos:** none named in run files
- **Notes:** Channel A = the partner as in the 21980 Aloe FT recipe (exact A content, words+Mimi vs other, not verified here). /data export-430 no longer exists in the dir (moved/deleted: unknown); backups went to GCS (pre no-GCS rule). These stores are the data source for every monoplex FT.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-pose-ft/aloe_pose.env:1 -> = aloe-ft/aloe.env (job 21980) except [POSE]/[RUN]`
  - `/data/home/amartyavarma/ppm/aloe-pose-ft/aloe_pose.env:8 -> one epoch = 27,517 B-track items = 430 steps, warmup 14, decay to 0`
  - `/data/home/amartyavarma/ppm/aloe-pose-ft/aloe_pose.env:68 -> QQA_EXPORT_STEPS 72,143,215,287,358`
  - `/data/home/amartyavarma/ppm/aloe-pose-ft/logs/aloeposeft-22091.log:1 -> job 22091 on 8 nodes world 64, 2026-10-16 05:02:32 PT`
  - `/data/home/amartyavarma/ppm/aloe-pose-ft/logs/aloeposeft-22091.log:60 -> cfg: init local-11488, steps 430, persona_sides both, voice_mod 3`
  - `/data/home/amartyavarma/ppm/aloe-pose-ft/logs/aloeposeft-22091.log:61 -> plan 9ae948a59f971fc5 17,527 rows -> 27,517 micros; eval 531`
  - `/data/home/amartyavarma/ppm/aloe-pose-ft/logs/aloeposeft-22091.log:66 -> step-0 word_ce_norm 1.8576, pose cb0 1.2758, light 3.6254`
  - `/data/home/amartyavarma/ppm/aloe-pose-ft/logs/aloeposeft-22091.log:718 -> step-430 word_ce_norm 1.7789, pose cb0 1.1111, light 3.0438`
  - `/data/home/amartyavarma/ppm/aloe-pose-ft/gcs_local_exports.sbatch:5 -> backup gs://podcast-raw-audio-data-mass/amartyavarma/ckpt-backup/aloe-pose-ft-local/export-N/`
  - `/data/home/amartyavarma/ppm/aloe-pose-ft/stores/STORES.json:1 -> 8 stores, 2,233-2,235 rows each`
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020/README.md:13 -> the same 27,517 B tracks = 3,033 h`

### mono8020

**aloe-pose-mono-8020 (job 24272): THE monoplex pose FT (Aloe handover rows, 80/20 prompt)**  
group `monoplex` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-pose-mono-8020`

- **Jobs:** 24229 (aborted preflight, /scratch), 24272, 24241/24273 (release nodes), 24245 (HF backup controller)
- **Dates:** 2026-10-18 17:28 .. 2026-10-18 18:48 PT
- **Base model:** unknown
- **Init:** PPM main step 11488, /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 (resharded 128 -> 64; pose_head not loaded)
- **Parents (resolved):** ppm32b@11488
- **Produces:** mono8020@357
- **Data:** Aloe handover pose rows of job 22091 (ppm/aloe-pose-ft/stores p0..p7), one monoplex item per qualifying track: 27,517 train items (3,033 h, 136.5M frames) -> 22,825 rows -> 357 steps x 64 rows; 531 eval items; prompt = 10-18 rule voice clip p 0.8 / persona p 0.2 independent draws (realised clip present 28.2%, persona 20.1%, empty 57.6%)
- **Recipe:** job 23051 / 24198 code + env with [MONO8020] changes (ALOE_DATA=aloe-mono, persona rule, micro_layout row_per_rank); 357 steps, warmup 12, linear to 0 at 357, decoders x K 8; exports 60/119/178/238/298 node-local + export-357 on /data; world 64, ~12 s/step
- **Steps:** 0 -> 357 (target 357)
- **Evals:** norm_word_ce = 1.9156 @357 (step0 2.0112; 1.9574 under 23051's rule = 22317 identity); pose_cb0_ce = 1.1127 @357; pose_light_ce = 3.0469 @357
- **Checkpoints / exports:** export-60; export-119; export-178; export-238; export-298; export-357
- **Local paths:** /data/home/amartyavarma/ppm/aloe-pose-mono-8020/exports/export-357
- **HF repos:** [InternalCan/ppm-aloe-pose-mono-8020](https://huggingface.co/InternalCan/ppm-aloe-pose-mono-8020)
- **Deployed:** Playground (genweb) - monoplex 'mono8020 export-357' (former base weights, swappable) (down)
- **Notes:** Canonical monoplex model (export-357) served by the genweb playground; parent of the Rosie FT. Step 350 eval equals step 357 to 4 dp (log:798).
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020/README.md:1 -> PPM 32B monoplex pose FT on Aloe handover pose rows, 80/20`
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020/README.md:6 -> init PPM local-11488`
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020/README.md:13 -> 27,517 train items 3,033 h; 531 eval items`
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020/README.md:15 -> 22,825 rows`
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020/README.md:16 -> 357 steps at 64 rows/step`
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020/README.md:17 -> warmup 12, decay to 0 at 357`
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020/README.md:19 -> exports 60,119,178,238,298 + export-357`
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020/README.md:35 -> realised prompt fractions train`
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020/README.md:63 -> job 24272 launched 17:28 PT, world 64`
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020/README.md:68 -> step-0 1.9574 under 23051's rule = 22317 identity`
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020/RUNNING_JOB.txt:2 -> ended 2026-10-18 18:48:43 PT rc 0`
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020/logs/posemono8020-24272.log:157 -> init local-11488`
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020/logs/posemono8020-24272.log:227 -> step-0 word_ce_norm 2.0112, pose cb0 1.2780, light 4.0203`
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020/logs/posemono8020-24272.log:806 -> step-357 word_ce_norm 1.9156, pose cb0 1.1127, light 3.0469`
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020/exports/BACKUP_LOG.txt:39 -> HF has export-60..298 + export-357-dcp; export-357 shards not on HF; Weka export-357 kept`
  - `memory:monoplex-means-aloe-mono-rows.md:11 -> 'monoplex FT' = this recipe; export-357 serves the playground`

### mono8020-sam

**aloe-pose-mono-8020-sam: 24272 retrain on SAM-separated Mimi codes (prepared, not launched)**  
group `monoplex` · status **abandoned** · dir `/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam`

- **Jobs:** 30621 (SAM store build, in ppm/aloe-pose-ft-sam), 30624/30641 (CPU checks)
- **Dates:** prepared 2026-10-26 ~05:08 .. ~06:05 PT; never trained
- **Base model:** unknown
- **Init:** planned: PPM local-11488 (24272's)
- **Parents (resolved):** ppm32b@11488
- **Produces:** none recorded
- **Data:** ppm/aloe-pose-ft-sam/stores = 22091's stores with codes_assistant/codes_user swapped for SAM-separated isolated stems (HF InternalCan/stage2-sam-sep-v1-mimi @ecb72b2d, local copy); 17,872 rows, 0 missing; cb0 differs on 20.5% of trained B frames
- **Recipe:** 24272 exactly (8 keys differ: paths/tags); GO.sh would pause MoT, train 8 or 12 nodes, resume MoT
- **Steps:** none (planned 357)
- **Evals:** none recorded
- **Checkpoints / exports:** unknown
- **Local paths:** unknown
- **HF repos:** none named in run files
- **Notes:** No training log exists in the dir (logs/ holds only checks-30624/30641). The SAM-code monoplex was later trained from 8,1,1 17850 instead (mono17850sam). Planned HF repo InternalCan/ppm-aloe-pose-mono-8020-sam (STATUS.md:29) never created as far as files show.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/README_SAM.md:1 -> latest monoplex (24272) retrained on SAM codes`
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/README_SAM.md:5 -> recipe = 24272 exactly`
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/README_SAM.md:6 -> only data change = Mimi codes from ppm/aloe-pose-ft-sam/stores`
  - `/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/STATUS.md:3 -> READY except /scratch; needs main's OK, then user's go`
  - `/data/home/amartyavarma/ppm/aloe-pose-ft-sam/README.md:5 -> SAM codes source HF InternalCan/stage2-sam-sep-v1-mimi @ecb72b2d (local copy)`
  - `/data/home/amartyavarma/ppm/aloe-pose-ft-sam/README.md:12 -> 17,872 rows, 0 missing, 0 T mismatches`
  - `/data/home/amartyavarma/ppm/aloe-pose-ft-sam/README.md:26 -> code change fractions on trained B tracks`

### mono17850

**pose6 mono17850 (job 31240): monoplex 24272 recipe from 8,1,1 dcp-17850, raw codes**  
group `monoplex` · status **finished** · dir `/data/home/amartyavarma/ppm/pose6/mono17850`

- **Jobs:** 31198 (failed in 1 s: quoting bug), 31240, 31467 (HF push)
- **Dates:** 2026-10-27 16:29 .. 2026-10-27 17:49 PT
- **Base model:** unknown
- **Init:** 8,1,1 dcp-17850 exported to /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17850 (meta src hf://InternalCan/ppm-v1-c@c9b22137/dcp-17850)
- **Parents (resolved):** ppm811-append3@17850
- **Produces:** mono17850@357
- **Data:** = mono8020: Aloe handover pose rows ppm/aloe-pose-ft/stores (27,517 items, 357 x 64 rows), raw Mimi codes; 80/20 prompt rule
- **Recipe:** aloe-pose-mono-8020 (24272) EXACTLY with init local-17850 (old arch, no s2p/G0COND); decp_bb link on (alpha 1.0); saves none, final export only (export_steps []), ALOE_SKIP_FINAL_FULL=1; world 64 on 8 hosts
- **Steps:** 0 -> 357 (target 357)
- **Evals:** norm_word_ce = 1.8976 @357 (step0 1.9864); pose_cb0_ce = 1.0700 @357; pose_light_ce = 2.9462 @357
- **Checkpoints / exports:** export-357
- **Local paths:** /data/home/amartyavarma/ppm/pose6/mono17850/export-357
- **HF repos:** [InternalCan/ppm-pose6-mono17850-final](https://huggingface.co/InternalCan/ppm-pose6-mono17850-final)
- **Deployed:** Playground (genweb) - monoplex slot 'mono17850raw' (extra weights picker entry) (live)
- **Notes:** pose6 STATUS stamps these as '10-28 ... (UTC date)'; the PT date from RUNNING_JOB.txt is 10-27. The s2p/G0COND pose6 FT variant on ppm811 31101 was prepared in pose6 but not found launched.
- **Citations:**
  - `/data/home/amartyavarma/ppm/pose6/mono17850/LAUNCHED_JOB:2 -> 31240 10-27 16:29 PT, 8 hosts world 64, base local-17850, link on, saves none, exports final`
  - `/data/home/amartyavarma/ppm/pose6/mono17850/RUNNING_JOB.txt:1 -> started 2026-10-27 16:29:50 PT`
  - `/data/home/amartyavarma/ppm/pose6/mono17850/RUNNING_JOB.txt:2 -> ended 17:49:24 PT rc 0`
  - `/data/home/amartyavarma/ppm/pose6/STATUS.md:317 -> NEW KIT mono17850 = aloe-pose-mono-8020 (24272) EXACTLY, init 8,1,1 dcp-17850`
  - `/data/home/amartyavarma/ppm/pose6/STATUS.md:391 -> 31198 failed in 1 s (quoting bug)`
  - `/data/home/amartyavarma/ppm/pose6/STATUS.md:394 -> local-17850 verified; meta src hf://InternalCan/ppm-v1-c@c9b22137/dcp-17850`
  - `/data/home/amartyavarma/ppm/pose6/STATUS.md:460 -> export-357 COMPLETE on /data`
  - `/data/home/amartyavarma/ppm/pose6/STATUS.md:465 -> HF InternalCan/ppm-pose6-mono17850-final/export-357-dcp VERIFY PASS`
  - `/data/home/amartyavarma/ppm/pose6/mono17850/logs/ft-31240.log:193 -> cfg init local-17850, stores aloe-pose-ft/stores, steps 357, decp_bb_alpha 1.0`
  - `/data/home/amartyavarma/ppm/pose6/mono17850/logs/ft-31240.log:248 -> step-0 word_ce_norm 1.9864, pose cb0 2.2301, light 5.2313`
  - `/data/home/amartyavarma/ppm/pose6/mono17850/logs/ft-31240.log:816 -> step-357 word_ce_norm 1.8976, pose cb0 1.0700, light 2.9462`
  - `memory:playground-extra-mono-models.md:26 -> raw model live in playground, unit 31458 slinky-18`

### mono17850sam

**pose6 mono17850sam (job 31443): monoplex 24272 recipe from 8,1,1 dcp-17850, SAM codes**  
group `monoplex` · status **finished** · dir `/data/home/amartyavarma/ppm/pose6/mono17850sam`

- **Jobs:** 31443, 31587 (HF push)
- **Dates:** 2026-10-27 17:15 .. 2026-10-27 18:35 PT
- **Base model:** unknown
- **Init:** 8,1,1 dcp-17850 = /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17850
- **Parents (resolved):** ppm811-append3@17850
- **Produces:** mono17850sam@357
- **Data:** SAM-code stores ppm/aloe-pose-ft-sam/stores (22091's rows with SAM-separated Mimi codes; voice clips SAM codes too); otherwise = mono8020 items/rows/prompt (80/20)
- **Recipe:** mono17850 kit with paths/tag changed (8 cfg keys differ: control_dir, export, local_export_root, log_root, plan_dir, run_tag, save_root, stores); link on; saves none; world 64 on 8 hosts
- **Steps:** 0 -> 357 (target 357)
- **Evals:** norm_word_ce = 1.8994 @357 (step0 1.9848); pose_cb0_ce = 1.0700 @357; pose_light_ce = 2.9462 @357
- **Checkpoints / exports:** export-357
- **Local paths:** /data/home/amartyavarma/ppm/pose6/mono17850sam/export-357
- **HF repos:** [InternalCan/ppm-pose6-mono17850sam-final](https://huggingface.co/InternalCan/ppm-pose6-mono17850sam-final)
- **Deployed:** Playground (genweb) - monoplex slot 'mono17850sam' (extra weights picker entry) (live)
- **Notes:** Word CE eval here is on SAM-code eval items, so not strictly comparable to mono17850's raw-code eval. SAM model gets SAM data only at inference (5 actors).
- **Citations:**
  - `/data/home/amartyavarma/ppm/pose6/mono17850sam/LAUNCHED_JOB:1 -> 31443 10-27 17:15 PT, 8 hosts world 64, base local-17850, link on`
  - `/data/home/amartyavarma/ppm/pose6/mono17850sam/RUNNING_JOB.txt:2 -> ended 2026-10-27 18:35:45 PT rc 0`
  - `/data/home/amartyavarma/ppm/pose6/STATUS.md:443 -> user order: second FT mono17850sam`
  - `/data/home/amartyavarma/ppm/pose6/STATUS.md:454 -> resolved cfg vs 31240: 8 keys differ`
  - `/data/home/amartyavarma/ppm/pose6/STATUS.md:476 -> 31443 exit 0, export-357 COMPLETE`
  - `/data/home/amartyavarma/ppm/pose6/STATUS.md:477 -> verify_final PASS + HF InternalCan/ppm-pose6-mono17850sam-final/export-357-dcp VERIFY PASS`
  - `/data/home/amartyavarma/ppm/pose6/mono17850sam/logs/ft-31443.log:193 -> stores aloe-pose-ft-sam/stores, init local-17850`
  - `/data/home/amartyavarma/ppm/pose6/mono17850sam/logs/ft-31443.log:249 -> step-0 word_ce_norm 1.9848, pose cb0 2.2288, light 5.2326`
  - `/data/home/amartyavarma/ppm/pose6/mono17850sam/logs/ft-31443.log:817 -> step-357 word_ce_norm 1.8994, pose cb0 1.0700, light 2.9462`
  - `memory:playground-extra-mono-models.md:27 -> SAM model live in playground, unit 31459 slinky-10`

### aloe-pose-rosie

**aloe-pose-rosie: full FT of the monoplex model on Rosie, 2 epochs**  
group `monoplex` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-pose-rosie`

- **Jobs:** 30836 (exit 143), 30837 (exit 143), 30839 (exit 1), 30844 (exit 1), 30847
- **Dates:** 2026-10-27 ~02:00 .. 2026-10-27 02:40 PT (30847: 02:30:49 .. 02:40:16)
- **Base model:** unknown
- **Init:** monoplex export-357: /data/home/amartyavarma/ppm/aloe-pose-mono-8020/exports/export-357
- **Parents (resolved):** mono8020@357
- **Produces:** aloe-pose-rosie@18
- **Data:** Rosie items (items.json; ALOE_MONO_ITEMS): 176 train clips (858,606 frames = 19.1 h) -> 140 rows -> 18 steps x 8 rows (2 epochs), eval 4 clips; prompt = 24272's 80/20 rule
- **Recipe:** 24272 env verbatim except dirs, rows/step 8 (1 row/rank at world 8 on slinky-18), eval every 5, no intermediate exports, STOP-only saves, ALOE_EPOCHS=2, AdamW host offload; user: 'full finetune ... 2 epochs, not LoRA'
- **Steps:** 0 -> 18 (target 18)
- **Evals:** norm_word_ce = 1.8028 @18 (step0 1.8412; Rosie eval, 4 clips); pose_cb0_ce = 1.1879 @18; pose_light_ce = 3.6276 @18
- **Checkpoints / exports:** export-final (step 18)
- **Local paths:** /data/home/amartyavarma/ppm/aloe-pose-rosie/export-final
- **HF repos:** none named in run files
- **Deployed:** Playground (genweb) - monoplex 'Rosie FT' weights (paused)
- **Notes:** Tiny single-speaker FT; eval is 4 Rosie clips. No HF repo named in the dir.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-pose-rosie/rosie.env:147 -> user 10-27 'full finetune [the monoplex pose model] on rosie, 2 epochs, not LoRA'; 24272 env verbatim except listed`
  - `/data/home/amartyavarma/ppm/aloe-pose-rosie/rosie.env:150 -> ALOE_EPOCHS=2`
  - `/data/home/amartyavarma/ppm/aloe-pose-rosie/RUNNING_JOB.txt:1 -> 30847 slinky-18 started 2026-10-27 02:30:49 PT`
  - `/data/home/amartyavarma/ppm/aloe-pose-rosie/RUNNING_JOB.txt:2 -> ended 02:40:16 PT rc 0`
  - `/data/home/amartyavarma/ppm/aloe-pose-rosie/logs/rosie-30847.log:98 -> init mono8020 export-357, steps 18, rows_per_step 8, warmup 1`
  - `/data/home/amartyavarma/ppm/aloe-pose-rosie/logs/rosie-30847.log:166 -> 176 clips 19.1 h -> 140 rows -> 18 x 8`
  - `/data/home/amartyavarma/ppm/aloe-pose-rosie/logs/rosie-30847.log:176 -> step-0 word_ce_norm 1.8412, pose cb0 1.1940, light 3.6353`
  - `/data/home/amartyavarma/ppm/aloe-pose-rosie/logs/rosie-30847.log:226 -> step-18 word_ce_norm 1.8028, pose cb0 1.1879, light 3.6276`
  - `/data/home/amartyavarma/ppm/pose6/STATUS.md:130 -> 'Rosie export-18' referenced among deployed ckpts`

### poseduplex-v5

**Aloe POSE-DUPLEX (job 30074): SAM speech duplex + B pose, init PPMVOX v5**  
group `poseduplex` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex`

- **Jobs:** 30054 (terminated), 30072 (exit 2), 30074, 30215 (HF dcp upload)
- **Dates:** 2026-10-25 03:16 .. 2026-10-25 04:36 PT
- **Base model:** unknown
- **Init:** PPMVOX v5 final export w-18100, node-local subset /scratch/amartyavarma/poseduplex-init/w-18100 (ALOE_INIT_VOX=1: its own vadapter + vgain)
- **Parents (resolved):** ppmvox-v5@18100
- **Produces:** poseduplex-v5@498
- **Data:** s2h-m1-samsep stores (Aloe handover none-crossbleed first chunks, SAM-separated Mimi, Voxtral latents of partner on A) + B pose tokens from /data/home/amartyavarma/data/aloe-handover-full/DATA/files/none (B pose ok on 85.4% of train micros); plan 2275f1eab21f3326: 31,796 train micros = 2 epochs; eval 316 micros; prompt = 29345's (B clip 30%, no persona)
- **Recipe:** vox/run_vox.env (29345 = 28511 recipe, world 64 on 8x8) + channel-B pose exactly as 26089 ([POSE]); trainer code/train_samsep_posedup.py; 498 steps (QQA_STEPS knob = 2 epochs), warmup 16, linear decay to 0; decp_bb link ON + trained (POSEDUP_LINK=1); exports 83/165/248/333/415 + 498
- **Steps:** 0 -> 498 (target 498)
- **Evals:** norm_word_ce = 1.7876 @498 (step0 1.8715); pose_cb0_ce = 1.0695 @498 (1.0695 also @475); pose_light_ce = 2.9394 @498
- **Checkpoints / exports:** export-83; export-165; export-248; export-333; export-415; export-498; export-poseduplex-v5-498 (consolidated DCP)
- **Local paths:** /scratch/amartyavarma/aloe-posedup-live/export-498-dcp (slinky-10 serving copy; /data copy deleted)
- **HF repos:** [InternalCan/ppm-aloe-poseduplex-v5-final](https://huggingface.co/InternalCan/ppm-aloe-poseduplex-v5-final), [InternalCan/ppm-aloe-poseduplex-v5](https://huggingface.co/InternalCan/ppm-aloe-poseduplex-v5), [InternalCan/ppm-aloe-poseduplex-v5-498-state](https://huggingface.co/InternalCan/ppm-aloe-poseduplex-v5-498-state) *(name match)*
- **Deployed:** Playground (genweb) - duplex view 'Live mic' (only human-facing pose-duplex UI) (live); Pose-duplex backend instance P ('playground live', hidden lb entry slinky-18:9126) (paused); Playground (genweb) - duplex 'Generate both sides' worker (genweb-dup) (down); Pose-duplex live demo - public front + APIs (/ws, /api/*, /clipwav, /avatar/, /avatar3/) and backend instance 1 (live); Pose-duplex live demo PAGE (Aloe Pose-Duplex Live index.html) (destroyed)
- **Notes:** Parent run id for PPMVOX v5 guessed as 'ppmvox-v5' (w-18100 = its final export). Demo note: generated pose mostly static (memory posedup-live-demo.md:24).
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/LAUNCHED_JOB:1 -> job 30074, 8x8 world 64, init_tag w-18100, steps 498, 2026-10-25 03:16 PT`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/run_poseduplex.env:1 -> same as SAM speech duplex with pose on assistant side`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/run_poseduplex.env:3 -> init = PPMVOX v5 final export, ALOE_INIT_VOX=1`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/run_poseduplex.env:9 -> QQA_STEPS_SET 498 = 2 epochs`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/run_poseduplex.env:112 -> POSEDUP_LINK=1 decp_bb ON + trained`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/PHASE1.md:7 -> pose source path aloe-handover-full DATA/files/none`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/PHASE1.md:17 -> B pose coverage 13,578/15,898 micros = 85.4%`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/aloeposedup-30074.log:163 -> cfg init w-18100, steps 498, warmup 16`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/aloeposedup-30074.log:520 -> plan 2275f1eab21f3326, 31,796 micros, 498 x 64, eval 316, stores s2h-m1-samsep`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/aloeposedup-30074.log:537 -> step-0 word_ce_norm 1.8715, pose cb0 2.2318, light 5.1634`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/aloeposedup-30074.log:1182 -> step-498 word_ce_norm 1.7876, pose cb0 1.0695, light 2.9394`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/BACKUP_LOG.txt:1 -> export-83 to InternalCan/ppm-aloe-poseduplex-v5 VERIFY PASS`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/BACKUP_LOG.txt:95 -> export-498 -> InternalCan/ppm-aloe-poseduplex-v5-final/export-498 VERIFY PASS`
  - `/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/BACKUP_LOG.txt:97 -> export-poseduplex-v5-498 DCP -> -final/export-498-dcp VERIFY PASS (job 30215)`
  - `memory:posedup-live-demo.md:11 -> live pose-duplex demo serves job 30074 export-498`
  - `memory:posedup-live-demo.md:23 -> served from slinky-10 local copy; /data copy deleted`

### aloe-v2init-8869

**aloe-v2init-8869 (job 23508): aloe-v2 recipe on PPM w-8869**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-v2init-8869`

- **Jobs:** 23507 (cancelled), 23508
- **Dates:** 2026-10-18 14:03 .. 2026-10-18 14:30 PT
- **Base model:** unknown
- **Init:** PPM main w-8869, /data/home/amartyavarma/ppm/ckpt-restore/ppm-v1-w-8869 (consolidated bf16 DCP)
- **Parents (resolved):** ppm32b@8869
- **Produces:** aloe-v2init-8869@253
- **Data:** = aloe-v2 (22383): handover first chunks 8,067 train rows x 2 (16,134 micros), eval 162 x 2; plan ba2d70729ea437ab
- **Recipe:** exact copy of aloe-v2init recipe (aloe-v2 job 22383) with init w-8869; 253 steps, warmup 8, LR trunk 3e-6, world 64
- **Steps:** 0 -> 253 (target 253)
- **Evals:** norm_word_ce = 1.8442 @253 (step0 1.8888)
- **Checkpoints / exports:** export-42; export-84; export-126; export-169; export-211; export-253
- **Local paths:** /data/home/amartyavarma/ppm/aloe-v2init-8869/exports/export-253
- **HF repos:** [InternalCan/ppm-aloe-v2init-8869](https://huggingface.co/InternalCan/ppm-aloe-v2init-8869)
- **Notes:** Speech-only Aloe FT (no pose) - part of the init-comparison series (8869/10351/11488=aloe-v2-ft/12529).
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-v2init-8869/README.md:3 -> exact copy of aloe-v2init recipe (aloe-v2 job 22383), init PPM main w-8869`
  - `/data/home/amartyavarma/ppm/aloe-v2init-8869/README.md:4 -> partner run aloe-next-6242 (14B) at matched pretraining CE`
  - `/data/home/amartyavarma/ppm/aloe-v2init-8869/README.md:12 -> exports 42..253, HF backup, final export-253`
  - `/data/home/amartyavarma/ppm/aloe-v2init-8869/logs/aloev2i-8869-23508.log:1 -> job 23508 2026-10-18 14:03:00 PT`
  - `/data/home/amartyavarma/ppm/aloe-v2init-8869/logs/aloev2i-8869-23508.log:104 -> init ppm-v1-w-8869, 253 steps`
  - `/data/home/amartyavarma/ppm/aloe-v2init-8869/logs/aloev2i-8869-23508.log:122 -> step-0 word_ce_norm 1.8888`
  - `/data/home/amartyavarma/ppm/aloe-v2init-8869/logs/aloev2i-8869-23508.log:590 -> step-253 word_ce_norm 1.8442`
  - `/data/home/amartyavarma/ppm/aloe-v2init-8869/exports/BACKUP_LOG.txt:1 -> HF InternalCan/ppm-aloe-v2init-8869`

### aloe-v2init-10351

**aloe-v2init-10351 (job 22562): aloe-v2 recipe on PPM w-10351**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-v2init-10351`

- **Jobs:** 22562
- **Dates:** 2026-10-16 22:06 .. 2026-10-16 22:42 PT
- **Base model:** unknown
- **Init:** PPM main w-10351, /data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-10351
- **Parents (resolved):** ppm32b@10351
- **Produces:** aloe-v2init-10351@253
- **Data:** = aloe-v2 (22383) first chunks 8,067 x 2 micros, eval 162 x 2; plan ba2d70729ea437ab
- **Recipe:** aloe-v2 (22383) recipe, init w-10351; 253 steps, warmup 8, LR trunk 3e-6, world 64
- **Steps:** 0 -> 253 (target 253)
- **Evals:** norm_word_ce = 1.8351 @253 (step0 1.8797)
- **Checkpoints / exports:** export-253
- **Local paths:** unknown
- **HF repos:** [InternalCan/ppm-aloe-v2init-10351](https://huggingface.co/InternalCan/ppm-aloe-v2init-10351)
- **Notes:** Speech-only. Local export location not checked (no exports/ dir on /data).
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-v2init-10351/logs/aloev2i-10351-22562.log:1 -> job 22562 2026-10-16 22:06:40 PT`
  - `/data/home/amartyavarma/ppm/aloe-v2init-10351/logs/aloev2i-10351-22562.log:103 -> init w-10351, 253 steps`
  - `/data/home/amartyavarma/ppm/aloe-v2init-10351/logs/aloev2i-10351-22562.log:121 -> step-0 word_ce_norm 1.8797`
  - `/data/home/amartyavarma/ppm/aloe-v2init-10351/logs/aloev2i-10351-22562.log:590 -> step-253 word_ce_norm 1.8351`
  - `/data/home/amartyavarma/ppm/aloe-v2init-10351/logs/BACKUP_LOG.txt:1 -> HF InternalCan/ppm-aloe-v2init-10351`

### aloe-v2init-12529

**aloe-v2init-12529 (job 22575): aloe-v2 recipe on PPM 12529**  
group `aloe-speech-ft` · status **finished** · dir `/data/home/amartyavarma/ppm/aloe-v2init-12529`

- **Jobs:** 22564 (cancelled), 22575
- **Dates:** 2026-10-16 22:10 .. 2026-10-16 22:40 PT
- **Base model:** unknown
- **Init:** PPM main step 12529, /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-12529 (converted from node-local dcp save, ckpt_conv.sbatch)
- **Parents (resolved):** ppm32b@12529
- **Produces:** aloe-v2init-12529@253
- **Data:** = aloe-v2 (22383) first chunks 8,067 x 2 micros, eval 162 x 2; plan ba2d70729ea437ab
- **Recipe:** aloe-v2 (22383) recipe, init local-12529; 253 steps, warmup 8, LR trunk 3e-6, world 64
- **Steps:** 0 -> 253 (target 253)
- **Evals:** norm_word_ce = 1.8230 @253 (step0 1.8665)
- **Checkpoints / exports:** export-253
- **Local paths:** unknown
- **HF repos:** [InternalCan/ppm-aloe-v2init-12529](https://huggingface.co/InternalCan/ppm-aloe-v2init-12529)
- **Notes:** Speech-only.
- **Citations:**
  - `/data/home/amartyavarma/ppm/aloe-v2init-12529/logs/aloev2i-12529-22575.log:1 -> job 22575 2026-10-16 22:10:03 PT`
  - `/data/home/amartyavarma/ppm/aloe-v2init-12529/logs/aloev2i-12529-22575.log:114 -> init local-12529, 253 steps`
  - `/data/home/amartyavarma/ppm/aloe-v2init-12529/logs/aloev2i-12529-22575.log:132 -> step-0 word_ce_norm 1.8665`
  - `/data/home/amartyavarma/ppm/aloe-v2init-12529/logs/aloev2i-12529-22575.log:601 -> step-253 word_ce_norm 1.8230`
  - `/data/home/amartyavarma/ppm/aloe-v2init-12529/logs/BACKUP_LOG.txt:1 -> HF InternalCan/ppm-aloe-v2init-12529`
  - `/data/home/amartyavarma/ppm/aloe-v2init-8869/README.md:5 -> ckpt_conv.sbatch was for 12529's node-local dcp save`

### sam8b-aloe-ft

**sam-aloe-ft (job 19440): SAM 8B pose model FT on Aloe pose (first sam-cache)**  
group `aloe-pose-ft` · status **finished** · dir `/data/home/amartyavarma/sam-aloe-ft`

- **Jobs:** 19440
- **Dates:** 2026-10-12 19:53 .. 2026-10-12 21:13 PT
- **Base model:** unknown
- **Init:** SAM 8B pose model step 61258 (fp32): /data/home/amartyavarma/sam-aloe-ft/init-step-61258-fp32
- **Parents (resolved):** ext:sam8b-pose@61258
- **Produces:** sam8b-aloe-ft@1132
- **Data:** Aloe pose sam-cache (/data/home/amartyavarma/aloe-pose/sam-cache/train-*), built from gs://aloe-processed-data/pose-speech-vad clean selection
- **Recipe:** world 24 (slinky-7/15/18), 1132 steps, lr 1e-4 (light/new/pose_light 2e-4), warmup 50, k_pose 8
- **Steps:** 0 -> 1132 (target 1132)
- **Evals:** pose_cb0_ce = 1.7752 heldout @1132 (eval split 1.7681; step0 heldout 1.9103); pose_light_ce = 3.8048 heldout @1132 (eval 3.7509)
- **Checkpoints / exports:** export-step-1132
- **Local paths:** /data/home/amartyavarma/sam-aloe-ft/export-step-1132
- **HF repos:** [InternalCan/sam-pose-aloe-step1132](https://huggingface.co/InternalCan/sam-pose-aloe-step1132) *(name match)*
- **Notes:** NOT a 32B FT: base is the SAM 8B pose model (canonical id 'sam8b-pose@61258' is a guess). HF repo for step 1132 not verified (only the naming template). /data/home/amartyavarma/aloe-pose is the data dir (no training run).
- **Citations:**
  - `/data/home/amartyavarma/sam-aloe-ft/logs/run.json:2 -> world 24`
  - `/data/home/amartyavarma/sam-aloe-ft/logs/run.json:3 -> total_steps 1132`
  - `/data/home/amartyavarma/sam-aloe-ft/logs/run.json:4 -> lr 1e-4`
  - `/data/home/amartyavarma/sam-aloe-ft/logs/run.json:17 -> warmup 50`
  - `/data/home/amartyavarma/sam-aloe-ft/logs/run.json:18 -> stage1 init-step-61258-fp32`
  - `/data/home/amartyavarma/sam-aloe-ft/logs/run.json:19 -> data aloe-pose/sam-cache/train-*`
  - `/data/home/amartyavarma/sam-aloe-ft/logs/train-19440.log:1 -> job 19440 2026-10-12 19:53:12 PT`
  - `/data/home/amartyavarma/sam-aloe-ft/logs/log.jsonl:2 -> step-0 heldout pose_cb0 1.9103, light 4.1909`
  - `/data/home/amartyavarma/sam-aloe-ft/logs/log.jsonl:1157 -> step-1132 eval pose_cb0 1.7681, light 3.7509`
  - `/data/home/amartyavarma/sam-aloe-ft/logs/log.jsonl:1158 -> step-1132 heldout pose_cb0 1.7752, light 3.8048`
  - `/data/home/amartyavarma/sam-aloe-ft/hf/hf.sbatch:8 -> HF naming template InternalCan/sam-pose-aloe-step<N>`
  - `/data/home/amartyavarma/aloe-pose/README.md:3 -> FT target is the SAM 8B pose model; data sam-cache/`

### sam8b-aloe-active

**sam-aloe-ft-active (job 19623): SAM 8B FT on active-speaker Aloe cache**  
group `aloe-pose-ft` · status **finished** · dir `/data/home/amartyavarma/sam-aloe-ft-active`

- **Jobs:** 19623, 19620 (init download), 19661/19662/19689 (eval-only)
- **Dates:** 2026-10-12 22:24 .. 2026-10-12 23:06 PT
- **Base model:** unknown
- **Init:** SAM 8B pose step 61258 fp32 (/scratch/amartyavarma/sam-aloe-ft-active/init-step-61258-fp32)
- **Parents (resolved):** ext:sam8b-pose@61258
- **Produces:** sam8b-aloe-active@636
- **Data:** /data/home/amartyavarma/aloe-pose/sam-cache-active/train-*
- **Recipe:** world 24, 636 steps, lr 1e-4, warmup 50
- **Steps:** 0 -> 636 (target 636)
- **Evals:** pose_cb0_ce = 2.1434 heldout @636 (eval 2.1292); pose_light_ce = 4.1227 heldout @636 (eval 4.0904)
- **Checkpoints / exports:** export-step-636
- **Local paths:** /data/home/amartyavarma/sam-aloe-ft-active/export-step-636
- **HF repos:** [InternalCan/sam-pose-aloe-active-step636](https://huggingface.co/InternalCan/sam-pose-aloe-active-step636)
- **Notes:** SAM 8B, not 32B. Held-out sets differ between sam-cache versions, so CE is not comparable across these rows.
- **Citations:**
  - `/data/home/amartyavarma/sam-aloe-ft-active/logs/run.json:3 -> total_steps 636`
  - `/data/home/amartyavarma/sam-aloe-ft-active/logs/run.json:20 -> init init-step-61258-fp32`
  - `/data/home/amartyavarma/sam-aloe-ft-active/logs/run.json:23 -> data sam-cache-active`
  - `/data/home/amartyavarma/sam-aloe-ft-active/logs/train-19623.log:1 -> job 19623 2026-10-12 22:24:29 PT`
  - `/data/home/amartyavarma/sam-aloe-ft-active/logs/log.jsonl:651 -> step-636 eval pose_cb0 2.1292, light 4.0904`
  - `/data/home/amartyavarma/sam-aloe-ft-active/logs/log.jsonl:652 -> step-636 heldout pose_cb0 2.1434, light 4.1227`
  - `/data/home/amartyavarma/sam-aloe-ft-active/hf/hf.sbatch:8 -> HF template InternalCan/sam-pose-aloe-active-step<N>`

### sam8b-aloe-posev2

**sam-aloe-ft-posev2 (job 19949): SAM 8B FT on Aloe pose v2 cache**  
group `aloe-pose-ft` · status **finished** · dir `/data/home/amartyavarma/sam-aloe-ft-posev2`

- **Jobs:** 19934 (aborted 4-node), 19929 (init), 19949
- **Dates:** 2026-10-13 05:41 .. 2026-10-13 06:22 PT
- **Base model:** unknown
- **Init:** SAM 8B pose step 61258 fp32 (/scratch/amartyavarma/sam-aloe-ft-posev2/init-step-61258-fp32)
- **Parents (resolved):** ext:sam8b-pose@61258
- **Produces:** sam8b-aloe-posev2@636
- **Data:** /data/home/amartyavarma/aloe-pose/sam-cache-active-posev2/train-*
- **Recipe:** world 24, 636 steps, lr 1e-4, warmup 50
- **Steps:** 0 -> 636 (target 636)
- **Evals:** pose_cb0_ce = 1.8465 heldout @636 (eval 1.8117); pose_light_ce = 3.8981 heldout @636 (eval 3.8517)
- **Checkpoints / exports:** export-step-636
- **Local paths:** /data/home/amartyavarma/sam-aloe-ft-posev2/export-step-636
- **HF repos:** [InternalCan/sam-pose-aloe-posev2-step636](https://huggingface.co/InternalCan/sam-pose-aloe-posev2-step636) *(name match)*
- **Notes:** SAM 8B. HF repo name only as template.
- **Citations:**
  - `/data/home/amartyavarma/sam-aloe-ft-posev2/logs/run.json:3 -> total_steps 636`
  - `/data/home/amartyavarma/sam-aloe-ft-posev2/logs/run.json:23 -> data sam-cache-active-posev2`
  - `/data/home/amartyavarma/sam-aloe-ft-posev2/logs/train-19949.log:1 -> job 19949 2026-10-13 05:41:29 PT`
  - `/data/home/amartyavarma/sam-aloe-ft-posev2/logs/log.jsonl:652 -> step-636 heldout pose_cb0 1.8465, light 3.8981`
  - `/data/home/amartyavarma/sam-aloe-ft-posev2/hf/hf.sbatch:8 -> HF template InternalCan/sam-pose-aloe-posev2-step<N>`

### sam8b-aloe-posev3

**sam-aloe-ft-posev3 (job 20003): SAM 8B FT on Aloe pose v3 cache**  
group `aloe-pose-ft` · status **finished** · dir `/data/home/amartyavarma/sam-aloe-ft-posev3`

- **Jobs:** 20003, 20004 (prune)
- **Dates:** 2026-10-13 09:32 .. 2026-10-13 10:14 PT
- **Base model:** unknown
- **Init:** SAM 8B pose step 61258 fp32 (posev2's init copy)
- **Parents (resolved):** ext:sam8b-pose@61258
- **Produces:** sam8b-aloe-posev3@636
- **Data:** /data/home/amartyavarma/aloe-pose/sam-cache-active-posev3/train-*
- **Recipe:** world 24, 636 steps, lr 1e-4, warmup 50
- **Steps:** 0 -> 636 (target 636)
- **Evals:** pose_cb0_ce = 1.6305 heldout @636 (eval 1.6213); pose_light_ce = 3.6700 heldout @636 (eval 3.6476)
- **Checkpoints / exports:** export-step-636
- **Local paths:** /data/home/amartyavarma/sam-aloe-ft-posev3/export-step-636
- **HF repos:** [InternalCan/sam-pose-aloe-posev3-step636](https://huggingface.co/InternalCan/sam-pose-aloe-posev3-step636)
- **Notes:** SAM 8B. HF repo name taken from other runs' hf files (sam-aloe-ft-v2/hf mentions InternalCan/sam-pose-aloe-posev3-step636).
- **Citations:**
  - `/data/home/amartyavarma/sam-aloe-ft-posev3/logs/run.json:20 -> init posev2 init-step-61258-fp32`
  - `/data/home/amartyavarma/sam-aloe-ft-posev3/logs/run.json:23 -> data sam-cache-active-posev3`
  - `/data/home/amartyavarma/sam-aloe-ft-posev3/logs/train-20003.log:1 -> job 20003 2026-10-13 09:32:36 PT`
  - `/data/home/amartyavarma/sam-aloe-ft-posev3/logs/log.jsonl:652 -> step-636 heldout pose_cb0 1.6305, light 3.6700`
  - `/data/home/amartyavarma/sam-aloe-ft-v2/logs/STATUS-v2-final.md:7 -> posev3 final held-out pose_cb0 1.6305, light 3.6700`

### sam8b-aloe-v2

**sam-aloe-ft-v2 (job 20630): SAM 8B FT on Aloe pose release v2 cache**  
group `aloe-pose-ft` · status **finished** · dir `/data/home/amartyavarma/sam-aloe-ft-v2`

- **Jobs:** 20630, 20631 (prune), 20697 (HF upload), 20700 (samples)
- **Dates:** 2026-10-14 11:12 .. 2026-10-14 13:48 PT
- **Base model:** unknown
- **Init:** SAM 8B pose step 61258 fp32 (posev2's init copy)
- **Parents (resolved):** ext:sam8b-pose@61258
- **Produces:** sam8b-aloe-v2@636
- **Data:** /data/home/amartyavarma/aloe-pose/sam-cache-active-v2/train-* (release v2, sliding-window Mimi codes)
- **Recipe:** posev3 recipe, 1 epoch ~636 steps, world 24, lr 1e-4, warmup 50
- **Steps:** 0 -> 636 (target 636)
- **Evals:** pose_cb0_ce = 1.6305 heldout @636 (eval 1.6218); pose_light_ce = 3.6700 heldout @636 (eval 3.6476); other = audio_cb0 heldout 1.5239 @636
- **Checkpoints / exports:** export-step-636
- **Local paths:** /data/home/amartyavarma/sam-aloe-ft-v2/export-step-636
- **HF repos:** [InternalCan/sam-pose-aloe-v2-step636](https://huggingface.co/InternalCan/sam-pose-aloe-v2-step636)
- **Notes:** SAM 8B. Pose data identical to posev3 (STATUS-v2-final.md:10), hence near-identical pose CE.
- **Citations:**
  - `/data/home/amartyavarma/sam-aloe-ft-v2/logs/run.json:23 -> data sam-cache-active-v2`
  - `/data/home/amartyavarma/sam-aloe-ft-v2/logs/train-20630.log:1 -> job 20630 2026-10-14 11:12:19 PT`
  - `/data/home/amartyavarma/sam-aloe-ft-v2/logs/log.jsonl:652 -> step-636 heldout pose_cb0 1.6305, light 3.6700`
  - `/data/home/amartyavarma/sam-aloe-ft-v2/logs/STATUS-v2-final.md:4 -> HF private InternalCan/sam-pose-aloe-v2-step636 verified`
  - `/data/home/amartyavarma/sam-aloe-ft-v2/logs/STATUS-v2-final.md:7 -> final held-out pose 1.6305/3.6700, audio_cb0 1.5239`
  - `/data/home/amartyavarma/sam-aloe-ft-v2/logs/STATUS-v2-final.md:18 -> chained prune 20631, HF 20697, samples 20700`

### sam8b-aloe-v2pure

**sam-aloe-ft-v2pure (job 20759): SAM 8B FT on Aloe release v2 'v2pure' cache**  
group `aloe-pose-ft` · status **finished** · dir `/data/home/amartyavarma/sam-aloe-ft-v2pure`

- **Jobs:** 20759, 20760 (prune), 20758 (weka clean)
- **Dates:** 2026-10-14 14:13 .. 2026-10-14 14:55 PT
- **Base model:** unknown
- **Init:** SAM 8B pose step 61258 fp32 (posev2's init copy)
- **Parents (resolved):** ext:sam8b-pose@61258
- **Produces:** sam8b-aloe-v2pure@636
- **Data:** /data/home/amartyavarma/aloe-pose/sam-cache-active-v2pure/train-* (the cache the 32B v2pure stores were later built from)
- **Recipe:** world 24, 636 steps, lr 1e-4, warmup 50
- **Steps:** 0 -> 636 (target 636)
- **Evals:** pose_cb0_ce = 1.6305 heldout @636 (eval 1.6219); pose_light_ce = 3.6700 heldout @636 (eval 3.6476)
- **Checkpoints / exports:** export-step-636
- **Local paths:** /data/home/amartyavarma/sam-aloe-ft-v2pure/export-step-636
- **HF repos:** [InternalCan/sam-pose-aloe-v2pure-step636](https://huggingface.co/InternalCan/sam-pose-aloe-v2pure-step636) *(name match)*
- **Notes:** SAM 8B. HF repo only as template (likely InternalCan/sam-pose-aloe-v2pure-step636, unverified).
- **Citations:**
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure/logs/run.json:23 -> data sam-cache-active-v2pure`
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure/logs/train-20759.log:1 -> job 20759 2026-10-14 14:13:14 PT`
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure/logs/log.jsonl:652 -> step-636 heldout pose_cb0 1.6305, light 3.6700`
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure/hf/hf.sbatch:8 -> HF template InternalCan/sam-pose-aloe-v2pure-step<N>`
  - `/data/home/amartyavarma/ppm/aloe-v2pure-ft/README.md:7 -> 32B v2pure stores built from sam-cache-active-v2pure, the cache this 8B trained on`

### sam8b-v2pure-cmp

**sam-aloe-ft-v2pure-cmp (job 26141): SAM 8B v2pure FT comparable with PPM 26089**  
group `aloe-pose-ft` · status **finished** · dir `/data/home/amartyavarma/sam-aloe-ft-v2pure-cmp`

- **Jobs:** 26121 (aborted), 26141, 26142 (HF)
- **Dates:** 2026-10-20 17:00 .. 2026-10-20 17:29 PT
- **Base model:** unknown
- **Init:** SAM 8B pose export-step-61258: /data/home/amartyavarma/qwen-1ch/infer/sam_pose/export-step-61258/model.safetensors
- **Parents (resolved):** ext:sam8b-pose@61258
- **Produces:** sam8b-v2pure-cmp@200
- **Data:** Aloe v2pure, plan /data/home/amartyavarma/sam-aloe-ft-v2pure-cmp/plan/plan.json (matched to 26089's 200 x 64-row schedule)
- **Recipe:** world 64 (8 nodes), 200 steps, lr 1e-4, warmup 7; code/train/sam_cmp.env + train_pose_cmp.py; bf16
- **Steps:** 0 -> 200 (target 200)
- **Evals:** pose_cb0_ce = 1.6541 heldout @200 (step0 1.7254); pose_light_ce = 3.6958 heldout @200
- **Checkpoints / exports:** export-step-200
- **Local paths:** /data/home/amartyavarma/sam-aloe-ft-v2pure-cmp/export-step-200
- **HF repos:** [InternalCan/sam-pose-aloe-v2pure-cmp-step200](https://huggingface.co/InternalCan/sam-pose-aloe-v2pure-cmp-step200)
- **Notes:** SAM 8B reference for the 32B v2pure series (same 200-step schedule); held-out set = whether identical to 26089's 3,481 clips not verified.
- **Citations:**
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure-cmp/run.sbatch:4 -> SAM 8B pose FT on Aloe v2pure, comparable with PPM 26089`
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure-cmp/RUNNING_JOB.txt:1 -> 26141 started 2026-10-20 17:00:35 PT`
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure-cmp/RUNNING_JOB.txt:2 -> ended 17:29:48 PT rc 0`
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure-cmp/logs/run.json:2 -> world 64`
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure-cmp/logs/run.json:3 -> steps 200`
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure-cmp/logs/run.json:4 -> lr 1e-4`
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure-cmp/logs/run.json:17 -> warmup 7`
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure-cmp/logs/run.json:19 -> init sam_pose export-step-61258`
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure-cmp/logs/log.jsonl:1 -> step-0 heldout pose_cb0 1.7254, light 3.8684`
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure-cmp/logs/log.jsonl:209 -> step-200 heldout pose_cb0 1.6541, light 3.6958`
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure-cmp/HF_BACKUP.txt:1 -> hf://InternalCan/sam-pose-aloe-v2pure-cmp-step200 verified`

### sam8b-v2pure-cmp-lr3

**sam-aloe-ft-v2pure-cmp-lr3 (job 26144): same as 26141 at LR/3**  
group `aloe-pose-ft` · status **finished** · dir `/data/home/amartyavarma/sam-aloe-ft-v2pure-cmp-lr3`

- **Jobs:** 26144
- **Dates:** 2026-10-20 17:30 .. 2026-10-20 18:02 PT
- **Base model:** unknown
- **Init:** SAM 8B pose export-step-61258 (/data/home/amartyavarma/qwen-1ch/infer/sam_pose/export-step-61258/model.safetensors)
- **Parents (resolved):** ext:sam8b-pose@61258
- **Produces:** sam8b-v2pure-cmp-lr3@200
- **Data:** Aloe v2pure, plan /data/home/amartyavarma/sam-aloe-ft-v2pure-cmp-lr3/plan/plan.json
- **Recipe:** = sam-aloe-ft-v2pure-cmp with lr 3.333e-5; world 64, 200 steps, warmup 7
- **Steps:** 0 -> 200 (target 200)
- **Evals:** pose_cb0_ce = 1.6611 heldout @200 (step0 1.7254); pose_light_ce = 3.7042 heldout @200
- **Checkpoints / exports:** export-step-200
- **Local paths:** /data/home/amartyavarma/sam-aloe-ft-v2pure-cmp-lr3/export-step-200
- **HF repos:** [InternalCan/sam-pose-aloe-v2pure-cmp-lr3-exports](https://huggingface.co/InternalCan/sam-pose-aloe-v2pure-cmp-lr3-exports) *(name match)*
- **Notes:** SAM 8B. No HF backup file in the dir.
- **Citations:**
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure-cmp-lr3/run.sbatch:4 -> SAM 8B v2pure FT comparable with PPM 26089`
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure-cmp-lr3/RUNNING_JOB.txt:1 -> 26144 started 2026-10-20 17:30:46 PT`
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure-cmp-lr3/RUNNING_JOB.txt:2 -> ended 18:02:23 PT rc 0`
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure-cmp-lr3/logs/run.json:4 -> lr 3.333e-05`
  - `/data/home/amartyavarma/sam-aloe-ft-v2pure-cmp-lr3/logs/log.jsonl:209 -> step-200 heldout pose_cb0 1.6611, light 3.7042`

## Deployments

As audited 2026-10-27 20:14 PT (read-only: squeue + live-state files; nothing was probed). Detail and citations: runs.json `deployments`, and [LINEAGE.md](LINEAGE.md#deployed-where).

| deployment | status | model | checkpoint | lineage node | jobs | hosts / GPUs |
|---|---|---|---|---|---|---|
| Playground (genweb) - monoplex slot 'base' (default) | live | pose 32B Xabi-data FT (v2pure, from 8,1,1 17850), export-200 | export-200 (DCP) at /data/home/amartyavarma/ppm/aloe-v2pure-main-17850/export-200 | xabi-v2pure-17850@200 | 31177, 31637, 31640 | slinky-4 GPUs 4,5 (31177 genweb-u-hot4b, + GPU-5 fallback render lane); slinky-… |
| Playground (genweb) - monoplex slot 'mono17850raw' (extra weights picker entry) | live | monoplex 17850 raw (pose6 mono17850, raw Mimi codes, mono8020/24272 recipe from… | export-357 at /data/home/amartyavarma/ppm/pose6/mono17850/export-357 | mono17850@357 | 31458, 31639 | slinky-18 GPUs 0,1 (31458 genweb-mono17850raw); slinky-18 GPUs 3,4 (31639 genwe… |
| Playground (genweb) - monoplex slot 'mono17850sam' (extra weights picker entry) | live | monoplex 17850 SAM (pose6 mono17850sam, SAM-separated codes, from 8,1,1 17850),… | export-357 at /data/home/amartyavarma/ppm/pose6/mono17850sam/export-357 | mono17850sam@357 | 31459, 31638 | slinky-10 GPUs 0,1 (31459 genweb-mono17850sam); slinky-10 GPUs 5,6 (31638 genwe… |
| Playground (genweb) - monoplex 'mono8020 export-357' (former base weights, swappable) | down | Aloe pose monoplex mono-8020 FT, export-357 (original playground model 10-25..1… | export-357 at /data/home/amartyavarma/ppm/aloe-pose-mono-8020/exports/export-357 | mono8020@357 |  | none (not loaded) |
| Playground (genweb) - monoplex 'Rosie FT' weights | paused | Rosie pose full-FT (aloe-pose-rosie, on her 9.5 h), export-18 | export-18 at /data/home/amartyavarma/ppm/aloe-pose-rosie/exports-local/export-18 | rosie-pose-ft@18 | 30853 (cancelled 10-27 18:27 UTC) | was slinky-18 (GPU pair unknown); none now |
| Playground (genweb) - duplex view 'Live mic' (only human-facing pose-duplex UI) | live | Aloe pose-duplex FT v5 (job 30074, run ppm-aloe-poseduplex-v5-j30074), export-4… | export-498 consolidated DCP; served now from /data/home/amartyavarma/ppm/aloe-posedup-liv… | poseduplex-v5@498 | 30738, 30739, 30939, 29747 | model: slinky-4 GPUs 0,1 (30738 posedup-lb-S4, port 9136); front: 30739 posedup… |
| Pose-duplex backend instance P ('playground live', hidden lb entry slinky-18:9126) | paused | Aloe pose-duplex FT v5, export-498 | /scratch/amartyavarma/aloe-posedup-live/export-498-dcp (slinky-18 local copy) | poseduplex-v5@498 | 30849 (cancelled 10-27 18:27 UTC), 3077… | was slinky-18 GPUs 2-3 (per PLAN-motasr-draft.md:43); none now |
| Playground (genweb) - duplex 'Generate both sides' worker (genweb-dup) | down | Aloe pose-duplex FT v5, export-498 (two LivePPM states, duo mode) | /scratch/amartyavarma/aloe-posedup-live/export-498-dcp (slinky-18/-10 node-local) | poseduplex-v5@498 | 30831 (cancelled 10-27 02:21 UTC), 3077… | was slinky-18 GPUs 2,3; none now |
| Playground (genweb) - front/app/tunnel + render/record lanes (infrastructure) | live | n/a (serves the slots above; video = videort SeqEngine + Param student EMA) | n/a | n/a | 30672, 30452, 30673, 30613, 30939, 3163… | 30672 genweb-app slinky-18 (front :9205 -> app :9207); 30452 genweb-web slinky-… |
| Pose-duplex live demo - public front + APIs (/ws, /api/*, /clipwav, /avatar/, /avatar3/)… | live | Aloe pose-duplex FT v5, export-498 | /data/home/amartyavarma/ppm/aloe-posedup-live/ck/export-498-dcp | poseduplex-v5@498 | 30738, 30739 | 30738 posedup-lb-S4 slinky-4 GPUs 0,1 (:9136); 30739 posedup-front-s4 CPU-only… |
| Pose-duplex live demo PAGE (Aloe Pose-Duplex Live index.html) | destroyed | n/a (page only; backend export-498 lives on, see APIs row) | n/a | poseduplex-v5@498 | 30739 | n/a |
| Speech twin of the pose demo (ppm/aloe-m1vox-live server_twin.py, front :8935) | paused | Aloe FT (no persona) on PPM-Voxtral vox17127, export-249 (SAM-sep), actor 06 Ka… | export-249 at /data/home/amartyavarma/ppm/aloe-ft-32-samsep/vox/export-vox17127-249 | aloe-vox17127@249 | 30766, 31154 (cancelled 10-27 21:51 UTC… | front 30766 speechtwin-front on slinky-4 (CPU); backend was slinky-4 GPUs 4,5 (… |
| Aloe speech demo 10-24 clone (ppm/aloe-m1vox-live-1024, front :8905) - model 'aloe' (defa… | paused | Aloe FT (no persona) on PPM-Voxtral vox17127, export-249 | /data/home/amartyavarma/ppm/aloe-ft-32-samsep/vox/export-vox17127-249 | aloe-vox17127@249 | 30818, 30820 (cancelled 10-27 18:27 UTC) | front 30818 clone1024-front CPU on slinky-18; backend was slinky-10 GPUs 6,7 (:… |
| Aloe speech demo 10-24 clone - model 'persona' | paused | Aloe FT + persona prose on PPM-Voxtral (vox17127-persona), export-249 | /data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona/export-vox17127-persona-249 | aloe-vox17127-persona@249 | 30851 (cancelled 10-27 13:42 UTC) | was slinky-18 GPUs 6,7 (:8946) |
| Aloe speech demo 10-24 clone - model 'rosie' | paused | Rosie voice FT (e2, export-48) merged on vox17127 | /scratch/amartyavarma/aloe-demo-rosie-48/merged-48 (slinky-18 local) | rosie-voice-ft@48 | 30852 (cancelled 10-27 18:27 UTC), 3081… | was slinky-18 GPUs 0,1 (:8966) |
| Aloe speech demo 10-24 clone - model 'brianna' (removed from picker) | down | Brianna voice FT (A-10: tower/light/audio/proj + LoRA r4 merged onto vox17127) | /scratch/amartyavarma/aloe-demo-brianna-A10/merged-A10 (slinky-10 local) | brianna-voice-ft-A10@10 | 30819 (stopped 10-27 ~01:42 UTC) | was slinky-10 GPUs 4,5 |
| Aloe speech w-15973 site (ppm/aloe-m1vox-live-w15973, front :8925, one model) | paused | Aloe v2vox (regular codes) on PPM w-15973 + ASR adapter, export-253 (FT job 262… | /scratch/amartyavarma/aloe-w15973/export-253 (slinky-10 local) | aloe-v2vox-asr-w15973@253 | 30835, 30951 (cancelled 10-27 18:28 UTC… | front 30835 w15973-front CPU on slinky-10; backend was slinky-10 GPUs 0,1 (:899… |
| Original Aloe speech live demo (ppm/aloe-m1vox-live, 4-model picker: Brianna / Rosie / Al… | down | Brianna voice FT A-10 (default) + Rosie voice FT e2/48 + Aloe FT vox17127 expor… | merged-A10, merged-48, aloe-ft-32-samsep/vox/export-vox17127-249, aloe-ft-32-samsep/perso… | aloe-vox17127@249 (+ brianna-voice-ft-A10@10, rosie-voice-ft@48, aloe-vox17127-persona@249) |  | none (was slinky-10 x3, slinky-18 x4, slinky-4 GPUs 0-5 x3) |
| MoT 14B live demo (ppm/mot/live, motlive) | paused | MoT 14B (INTL layout + 3 Qwen3-14B experts) step 13814, streamed from HF | w-13814 via hf-stream://InternalCan/ppm-mot-14b/w-13814 (never on disk) | mot@13814 | 30829 (cancelled 10-27 18:28 UTC) | was slinky-10 GPUs 4,5; none now |
| videort DMD real-time video avatar (video/dmd-rt; front 1 + live avatar backend + record… | live | NOT OURS: Vansh's universal-bimodal sequential sampler (unmodified) + Param's D… | /scratch/amartyavarma/videort/hf/step-002911/model_ema.safetensors | ext:param-dmd-student@2911-ema | 29747, 30939, 31635, 31636, 31641, 31642 | 29747 videort-front CPU slinky-10 (:8960, feed :8990); 30939 genweb-avhybrid sl… |
| klweb live curve pages (setup-logs/klweb; stage3_ppm811.html etc.) | live | n/a | n/a | n/a | 28811, 31096, 28843, 28844, 28845, 2884… | CPU only: 28811 klweb-ppm (serve + tunnel) slinky-10; 31096 klweb-ppm811-refres… |
| Legacy demos (DEMOS_STOPPED_1019 / DEMOS_PAUSED_1020 / DEMOS_STOPPED_1021: m1vox-live x3,… | down | various (aloe-v2vox-raw w-14503/w-11488 export-253, nextvox-6242 export-253, SA… | see DEMOS_PAUSED_1020.txt per job | unknown (multiple) |  | none |

## HF model repos not mapped to a run

94 InternalCan model repos created since 2026-09-25 are not named by any run row (many belong to other team members' projects). Source: `sources/hf_models_InternalCan.json` (metadata-only `list_models(author='InternalCan')`, 768 repos total).

`aloe-studio-richmond-fast-teacher-dmd-2-epochs`, `bimodal-resolver-gt-pose-v3-step2000`, `bimodal-stabilizer-full-gt-pose-v3-step1200`, `bimodal-stabilizer-no-gt-pose-v3-step1200`, `bimodal-teacher-pose-v3-step200`, `conversational-eval-judge`, `dense35-devon-rick-421-lora-final`, `dense35-prefixed-aloe-all-lora-final`, `dryrun-ppmdecay-hfb-1023`, `dual-experiments-pretrain-all-ckpts`, `duplex-demo`, `facial-extractor-finetuned`, `flash-next-r197-podcast-50k-checkpoints`, `full-gt-nine-ref-endpoint-unguided-20261020`, `hera-42b-podcast-intelligence-high-200h`, `hera-42b-podcast-intelligence-low-200h`, `hera-42b-podcast-substance-high-200h`, `hera-42b-podcast-substance-low-200h`, `hera-42b-stitched-aloe-intelligence-high-200h`, `hera-42b-stitched-aloe-intelligence-low-200h`, `hera-42b-stitched-aloe-substance-high-200h`, `hera-42b-stitched-aloe-substance-low-200h`, `hera-42b-synthetic-dumb-intelligence-high-200h`, `hera-42b-synthetic-dumb-intelligence-low-200h`, `hera-42b-synthetic-dumb-substance-high-200h`, `hera-42b-synthetic-dumb-substance-low-200h`, `hera2-ft-aloe`, `hera2-ft-dumb`, `hera2-ft-dumb-bad-transcript`, `hera2-ft-matched-aloe-1000h`, `hera2-ft-matched-synthetic-1000h`, `hera2-main-ckpts`, `hera3s2-ckptship-dryrun`, `hera3s2-stage2-e28-final2-j12355`, `hera3s2-stage2-e28-final2-j12491`, `hera3s2-stage2-e28-final2-j12554`, `hera3s2-stage2-e28-final2-j12602`, `hera3s2-stage2-e28-final2-j12694`, `hera3s2-stage2-e28-final2-j12747`, `hera3s2-stage2-e28-final2-j13034`, `hera3s2-stage2-e28-final2-j13390`, `hera3s2-stage2-e28-final2-j13543`, `hera3s2-stage2-e28-final2-j13563`, `hera3s2-stage2-e28-final2-j13586`, `hera3s2-stage2-e28-final2-j13836`, `hera3s2-stage2-e28-final2-j14141`, `hera3s2-stage2-e28-final2-j14680`, `hera3s2-stage3-e28-j15481`, `hera3s2-stage3-e28-j15539`, `klpre-backup-test`, `klpre-project`, `minerva-misc-models-v0`, `mistral-distill-bf16`, `nemotron3-ultra-aloe-lora-ckpts`, `pose-judge`, `pose-repo`, `pose-tokenizer-32cb-live`, `ppm-intelligence-opd`, `qwen-duplex-handoff`, `qwen-s4-val-bs4x-j9453-dcp-6207`, `qwen-stage3`, `qwen-tts-kl-stage2`, `qwen-voxtral-stage4`, `qwen38-27b-full-podcast-general-matched-final`, `qwen38-27b-full-podcast-per-id-loss-v2-final`, `qwen38-27b-full-synthetic-zoom-system-step50`, `qwen38-27b-full-zoom-podcast-even-step100`, `qwen38-27b-game-grpo-from-epoch3-r1`, `qwen38-27b-podcast-lora-r48`, `qwen38-27b-r197-aloe-cleaned-overlap-from-prompt30000-final`, `qwen38-27b-r197-aloe-stage2-personality-from-prompt22500-step1500`, `qwen38-27b-r197-podcast-50khr-lora-final`, `qwen38-27b-r197-podcast-masked-prompts-step30000`, `qwen38-27b-r197-podcast-masked-prompts-step5000`, `qwen38-27b-r48-devon-masked-direct-final`, `qwen38-27b-r48-devon-masked-transfer-final`, `qwen38-27b-r48-devon-rick-zoom-transfer-final`, `qwen38-27b-r48-synthetic-zoom-system-final`, `qwen38-27b-stage2-grpo-fullconv-overall-intellect-discourse-epoch1`, `qwen38-27b-stage2-grpo-fullconv-overall-intellect-discourse-epoch10`, `qwen38-27b-stage2-grpo-fullconv-overall-intellect-discourse-epoch2`, `qwen38-27b-stage2-grpo-fullconv-overall-intellect-discourse-epoch3`, `qwen38-27b-stage2-grpo-fullconv-overall-intellect-discourse-epoch4`, `qwen38-27b-stage2-grpo-fullconv-overall-intellect-discourse-epoch5`, `qwen38-27b-stage2-grpo-fullconv-overall-intellect-discourse-epoch6`, `qwen38-27b-stage2-grpo-fullconv-overall-intellect-discourse-epoch7`, `qwen38-27b-stage2-grpo-fullconv-overall-intellect-discourse-epoch8`, `qwen38-27b-stage2-grpo-fullconv-overall-intellect-discourse-epoch9`, `qwen38-27b-stage2-grpo-fullconv-overall-intellect-discourse-epochs1-10`, `sam-speech-pose-stage2-fast`, `sam-tts-stage1-1k`, `scratch-backup-1024-aloevox`, `si20k-tin-s3`, `tt-v3-retrain-ckpt`
