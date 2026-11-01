Frozen copies (2026-10-11, PPM T7) of the stage-2/3 modules PPM imports. Do not edit; PPM-specific logic lives in ../*.py.
speech_q.py, heads_s2q.py        <- qwen-1ch/stage2
s3rows.py                         <- qwen-1ch/stage3/data/rows_build/scripts
data_s3q.py pose_s3q.py txtdup_s3q.py sysprompt.py persona_s3q.py persona_text_s3q.py lightms_s3q.py sched_s3q.py <- qwen-1ch/stage3
../vendor: fused_triton.py <- qwen-1ch/train/vendor; fa3_op.py pinned_fa3.py <- qwen-1ch/vendor

PPM EDITS (10-11, user decision "code delay 2 everywhere, one knob"): speech_q.CD <- env PPM_CD (default 2); sysprompt.TAIL = speech_q.CD and
build_prefix(cd=None -> speech_q.CD, asserted == TAIL). pose_s3q / data_s3q read SP.CD (unchanged code; their docstrings still say 3).
codep_s3q.py                      <- qwen-1ch/stage3 (10-12, frozen copy: the pose trainer's codependence probe; train_ppm.py builds its subset with it)
aloe-vox-ft (10-16): audio_s4q.py voxtral_front.py vadapter_s4q.py <- qwen-1ch/stage4-asr2 (the ASR adapter run's front + adapter class;
sha256 c87bbc44.. / c4d73b11.. / fa745b09.., identical to the source at copy time). ../model_ppm.py: VOX EDIT = optional vadapter module
(group 'vadapter', own FSDP unit) + forward adds vadapter(b['vlat']) at b['vpos']; flag off = unchanged PPM.
