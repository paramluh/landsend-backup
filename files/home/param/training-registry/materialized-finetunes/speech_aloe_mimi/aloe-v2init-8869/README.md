# aloe-v2init-8869: Aloe fine-tune of PPM main at w-8869

This is an exact copy of the aloe-v2init recipe (aloe-v2init-12529 / -10351, aloe-v2 job 22383), with init = PPM main w-8869.
- w-8869's pretraining eval speech word CE norm is ~1.999, about equal to the 14B next-arch w-6242 (~1.9986). The partner run is /data/home/amartyavarma/ppm/aloe-next-6242.
- Init: /data/home/amartyavarma/ppm/ckpt-restore/ppm-v1-w-8869. It is a consolidated bf16 DCP export, loaded directly by train_aloe_v2.load_init_dcp, as aloe-v2init-10351 loaded w-10351. No conversion is needed: ckpt_conv.sbatch was only for 12529's node-local dcp save, and is kept here for reference only as ckpt_conv.sbatch.ref-12529-unused.
- Files:
  - code/ is byte-identical to aloe-v2init-10351/code and -12529/code.
  - aloe_v2init.env, aloe_v2init.sbatch, run_train.sh, stage_check.sh, plan_check.py and ref_cfg_22383.json are the 10351 files with only the run name and paths changed. run_train.sh also has the init path and MASTER_PORT 29861.
- Resolved cfg vs aloe-v2init-12529 (job 22575 log vs job 23508 log): ONLY init, run_tag, log_root, plan_dir, control_dir, export, local_export_root and ref_cfg differ. The trainer's own diff vs 22383 also passes.
- Launch (job 23508, slinky-10,11,12,13,14,16,17,19): submitted with `--cpus-per-task=4 --gres-flags=disable-binding` so it fits beside a CPU HF-upload job on slinky-11. This is a scheduler setting only. Data loaders and maths are unchanged.
- Stop: `touch control/STOP_AFTER_SAVE` (or `scancel -b -s USR1 <job>`). This writes a weights-only export, then exits.
- Exports go node-local to /scratch/amartyavarma/aloe-v2init-8869/exports/export-{42,84,126,169,211,253}. They are copied to exports/ here on Weka and backed up to a private HF repo (never GCS). The final consolidated DCP is at export-253.
