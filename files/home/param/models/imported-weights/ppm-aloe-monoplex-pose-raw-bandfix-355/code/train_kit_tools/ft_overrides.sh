# pose6 kit (B3): the launcher's user-call switches, applied AFTER sourcing ft_s2pg0.env (by ft_s2pg0_run.sh, launch.sh and tests/dry_all.sbatch,
# so the resolved config is the same everywhere). Defaults = the env file = the recipe + link OFF.
#   FT_LINK    off (default: QQA_DECP_BB_OFF=1 = base 31101's state) | on (mono8020's knob value)                       [user call]
#   FT_SAVES   final (DEFAULT, main 10-27: no hourly saves, the final FULL save = masters + AdamW) | hourly (= recipe ALOE_SAVE_EVERY_MIN=60) |
#              none (+ ALOE_SKIP_FINAL_FULL=1)                                                                  [ops, maths unchanged]
#   FT_EXPORTS all (default = recipe 60/119/178/238/298 + 357) | final (357 only)                                      [user call, ops]
case "${FT_LINK:-off}" in off) export QQA_DECP_BB_OFF=1;; on) export QQA_DECP_BB_OFF=0;; *) echo "FT_LINK must be off | on" >&2; return 1;; esac
case "${FT_SAVES:-final}" in hourly) ;; final) export ALOE_SAVE_EVERY_MIN=100000;; none) export ALOE_SAVE_EVERY_MIN=100000 ALOE_SKIP_FINAL_FULL=1;;
  *) echo "FT_SAVES must be hourly | final | none" >&2; return 1;; esac
case "${FT_EXPORTS:-all}" in all) ;; final) export QQA_EXPORT_STEPS=;; *) echo "FT_EXPORTS must be all | final" >&2; return 1;; esac
