#!/bin/bash
# pose5: the resolved-cfg diff that switch_to_pose.sh --armed requires (kit/state/CFG_DIFF_switch.txt, must be newer than the env + trainer).
# = c/cfgdiff_pose5.sh (code-811-s2p/test811/cfg_diff.py vs job 27653's logged cfg) with the switch's own overrides (resume = the per-host link,
# max steps = the env's, or B's dry-run end while the env still says FILL_ME). CPU on slinky-4 (never slinky-9/10/18), tiny, nice.
# usage: bash cfgdiff_switch.sh [max_steps]
set -u
P5=/data/home/amartyavarma/ppm/pose5; K=$P5/kit; ENV0=$P5/prod_811_fresh_append5_s2p.env
M=${1:-$(grep -E '^PPM_MAX_STEPS=' $ENV0 | tail -1 | cut -d= -f2)}; [[ "$M" =~ ^[0-9]+$ ]] || M=19425
OUT=${CFGDIFF_OUT:-$K/state/CFG_DIFF_switch.txt}         # offline tests only: CFGDIFF_OUT (scratch path) + CFGDIFF_EXTRA ("PPM_S2P=1"); defaults = the switch's
timeout 1200 srun -p all -w ${SPARE:-slinky-4} -N1 -n1 -c 2 --mem=24G -t 15 -J pose5-cfgdiff nice -n 10 bash $P5/c/cfgdiff_pose5.sh $OUT.tmp \
  $P5/code-811-s2p PPM_RESUME_FROM=/scratch/amartyavarma/pose5-resume/dcp-17850 PPM_MAX_STEPS=$M ${CFGDIFF_EXTRA:-} > $K/logs/cfgdiff-switch$([ -n "${CFGDIFF_OUT:-}" ] && echo -test).log 2>&1 || { echo "cfg diff FAILED (see $K/logs/cfgdiff-switch*.log)"; exit 1; }
# main 10-27: s_emb / inp.g0 get their OWN optimizer groups -> also diff model_ppm.GROUPS (27653's code-811 vs code-811-s2p) as comment lines
GD=$(python3 - <<'EOF'
import ast
def groups(f):
    for n in ast.parse(open(f).read()).body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "GROUPS" for t in n.targets): return list(ast.literal_eval(n.value))
a, b = groups("/data/home/amartyavarma/ppm/code-811/model_ppm.py"), groups("/data/home/amartyavarma/ppm/pose5/code-811-s2p/model_ppm.py")
print(f"# GROUPS 27653: {a}"); print(f"# GROUPS new  : {b}"); print(f"# GROUPS added: {[g for g in b if g not in a]} removed: {[g for g in a if g not in b]}")
EOF
)
{ cat $OUT.tmp; echo "$GD"; echo "# pose5 switch cfg diff $(TZ=America/Los_Angeles date '+%F %T PT'): env $ENV0, code $P5/code-811-s2p, max_steps $M${CFGDIFF_EXTRA:+ overrides $CFGDIFF_EXTRA}$( grep -q '^PPM_MAX_STEPS=FILL_ME' $ENV0 && echo ' (PLACEHOLDER: env still FILL_ME)')"; } > $OUT && rm -f "${OUT:?}.tmp"
grep -vE '^#' $OUT
