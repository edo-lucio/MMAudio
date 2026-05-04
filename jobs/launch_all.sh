#!/bin/bash
# Submit the experiment matrix.
#
# 7 runs total:
#   gw_baseline                                                 (train_baseline.job)
#   gw_var_{global,projected,c_g,fused}    (4 array tasks)      (train_variants.job)
#   gw_ood_baseline + gw_ood_<OOD_VARIANT>  (2 array tasks)      (train_ood.job)
#
# Run from the login node:
#     bash jobs/launch_all.sh                # default: everything
#     bash jobs/launch_all.sh baseline       # just the baseline
#     bash jobs/launch_all.sh variants       # the 4 variants
#     bash jobs/launch_all.sh ood            # the OOD pair
#
# Override defaults via env vars:
#     ITERS=100000 LAMBDA=0.005 OOD_VARIANT=projected bash jobs/launch_all.sh

set -e

cd "$(dirname "$0")/.."

PHASE="${1:-all}"

ITERS="${ITERS:-300000}"
MODEL="${MODEL:-small_16k}"
# Defaults calibrated for V100 + B=32 against the collapse failure mode
# observed in the previous sweep (AC saturated at 221 for projected/c_g/fused).
# - LAMBDA halved to 0.005: lower GW gain from collapse (~0.035 instead of 0.07).
# - AC_WEIGHT raised to 0.1: barrier cost ~28.5 vs previous 2.21.
# - AC_ETA tightened to 1e-4: ceiling (B-1)|log eta| rises to ~285,
#   gradient at degeneracy 10x sharper than at eta=1e-3.
# Net: AC barrier dominates GW gain by ~800x, vs ~30x previously.
LAMBDA="${LAMBDA:-5e-3}"
OOD_VARIANT="${OOD_VARIANT:-global}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
COMPILE="${COMPILE:-False}"
AC_WEIGHT="${AC_WEIGHT:-1e-1}"
AC_ETA="${AC_ETA:-1e-4}"

EXPORT="ALL,ITERS=${ITERS},MODEL=${MODEL},LAMBDA=${LAMBDA},OOD_VARIANT=${OOD_VARIANT},BATCH_SIZE=${BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},COMPILE=${COMPILE},AC_WEIGHT=${AC_WEIGHT},AC_ETA=${AC_ETA}"

echo "== GW experiment launcher =="
echo "phase        = $PHASE"
echo "ITERS        = $ITERS"
echo "MODEL        = $MODEL"
echo "LAMBDA       = $LAMBDA"
echo "OOD_VARIANT  = $OOD_VARIANT"
echo "AC_WEIGHT    = $AC_WEIGHT"
echo "AC_ETA       = $AC_ETA"
echo "----------------------------"

submit() {
    local jobfile="$1"
    echo "sbatch --export=${EXPORT} ${jobfile}"
    sbatch --export="${EXPORT}" "${jobfile}"
}

case "$PHASE" in
    baseline) submit jobs/train_baseline.job ;;
    variants) submit jobs/train_variants.job ;;
    ood)      submit jobs/train_ood.job ;;
    all)
        submit jobs/train_baseline.job
        submit jobs/train_variants.job
        submit jobs/train_ood.job
        ;;
    *)
        echo "Unknown phase: $PHASE  (use baseline | variants | ood | all)"
        exit 1
        ;;
esac

echo "----------------------------"
echo "Done. Watch with: squeue -u \$USER"
