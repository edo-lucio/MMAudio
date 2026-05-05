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
# Collapse prevention is now selected by gw_regularization.collapse_penalty in
# config/base_config.yaml (default: vicreg). To ablate, override here:
#     COLLAPSE_PENALTY=logdet bash jobs/launch_all.sh
#     COLLAPSE_PENALTY=none   bash jobs/launch_all.sh
LAMBDA="${LAMBDA:-5e-3}"
OOD_VARIANT="${OOD_VARIANT:-global}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
COMPILE="${COMPILE:-False}"
COLLAPSE_PENALTY="${COLLAPSE_PENALTY:-vicreg}"

EXPORT="ALL,ITERS=${ITERS},MODEL=${MODEL},LAMBDA=${LAMBDA},OOD_VARIANT=${OOD_VARIANT},BATCH_SIZE=${BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},COMPILE=${COMPILE},COLLAPSE_PENALTY=${COLLAPSE_PENALTY}"

echo "== GW experiment launcher =="
echo "phase             = $PHASE"
echo "ITERS             = $ITERS"
echo "MODEL             = $MODEL"
echo "LAMBDA            = $LAMBDA"
echo "OOD_VARIANT       = $OOD_VARIANT"
echo "COLLAPSE_PENALTY  = $COLLAPSE_PENALTY"
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
