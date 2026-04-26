#!/bin/bash
# Submit every GW-experiment training job in one go.
#
# Each train_*.job is a SLURM array (one task per config); this script just
# fires off `sbatch` for each one. Ordering on the cluster queue is not
# enforced -- they all enter the queue at once and run as GPUs become free.
#
# Run from the login node:
#     bash jobs/launch_all.sh
#     bash jobs/launch_all.sh phase1                # baseline + variants only
#     bash jobs/launch_all.sh phase2                # ablations using BEST_VARIANT
#     bash jobs/launch_all.sh phase3                # ood + gwxsync
#     bash jobs/launch_all.sh all                   # default: everything
#
# Override defaults via env vars:
#     ITERS=100000 MODEL=small_16k LAMBDA=0.005 BEST_VARIANT=projected \
#         bash jobs/launch_all.sh phase2
#
# Re-submitting any of these auto-resumes from output/<exp_id>/<exp_id>_ckpt_last.pth
# because exp_id is fixed per array index.

set -e

cd "$(dirname "$0")/.."   # repo root

PHASE="${1:-all}"

ITERS="${ITERS:-300000}"
MODEL="${MODEL:-small_16k}"
LAMBDA="${LAMBDA:-0.01}"
BEST_VARIANT="${BEST_VARIANT:-global}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
COMPILE="${COMPILE:-False}"   # default False because sm_75 GPUs can't JIT bf16 kernels; flip to True if you secure Ampere+

EXPORT="ALL,ITERS=${ITERS},MODEL=${MODEL},LAMBDA=${LAMBDA},BEST_VARIANT=${BEST_VARIANT},BATCH_SIZE=${BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},COMPILE=${COMPILE}"

echo "== GW experiment launcher =="
echo "phase           = $PHASE"
echo "ITERS           = $ITERS"
echo "MODEL           = $MODEL"
echo "LAMBDA          = $LAMBDA"
echo "BEST_VARIANT    = $BEST_VARIANT"
echo "BATCH_SIZE      = $BATCH_SIZE"
echo "EVAL_BATCH_SIZE = $EVAL_BATCH_SIZE"
echo "COMPILE         = $COMPILE"
echo "----------------------------"

submit() {
    local jobfile="$1"
    echo "sbatch --export=${EXPORT} ${jobfile}"
    sbatch --export="${EXPORT}" "${jobfile}"
}

case "$PHASE" in
    phase1)
        # E1 + E2: pick a winning variant before running the rest.
        submit jobs/train_baseline.job
        submit jobs/train_variants.job
        ;;
    phase2)
        # E3 + E4 + E5: ablate the chosen variant.
        # Pass BEST_VARIANT=<winner> in the env once you know it.
        submit jobs/train_lambda.job
        submit jobs/train_detach.job
        submit jobs/train_schedule.job
        ;;
    phase3)
        # E6 + E8: harder generalization / interaction tests.
        submit jobs/train_ood.job
        submit jobs/train_gwxsync.job
        ;;
    all)
        submit jobs/train_baseline.job
        submit jobs/train_variants.job
        submit jobs/train_lambda.job
        submit jobs/train_detach.job
        submit jobs/train_schedule.job
        submit jobs/train_ood.job
        submit jobs/train_gwxsync.job
        ;;
    *)
        echo "Unknown phase: $PHASE  (use phase1 | phase2 | phase3 | all)"
        exit 1
        ;;
esac

echo "----------------------------"
echo "Done submitting. Watch the queue with:"
echo "    squeue -u \$USER"
echo "Tail any run's log with:"
echo "    tail -f output/<exp_id>/train-*-rank0.log"
