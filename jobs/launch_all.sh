#!/bin/bash
# Submit the slimmed GW-experiment matrix.
#
# Slimmed plan (matches src/experiments/run_gw_experiments.sh):
#   phase1 = baseline + variants
#   phase2 = lambda sweep on BEST_VARIANT  (3 lambdas: 0.001 / 0.005 / 0.01)
#   phase3 = OOD baseline + OOD GW
#   all    = phase1 + phase2 + phase3
#
# Detach / schedule / GW x sync ablations are kept available as separate
# *.job files but are no longer submitted by `all`. Submit explicitly:
#     sbatch --export=ALL,SEED=0 jobs/train_detach.job
#
# Each train_*.job is a SLURM array; this script fires sbatch for each phase
# once per seed in $SEEDS. Default 3 seeds gives error bars on every cell.
#
# Run from the login node:
#     bash jobs/launch_all.sh
#     bash jobs/launch_all.sh phase1
#     SEEDS="0 1 2" BEST_VARIANT=projected bash jobs/launch_all.sh phase2
#
# Override defaults via env vars:
#     ITERS=100000 LAMBDA=0.005 BEST_VARIANT=projected SEEDS="0 1" \
#         bash jobs/launch_all.sh all
#
# Re-submitting auto-resumes per (exp_id includes seed suffix), because
# train_*.job appends _s${SEED} to exp_id.

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
AC_WEIGHT="${AC_WEIGHT:-1e-2}"
SEEDS="${SEEDS:-0 1 2}"

echo "== GW experiment launcher (slim) =="
echo "phase           = $PHASE"
echo "seeds           = $SEEDS"
echo "ITERS           = $ITERS"
echo "MODEL           = $MODEL"
echo "LAMBDA          = $LAMBDA"
echo "BEST_VARIANT    = $BEST_VARIANT"
echo "BATCH_SIZE      = $BATCH_SIZE"
echo "EVAL_BATCH_SIZE = $EVAL_BATCH_SIZE"
echo "COMPILE         = $COMPILE"
echo "AC_WEIGHT       = $AC_WEIGHT"
echo "----------------------------------"

submit() {
    # submit <jobfile> <seed>
    local jobfile="$1"
    local seed="$2"
    local export_str="ALL,ITERS=${ITERS},MODEL=${MODEL},LAMBDA=${LAMBDA},BEST_VARIANT=${BEST_VARIANT},BATCH_SIZE=${BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},COMPILE=${COMPILE},AC_WEIGHT=${AC_WEIGHT},SEED=${seed}"
    echo "sbatch --export=${export_str} ${jobfile}"
    sbatch --export="${export_str}" "${jobfile}"
}

submit_phase1() {
    for s in $SEEDS; do
        submit jobs/train_baseline.job "$s"
        submit jobs/train_variants.job "$s"
    done
}

submit_phase2() {
    for s in $SEEDS; do
        submit jobs/train_lambda.job "$s"
    done
}

submit_phase3() {
    for s in $SEEDS; do
        submit jobs/train_ood.job "$s"
    done
}

case "$PHASE" in
    phase1) submit_phase1 ;;
    phase2) submit_phase2 ;;
    phase3) submit_phase3 ;;
    all)
        submit_phase1
        submit_phase2
        submit_phase3
        ;;
    *)
        echo "Unknown phase: $PHASE  (use phase1 | phase2 | phase3 | all)"
        exit 1
        ;;
esac

echo "----------------------------------"
echo "Done submitting. Watch the queue with:"
echo "    squeue -u \$USER"
echo "Tail any run's log with:"
echo "    tail -f output/<exp_id>/train-*-rank0.log"
