# Common environment for all MMAudio SLURM jobs.
# Sourced by each job script after module load + conda activate.

# Cluster's per-user RLIMIT_NPROC is ~12; cap every thread pool or BLAS will
# explode on import. Keep these in sync with --cpus-per-task in the job.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2

# Source layout: all entry points live under src/; jobs run from $PROJECT_DIR.
# Use these instead of hard-coding the path.
export SRC_DIR="${PROJECT_DIR}/src"
export EXPERIMENTS_DIR="${SRC_DIR}/experiments"
export TRAINING_DIR="${SRC_DIR}/training"
# Make `import mmaudio` work regardless of CWD.
export PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Useful log lines for debugging limits
echo "host=$(hostname)  ulimit -u: $(ulimit -u)  nproc: $(nproc)"
echo "cuda: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# One-shot TSV aliasing: config/data/base.yaml wants sets/vgg3-*.tsv
# but the repo ships sets/vgg-*.tsv.
(
    cd sets || exit 0
    for f in vgg-*.tsv; do
        [ -f "$f" ] && [ ! -f "${f/vgg-/vgg3-}" ] && cp "$f" "${f/vgg-/vgg3-}"
    done
) || true
