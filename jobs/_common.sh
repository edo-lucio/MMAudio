# Common environment for all MMAudio SLURM jobs.
# Sourced by each job script after module load + conda activate.

# Cluster's per-user RLIMIT_NPROC is ~12; cap every thread pool or BLAS will
# explode on import. Keep these in sync with --cpus-per-task in the job.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2

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
