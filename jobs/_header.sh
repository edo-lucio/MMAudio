# Sourced preamble: loads modules, activates env, cd's to project root,
# sources _common.sh. Include with:
#   source "$PROJECT_DIR/jobs/_header.sh"
# AFTER the #SBATCH directives and the PROJECT_DIR/ENV_PATH definitions.

set -euo pipefail
set -x

module load CUDA/12.1.1
module load NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1
module load Miniconda3/25.5.1-1
source /opt/itu/easybuild/software/Miniconda3/25.5.1-1/etc/profile.d/conda.sh
conda activate "$ENV_PATH"

cd "$PROJECT_DIR"
source jobs/_common.sh
