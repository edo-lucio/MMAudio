"""Repo-root resolution for path-sensitive references.

Hydra changes CWD to the run directory and SLURM jobs may invoke entry
points from anywhere, so we cannot rely on `./ext_weights/...` working.

`REPO_ROOT` is the parent of the `src/` directory that holds this file's
package (i.e. the directory that contains `ext_weights/`, `config/`,
`data/`, `output/`, ...).
"""
from pathlib import Path

# this file: <REPO>/src/mmaudio/utils/paths.py
REPO_ROOT: Path = Path(__file__).resolve().parents[3]


def repo_path(*parts: str) -> Path:
    """Resolve a path under the repo root."""
    return REPO_ROOT.joinpath(*parts)
