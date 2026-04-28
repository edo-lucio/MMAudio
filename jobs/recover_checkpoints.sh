#!/bin/bash
# Recover corrupt training checkpoints by promoting the atomic-write shadow.
#
# For every output/<exp_id>/ dir:
#   1. If <exp_id>_ckpt_last.pth loads cleanly → leave alone.
#   2. If last is corrupt/missing but <exp_id>_ckpt_shadow.pth loads cleanly →
#        mv <exp_id>_ckpt_last.pth <exp_id>_ckpt_last.pth.broken.<TS>   (if it existed)
#        mv <exp_id>_ckpt_shadow.pth <exp_id>_ckpt_last.pth
#   3. If both are dead → report and leave alone, unless WIPE_ON_FAIL=1 (then
#      `rm -rf output/<exp_id>` so the next sbatch starts that run from scratch).
#
# Why this is safe: runner.py::save_checkpoint always writes to _shadow.pth
# first, then renames atomically to _ckpt_last.pth. If the rename was
# interrupted (SIGKILL, ptxas crash, OOM), the shadow is the most recent
# *fully-written* checkpoint and is what you actually want.
#
# Usage:
#   bash jobs/recover_checkpoints.sh                       # scan all output/*
#   bash jobs/recover_checkpoints.sh output/gw_var_*       # specific glob
#   DRY_RUN=1 bash jobs/recover_checkpoints.sh             # preview only
#   WIPE_ON_FAIL=1 bash jobs/recover_checkpoints.sh        # rm dirs that have no recoverable ckpt

set -uo pipefail

# repo root (script lives in jobs/)
cd "$(dirname "$0")/.."

DRY_RUN="${DRY_RUN:-0}"
WIPE_ON_FAIL="${WIPE_ON_FAIL:-0}"

if [ "$#" -eq 0 ]; then
    set -- output/*
fi

TS=$(date +%Y%m%d_%H%M%S)

# Verify a .pth file loads via torch.load. Returns 0 on success, non-zero on failure.
check_pth() {
    local f="$1"
    [ -s "$f" ] || return 1
    python - "$f" <<'PYEOF' 2>/dev/null
import sys, torch
try:
    torch.load(sys.argv[1], map_location='cpu', weights_only=True)
except Exception:
    sys.exit(1)
PYEOF
}

OK=0
RECOVERED=0
DEAD=0
WIPED=0
NOT_RUN=0

for d in "$@"; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    last="$d/${name}_ckpt_last.pth"
    shadow="$d/${name}_ckpt_shadow.pth"

    # Skip dirs that aren't training runs (no ckpts and no ema_ckpts/)
    if [ ! -e "$last" ] && [ ! -e "$shadow" ] && [ ! -d "$d/ema_ckpts" ]; then
        echo "[skip] $name : not a run dir"
        NOT_RUN=$((NOT_RUN + 1))
        continue
    fi

    # Case 1: last is healthy
    if [ -f "$last" ] && check_pth "$last"; then
        echo "[ok]    $name : ${name}_ckpt_last.pth is healthy"
        OK=$((OK + 1))
        continue
    fi

    # Case 2: last is bad → try shadow
    if [ -f "$shadow" ] && check_pth "$shadow"; then
        echo "[fix]   $name : last is corrupt/missing, shadow is good — promoting"
        if [ "$DRY_RUN" = "1" ]; then
            if [ -f "$last" ]; then
                echo "          (dry-run) would mv ${last} -> ${last}.broken.${TS}"
            fi
            echo "          (dry-run) would mv ${shadow} -> ${last}"
        else
            if [ -f "$last" ]; then
                mv -- "$last" "${last}.broken.${TS}"
            fi
            mv -- "$shadow" "$last"
        fi
        RECOVERED=$((RECOVERED + 1))
        continue
    fi

    # Case 3: both dead
    echo "[dead]  $name : last and shadow both unloadable"
    DEAD=$((DEAD + 1))
    if [ "$WIPE_ON_FAIL" = "1" ]; then
        if [ "$DRY_RUN" = "1" ]; then
            echo "          (dry-run) would rm -rf ${d}"
        else
            rm -rf -- "$d"
            echo "          removed ${d}"
            WIPED=$((WIPED + 1))
        fi
    fi
done

echo
echo "=== summary ==="
echo "OK         : $OK"
echo "Recovered  : $RECOVERED"
echo "Dead       : $DEAD"
[ "$WIPE_ON_FAIL" = "1" ] && echo "Wiped      : $WIPED"
echo "Skipped    : $NOT_RUN  (non-run dirs)"

if [ "$DEAD" -gt 0 ] && [ "$WIPE_ON_FAIL" != "1" ]; then
    echo
    echo "Note: $DEAD run(s) have no recoverable checkpoint. To start them"
    echo "      from scratch on next sbatch, re-run with WIPE_ON_FAIL=1."
fi
