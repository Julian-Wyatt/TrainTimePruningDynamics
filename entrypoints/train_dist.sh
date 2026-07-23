#!/usr/bin/env bash
set -euo pipefail

# Usage: entrypoints/train_dist.sh NUM_GPUS FIVES/study/question_A/core/dense "description" [Hydra overrides...]
NUM_GPUS="${1:?missing GPU count}"
CONFIG_PATH="${2:?missing experiment config}"
DESCRIPTION="${3:-}"
DESC_ESCAPED="${DESCRIPTION//,/<<COMMA>>}"
SAVING_ROOT="${SAVING_ROOT:-./saves}"
MASTER_PORT="${MASTER_PORT:-$((RANDOM % 60000 + 20000))}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export TORCH_DIST_TIMEOUT="${TORCH_DIST_TIMEOUT:-1800}"

# Keep torchrun alive long enough for Python workers to checkpoint after Slurm
# forwards SIGUSR1. The manager coordinates the requeue from inside Python.
(
  trap '' USR1
  PYTHONPATH="src:${PYTHONPATH:-}" exec torchrun \
    --master_port="${MASTER_PORT}" --nproc_per_node="${NUM_GPUS}" \
    src/core/train_test.py \
    "config_path=experiments/${CONFIG_PATH}" \
    "saving_root_dir=${SAVING_ROOT}" \
    "desc='${DESC_ESCAPED}'" \
    "${@:4}"
) &
TORCHRUN_PID=$!

trap 'pkill -USR1 -f "train_test.py" || true' USR1
while true; do
  if wait "${TORCHRUN_PID}"; then
    status=0
  else
    status=$?
  fi
  [ "${status}" -ne 138 ] && break
done
exit "${status}"
