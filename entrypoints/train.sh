#!/usr/bin/env bash
set -euo pipefail

# Usage: entrypoints/train.sh FIVES/study/question_A/core/dense "description" [Hydra overrides...]
CONFIG_PATH="${1:?missing experiment config}"
DESCRIPTION="${2:-}"
DESC_ESCAPED="${DESCRIPTION//,/<<COMMA>>}"
SAVING_ROOT="${SAVING_ROOT:-./saves}"

PYTHONPATH="src:${PYTHONPATH:-}" \
python src/core/train_test.py \
  "config_path=experiments/${CONFIG_PATH}" \
  "saving_root_dir=${SAVING_ROOT}" \
  "desc='${DESC_ESCAPED}'" \
  "${@:3}"
