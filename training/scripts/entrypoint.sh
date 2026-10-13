#!/usr/bin/env bash
# SoundEx training container: mount already prepared, reviewed audio-library pairs.
# CONFIG selects the profile; MUSIC_LIBRARY_PATH/SPEECH_LIBRARY_PATH override roots.
# PATHS_MANIFEST optionally supplies explicit YAML path mappings.
# CHECK_ENV_ONLY=1 checks the environment; RUN_TRAIN=0 runs an optional command.
# EXTRA_TRAIN_ARGS supplies whitespace-separated train.py options (no shell eval).
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
RUN_TRAIN="${RUN_TRAIN:-1}"
CHECK_ENV_ONLY="${CHECK_ENV_ONLY:-0}"
cd /workspace

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

# Fail visibly for old automation instead of silently skipping requested data prep.
if [[ -n "${DOWNLOAD_DATASETS:-}" || "${PREPROCESS:-0}" == "1" ]]; then
  echo "Automatic legacy dataset acquisition was removed. Use data/preprocess_library.py with reviewed originals before training." >&2
  exit 2
fi

python scripts/check_env.py --config "${CONFIG}"
if [[ "${CHECK_ENV_ONLY}" == "1" ]]; then
  exit 0
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  train_args=(--config "${CONFIG}")
  if [[ -n "${PATHS_MANIFEST:-}" ]]; then
    if [[ ! -f "${PATHS_MANIFEST}" ]]; then
      echo "Paths manifest not found: ${PATHS_MANIFEST}" >&2
      exit 1
    fi
    train_args+=(--paths-manifest "${PATHS_MANIFEST}")
  fi
  echo ">>> Starting training with ${CONFIG}"
  # shellcheck disable=SC2086
  exec python train.py "${train_args[@]}" ${EXTRA_TRAIN_ARGS:-}
fi

if [[ "$#" -gt 0 ]]; then
  exec "$@"
fi
