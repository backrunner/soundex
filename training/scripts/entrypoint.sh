#!/usr/bin/env bash
# SoundEx training container entrypoint.
#
# Environment variables
# ---------------------
#   DATA_ROOT              Default parent for datasets (/data)
#   CACHE_DIR              Archive download cache (default $DATA_ROOT/.cache/downloads)
#   DOWNLOAD_CACHE_DIR     Alias for CACHE_DIR
#   MUSIC_LIBRARY_PATH    Generic audio-library processed-data root
#   SLAKH2100_PATH         Storage dir for Slakh2100
#   MEDLEYDB_PATH          Storage dir for MedleyDB
#   BABYSLAKH_PATH         Storage dir for BabySlakh
#   PATHS_MANIFEST         Where to write/read soundex_data_paths.yaml
#   DOWNLOAD_DATASETS      legacy datasets only when explicitly requested; current run uses audio libraries
#   SKIP_EXISTING          1 (default) skip download when tree looks ready
#   PREPROCESS             1 after download (default 1 if DOWNLOAD_DATASETS set)
#   KEEP_ARCHIVE           1 keep zip/tar after extract
#   MEDLEYDB_URL           Optional approved download URL for MedleyDB
#   RUN_TRAIN              1 (default) run train.py after data prep
#   CONFIG                 training config path (default configs/default.yaml)
#   EXTRA_TRAIN_ARGS       extra args appended to train.py
#   CHECK_ENV_ONLY         1 only run check_env.py and exit
#
# Examples
# --------
#   # Custom dataset + cache locations on the host (bind-mount both)
#   docker run --gpus all \
#     -v /mnt/datasets:/datasets -v /mnt/scratch:/scratch \
#     -e DATA_ROOT=/datasets \
#     -e CACHE_DIR=/scratch/soundex-cache \
#     -e MUSIC_LIBRARY_PATH=/datasets/music-library \
#     -e SLAKH2100_PATH=/datasets/slakh \
#     -e DOWNLOAD_DATASETS=slakh2100 \
#     soundex-train

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data}"
CONFIG="${CONFIG:-configs/default.yaml}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
KEEP_ARCHIVE="${KEEP_ARCHIVE:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
CHECK_ENV_ONLY="${CHECK_ENV_ONLY:-0}"
DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-}"
CACHE_DIR="${CACHE_DIR:-${DOWNLOAD_CACHE_DIR:-}}"
PATHS_MANIFEST="${PATHS_MANIFEST:-${DATA_ROOT}/soundex_data_paths.yaml}"

# Default preprocess on when downloading; off when not downloading unless set.
if [[ -z "${PREPROCESS+x}" ]]; then
  if [[ -n "${DOWNLOAD_DATASETS}" ]]; then
    PREPROCESS=1
  else
    PREPROCESS=0
  fi
fi

cd /workspace

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

echo "=== SoundEx container entrypoint ==="
echo "DATA_ROOT=${DATA_ROOT}"
echo "CACHE_DIR=${CACHE_DIR:-<default under DATA_ROOT>}"
echo "MUSIC_LIBRARY_PATH=${MUSIC_LIBRARY_PATH:-<unset>}"
echo "SLAKH2100_PATH=${SLAKH2100_PATH:-<default>}"
echo "MEDLEYDB_PATH=${MEDLEYDB_PATH:-<default>}"
echo "BABYSLAKH_PATH=${BABYSLAKH_PATH:-<default>}"
echo "PATHS_MANIFEST=${PATHS_MANIFEST}"
echo "CONFIG=${CONFIG}"
echo "DOWNLOAD_DATASETS=${DOWNLOAD_DATASETS:-<none>}"
echo "PREPROCESS=${PREPROCESS}"
echo "RUN_TRAIN=${RUN_TRAIN}"

python scripts/check_env.py --config "${CONFIG}" || {
  echo "Environment check reported issues (see above)." >&2
  if [[ "${CHECK_ENV_ONLY}" == "1" ]]; then
    exit 1
  fi
}

if [[ "${CHECK_ENV_ONLY}" == "1" ]]; then
  exit 0
fi

mkdir -p "${DATA_ROOT}"
if [[ -n "${CACHE_DIR}" ]]; then
  mkdir -p "${CACHE_DIR}"
fi
# Ensure per-dataset parents exist when overridden
for p in \
  "${MUSIC_LIBRARY_PATH:-}" \
  "${SLAKH2100_PATH:-}" \
  "${MEDLEYDB_PATH:-}" \
  "${BABYSLAKH_PATH:-}"
do
  if [[ -n "${p}" ]]; then
    mkdir -p "${p}"
  fi
done

if [[ -n "${DOWNLOAD_DATASETS}" ]]; then
  dl_args=(
    --datasets "${DOWNLOAD_DATASETS}"
    --data-root "${DATA_ROOT}"
    --config "${CONFIG}"
  )
  if [[ -n "${CACHE_DIR}" ]]; then
    dl_args+=(--cache-dir "${CACHE_DIR}")
  fi
  if [[ -n "${SLAKH2100_PATH:-}" ]]; then
    dl_args+=(--slakh2100-path "${SLAKH2100_PATH}")
  fi
  if [[ -n "${MEDLEYDB_PATH:-}" ]]; then
    dl_args+=(--medleydb-path "${MEDLEYDB_PATH}")
  fi
  if [[ -n "${BABYSLAKH_PATH:-}" ]]; then
    dl_args+=(--babyslakh-path "${BABYSLAKH_PATH}")
  fi
  if [[ -n "${PATHS_MANIFEST}" ]]; then
    dl_args+=(--paths-manifest "${PATHS_MANIFEST}")
  fi
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    dl_args+=(--skip-existing)
  fi
  if [[ "${KEEP_ARCHIVE}" == "1" ]]; then
    dl_args+=(--keep-archive)
  fi
  if [[ "${PREPROCESS}" == "1" ]]; then
    dl_args+=(--preprocess)
  fi
  if [[ -n "${MEDLEYDB_URL:-}" ]]; then
    dl_args+=(--medleydb-url "${MEDLEYDB_URL}")
  fi
  echo ">>> Downloading datasets: ${DOWNLOAD_DATASETS}"
  python scripts/download_datasets.py "${dl_args[@]}"
elif [[ "${PREPROCESS}" == "1" ]]; then
  echo ">>> PREPROCESS=1 without DOWNLOAD_DATASETS — preprocessing existing trees"
  slakh_root="${SLAKH2100_PATH:-${DATA_ROOT}/slakh2100}"
  if [[ -n "${SLAKH2100_PATH:-}" && -d "${slakh_root}" ]]; then
    python data/preprocess_slakh.py \
      --data-root "${slakh_root}" \
      --output-dir "${slakh_root}/processed" \
      --config "${CONFIG}" \
      --reuse-existing
  fi
fi

export DATA_ROOT
export PATHS_MANIFEST
# Propagate path env so train.py apply_data_path_overrides sees them
[[ -n "${MUSIC_LIBRARY_PATH:-}" ]] && export MUSIC_LIBRARY_PATH
[[ -n "${SLAKH2100_PATH:-}" ]] && export SLAKH2100_PATH
[[ -n "${MEDLEYDB_PATH:-}" ]] && export MEDLEYDB_PATH
[[ -n "${BABYSLAKH_PATH:-}" ]] && export BABYSLAKH_PATH
[[ -n "${CACHE_DIR}" ]] && export CACHE_DIR

if [[ "${RUN_TRAIN}" == "1" ]]; then
  train_args=(--config "${CONFIG}")
  if [[ -n "${PATHS_MANIFEST}" && -f "${PATHS_MANIFEST}" ]]; then
    train_args+=(--paths-manifest "${PATHS_MANIFEST}")
  fi
  echo ">>> Starting training: python train.py ${train_args[*]} ${EXTRA_TRAIN_ARGS:-}"
  # shellcheck disable=SC2086
  exec python train.py "${train_args[@]}" ${EXTRA_TRAIN_ARGS:-}
fi

echo "RUN_TRAIN=0 — data prep finished, not training."
if [[ "$#" -gt 0 ]]; then
  exec "$@"
fi
exit 0
