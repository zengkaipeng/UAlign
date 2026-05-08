#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="all"
SCILENCE_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    stage1|stage2|all)
      MODE="${arg}"
      ;;
    --scilence)
      SCILENCE_ARGS+=(--scilence)
      ;;
    *)
      echo "Usage: $0 [stage1|stage2|all] [--scilence]" >&2
      exit 1
      ;;
  esac
done

: "${DATA_DIR:?Set DATA_DIR to the processed USPTO-FULL dataset directory}"
MODEL_ARCH="${MODEL_ARCH:-./model_arch/uspto_full.json}"
TOKEN_JSON="${TOKEN_JSON:-./smiles_tokens/uspto_full_tokens.json}"

NUM_GPUS="${NUM_GPUS:-8}"
BS_PER_GPU="${BS_PER_GPU:-64}"
ACCU="${ACCU:-1}"

STAGE1_LOG_DIR="${STAGE1_LOG_DIR:-log_uspto_full/stage1}"
STAGE2_LOG_DIR="${STAGE2_LOG_DIR:-log_uspto_full/stage2}"
STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-uspto_full_stage1_bs64x8}"
STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-uspto_full_stage2_bs64x8}"

STAGE1_PORT="${STAGE1_PORT:-12345}"
STAGE2_PORT="${STAGE2_PORT:-10005}"

run_stage1() {
  python ddp_pretrain.py \
    --model_arch_path "${MODEL_ARCH}" \
    --data_path "${DATA_DIR}" \
    --seed 2023 \
    --bs "${BS_PER_GPU}" \
    --epoch 200 \
    --early_stop 8 \
    --lr 0.000125 \
    --base_log "${STAGE1_LOG_DIR}" \
    --log_name "${STAGE1_RUN_NAME}" \
    --token_path "${TOKEN_JSON}" \
    --lrgamma 0.993 \
    --warmup 4 \
    --accu "${ACCU}" \
    --num_workers 6 \
    --num_gpus "${NUM_GPUS}" \
    --port "${STAGE1_PORT}" \
    "${SCILENCE_ARGS[@]}"
}

resolve_stage1_outputs() {
  local ckpt token
  ckpt="${STAGE1_CHECKPOINT:-${STAGE1_LOG_DIR}/mod-${STAGE1_RUN_NAME}.pth}"
  token="${STAGE1_TOKEN_CKPT:-${STAGE1_LOG_DIR}/token-${STAGE1_RUN_NAME}.pkl}"
  if [[ ! -f "${ckpt}" ]]; then
    ckpt=""
  fi
  if [[ ! -f "${token}" ]]; then
    token=""
  fi
  if [[ -z "${ckpt}" || -z "${token}" ]]; then
    echo "Missing stage I outputs. Set STAGE1_CHECKPOINT/STAGE1_TOKEN_CKPT or run stage1 first." >&2
    exit 1
  fi
  export RESOLVED_STAGE1_CHECKPOINT="${ckpt}"
  export RESOLVED_STAGE1_TOKEN_CKPT="${token}"
}

run_stage2() {
  resolve_stage1_outputs
  python ddp_train_trans.py \
    --model_arch_path "${MODEL_ARCH}" \
    --aug_prob 0.5 \
    --data_path "${DATA_DIR}" \
    --seed 2023 \
    --bs "${BS_PER_GPU}" \
    --epoch 300 \
    --early_stop 8 \
    --num_gpus "${NUM_GPUS}" \
    --lr 0.00015 \
    --base_log "${STAGE2_LOG_DIR}" \
    --log_name "${STAGE2_RUN_NAME}" \
    --accu "${ACCU}" \
    --step_start 20 \
    --checkpoint "${RESOLVED_STAGE1_CHECKPOINT}" \
    --token_ckpt "${RESOLVED_STAGE1_TOKEN_CKPT}" \
    --warmup 4 \
    --gamma 0.985 \
    --label_smoothing 0.0 \
    --num_workers 8 \
    --port "${STAGE2_PORT}" \
    "${SCILENCE_ARGS[@]}"
}

case "${MODE}" in
  stage1)
    run_stage1
    ;;
  stage2)
    run_stage2
    ;;
  all)
    run_stage1
    run_stage2
    ;;
  *)
    echo "Usage: $0 [stage1|stage2|all] [--scilence]" >&2
    exit 1
    ;;
esac
