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

: "${DATA_DIR:?Set DATA_DIR to the processed USPTO-MIT dataset directory}"
MODEL_ARCH="${MODEL_ARCH:-./model_arch/uspto_mit.json}"
TOKEN_JSON="${TOKEN_JSON:-./smiles_tokens/uspto_mit_tokens.json}"

STAGE1_LOG_DIR="${STAGE1_LOG_DIR:-log_uspto_mit/stage1}"
STAGE2_LOG_DIR="${STAGE2_LOG_DIR:-log_uspto_mit/stage2}"
STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-uspto_mit_stage1}"
STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-uspto_mit_stage2}"

run_stage1() {
  python ddp_pretrain.py \
    --model_arch_path "${MODEL_ARCH}" \
    --data_path "${DATA_DIR}" \
    --seed 2023 \
    --bs 128 \
    --epoch 300 \
    --early_stop 7 \
    --lr 0.00005 \
    --base_log "${STAGE1_LOG_DIR}" \
    --log_name "${STAGE1_RUN_NAME}" \
    --token_path "${TOKEN_JSON}" \
    --lrgamma 0.992 \
    --warmup 3 \
    --accu 1 \
    --num_workers 6 \
    --num_gpus 4 \
    --port 12345 \
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
    --bs 128 \
    --epoch 200 \
    --early_stop 10 \
    --num_gpus 4 \
    --lr 0.000075 \
    --base_log "${STAGE2_LOG_DIR}" \
    --log_name "${STAGE2_RUN_NAME}" \
    --accu 1 \
    --step_start 10 \
    --checkpoint "${RESOLVED_STAGE1_CHECKPOINT}" \
    --token_ckpt "${RESOLVED_STAGE1_TOKEN_CKPT}" \
    --warmup 2 \
    --gamma 0.993 \
    --label_smoothing 0.0 \
    --num_workers 8 \
    --port 12135 \
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
