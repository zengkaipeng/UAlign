#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="all"
SCILENCE_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    stage1|stage2_unknown|stage2_known|stage2|all)
      MODE="${arg}"
      ;;
    --scilence)
      SCILENCE_ARGS+=(--scilence)
      ;;
    *)
      echo "Usage: $0 [stage1|stage2_unknown|stage2_known|stage2|all] [--scilence]" >&2
      exit 1
      ;;
  esac
done

: "${DATA_DIR:?Set DATA_DIR to the processed USPTO-50K dataset directory}"
MODEL_ARCH="${MODEL_ARCH:-./model_arch/uspto_50k.json}"
TOKEN_JSON="${TOKEN_JSON:-./smiles_tokens/uspto_50k_tokens.json}"
DEVICE="${DEVICE:-0}"

STAGE1_LOG_DIR="${STAGE1_LOG_DIR:-log_uspto_50k_single_card/stage1}"
STAGE2_UNKNOWN_LOG_DIR="${STAGE2_UNKNOWN_LOG_DIR:-log_uspto_50k_single_card/stage2_unknown}"
STAGE2_KNOWN_LOG_DIR="${STAGE2_KNOWN_LOG_DIR:-log_uspto_50k_single_card/stage2_known}"

STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-uspto_50k_stage1_single_card}"
STAGE2_UNKNOWN_RUN_NAME="${STAGE2_UNKNOWN_RUN_NAME:-uspto_50k_stage2_unknown_single_card}"
STAGE2_KNOWN_RUN_NAME="${STAGE2_KNOWN_RUN_NAME:-uspto_50k_stage2_known_single_card}"

run_stage1() {
  python pretrain.py \
    --model_arch_path "${MODEL_ARCH}" \
    --data_path "${DATA_DIR}" \
    --seed 2023 \
    --bs 128 \
    --epoch 200 \
    --early_stop 15 \
    --device "${DEVICE}" \
    --lr 0.000125 \
    --base_log "${STAGE1_LOG_DIR}" \
    --log_name "${STAGE1_RUN_NAME}" \
    --token_path "${TOKEN_JSON}" \
    --lrgamma 0.993 \
    --warmup 4 \
    --accu 1 \
    --num_worker 8 \
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

run_stage2_unknown() {
  resolve_stage1_outputs
  python train_trans.py \
    --model_arch_path "${MODEL_ARCH}" \
    --aug_prob 0.8 \
    --data_path "${DATA_DIR}" \
    --seed 2023 \
    --bs 256 \
    --epoch 300 \
    --early_stop 15 \
    --device "${DEVICE}" \
    --lr 0.0001 \
    --base_log "${STAGE2_UNKNOWN_LOG_DIR}" \
    --log_name "${STAGE2_UNKNOWN_RUN_NAME}" \
    --accu 1 \
    --step_start 15 \
    --checkpoint "${RESOLVED_STAGE1_CHECKPOINT}" \
    --token_ckpt "${RESOLVED_STAGE1_TOKEN_CKPT}" \
    --warmup 4 \
    --gamma 0.992 \
    --label_smoothing 0.0 \
    --num_workers 8 \
    "${SCILENCE_ARGS[@]}"
}

run_stage2_known() {
  resolve_stage1_outputs
  python train_trans.py \
    --model_arch_path "${MODEL_ARCH}" \
    --aug_prob 0.5 \
    --data_path "${DATA_DIR}" \
    --seed 2023 \
    --bs 256 \
    --epoch 300 \
    --early_stop 15 \
    --device "${DEVICE}" \
    --lr 0.0001 \
    --base_log "${STAGE2_KNOWN_LOG_DIR}" \
    --log_name "${STAGE2_KNOWN_RUN_NAME}" \
    --accu 1 \
    --step_start 20 \
    --checkpoint "${RESOLVED_STAGE1_CHECKPOINT}" \
    --token_ckpt "${RESOLVED_STAGE1_TOKEN_CKPT}" \
    --warmup 4 \
    --gamma 0.99 \
    --use_class \
    --label_smoothing 0.0 \
    --num_workers 8 \
    "${SCILENCE_ARGS[@]}"
}

case "${MODE}" in
  stage1)
    run_stage1
    ;;
  stage2_unknown)
    run_stage2_unknown
    ;;
  stage2_known)
    run_stage2_known
    ;;
  stage2)
    run_stage2_unknown
    run_stage2_known
    ;;
  all)
    run_stage1
    run_stage2_unknown
    run_stage2_known
    ;;
  *)
    echo "Usage: $0 [stage1|stage2_unknown|stage2_known|stage2|all] [--scilence]" >&2
    exit 1
    ;;
esac
