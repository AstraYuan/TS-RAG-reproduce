#!/usr/bin/env bash
set -euo pipefail

# Run from repository root so relative paths match the existing scripts.
cd "$(dirname "$0")/.."

LOG_DIR=${LOG_DIR:-../logs/debug_high_forecast_loss}
mkdir -p "$LOG_DIR"

export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

COMMON_GPU_ENV=(
  USE_MULTI_GPU=1
  DEVICES=${DEVICES:-0,1}
  ORACLE_DEVICE=${ORACLE_DEVICE:-cuda}
  ORACLE_DEVICES=${ORACLE_DEVICES:-0,1}
  ORACLE_DTYPE=${ORACLE_DTYPE:-float16}
)

COMMON_TRAIN_ENV=(
  LR=${LR:-0.00003}
  RETRIEVER_CL_LAMBDA=${RETRIEVER_CL_LAMBDA:-0.01}
  LOG_INTERVAL=${LOG_INTERVAL:-100}
  EVALUATION_STEPS=${EVALUATION_STEPS:-10000}
  TRAIN_STEPS=${TRAIN_STEPS:-1000}
  TMAX=${TMAX:-50}
  MOE_RESIDUAL_INIT=${MOE_RESIDUAL_INIT:--4.6}
)

run_stage() {
  local name="$1"
  shift
  local log_path="$LOG_DIR/${name}.log"
  echo "========== ${name} =========="
  echo "log: ${log_path}"
  env "${COMMON_GPU_ENV[@]}" "${COMMON_TRAIN_ENV[@]}" "$@" bash script/pretrain_learnable_retriever_arm.sh > "$log_path" 2>&1
  echo "done: ${name}"
}

# Experiment 0a: official/L2 pretrain pairs, no joint retriever, only diagnostics.
if [ "${RUN_BASELINE_INIT:-1}" -eq 1 ]; then
  run_stage baseline_official_pairs_init \
    MODEL_ID=debug_baseline_official_pairs_init \
    DATA_PATH=../datasets/pretrain/pretrain_pairs_ctx512 \
    JOINT_TRAIN_RETRIEVER=0 \
    DEBUG_INITIAL_LOSS=1 \
    DEBUG_INDEX_CHECK_BATCHES=5 \
    FAIL_ON_BAD_INDICES=1 \
    EXIT_AFTER_DEBUG=1
fi

# Experiment 1: learnable-retriever pairs, only diagnostics.
if [ "${RUN_LEARNABLE_INIT:-1}" -eq 1 ]; then
  run_stage learnable_pairs_init \
    MODEL_ID=debug_learnable_pairs_init \
    DATA_PATH=../datasets/pretrain/pretrain_pairs_ctx512_learnable_retriever \
    JOINT_TRAIN_RETRIEVER=0 \
    DEBUG_INITIAL_LOSS=1 \
    DEBUG_INDEX_CHECK_BATCHES=5 \
    FAIL_ON_BAD_INDICES=1 \
    EXIT_AFTER_DEBUG=1
fi

# Experiment 2: learnable-retriever pairs with MoE retrieval fusion disabled.
# This checks whether the frozen backbone/head is healthy before the MoE residual is added.
if [ "${RUN_BACKBONE_ONLY_INIT:-1}" -eq 1 ]; then
  run_stage learnable_pairs_backbone_only_init \
    MODEL_ID=debug_learnable_pairs_backbone_only_init \
    DATA_PATH=../datasets/pretrain/pretrain_pairs_ctx512_learnable_retriever \
    JOINT_TRAIN_RETRIEVER=0 \
    DISABLE_RETRIEVAL_FUSION=1 \
    DEBUG_INITIAL_LOSS=1 \
    DEBUG_INDEX_CHECK_BATCHES=5 \
    FAIL_ON_BAD_INDICES=1 \
    EXIT_AFTER_DEBUG=1
fi

# Experiment 0b: short baseline training with official/L2 pairs.
if [ "${RUN_BASELINE_TRAIN:-1}" -eq 1 ]; then
  run_stage baseline_official_pairs_train \
    MODEL_ID=debug_baseline_official_pairs_train \
    DATA_PATH=../datasets/pretrain/pretrain_pairs_ctx512 \
    JOINT_TRAIN_RETRIEVER=0 \
    DEBUG_INDEX_CHECK_BATCHES=5 \
    FAIL_ON_BAD_INDICES=1
fi

# Experiment 3: recommended fix, scaled MoE residual + low LR + joint CL.
if [ "${RUN_RESIDUAL_JOINT_TRAIN:-1}" -eq 1 ]; then
  run_stage learnable_pairs_residual_joint_train \
    MODEL_ID=debug_learnable_pairs_residual_joint_train \
    DATA_PATH=../datasets/pretrain/pretrain_pairs_ctx512_learnable_retriever \
    JOINT_TRAIN_RETRIEVER=1 \
    RETRIEVER_POSITIVE_STRATEGY=oracle \
    DEBUG_INDEX_CHECK_BATCHES=5 \
    FAIL_ON_BAD_INDICES=1
fi

echo "All selected stages finished. Logs are in ${LOG_DIR}"
