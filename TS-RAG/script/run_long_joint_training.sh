#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR=${LOG_DIR:-../logs/long_joint_training}
mkdir -p "$LOG_DIR"

export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

MODEL_ID=${MODEL_ID:-learnable_oracle_joint_long_30k}
LOG_PATH=${LOG_PATH:-${LOG_DIR}/${MODEL_ID}.log}

env \
  USE_MULTI_GPU=1 \
  DEVICES=${DEVICES:-0,1} \
  MODEL_ID="$MODEL_ID" \
  DATA_PATH=${DATA_PATH:-../datasets/pretrain/pretrain_pairs_ctx512_learnable_retriever} \
  TRAIN_STEPS=${TRAIN_STEPS:-30000} \
  EVALUATION_STEPS=${EVALUATION_STEPS:-10000} \
  LOG_INTERVAL=${LOG_INTERVAL:-500} \
  LR=${LR:-0.00003} \
  TMAX=${TMAX:-100} \
  MOE_RESIDUAL_INIT=${MOE_RESIDUAL_INIT:--4.6} \
  JOINT_TRAIN_RETRIEVER=${JOINT_TRAIN_RETRIEVER:-1} \
  RETRIEVER_POSITIVE_STRATEGY=${RETRIEVER_POSITIVE_STRATEGY:-oracle} \
  RETRIEVER_CL_LAMBDA=${RETRIEVER_CL_LAMBDA:-0.01} \
  ORACLE_DEVICE=${ORACLE_DEVICE:-cuda} \
  ORACLE_DEVICES=${ORACLE_DEVICES:-0,1} \
  ORACLE_DTYPE=${ORACLE_DTYPE:-float16} \
  CHECKPOINT_RESUME=${CHECKPOINT_RESUME:-} \
  OPTIMIZER_RESUME=${OPTIMIZER_RESUME:-} \
  bash script/pretrain_learnable_retriever_arm.sh > "$LOG_PATH" 2>&1

echo "Long joint training log: ${LOG_PATH}"
