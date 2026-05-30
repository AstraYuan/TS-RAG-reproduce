#!/usr/bin/env bash
set -euo pipefail

# Run from TS-RAG/TS-RAG. This script turns the 2026-05-29 QA plan into
# executable stages. Each stage can be disabled with RUN_*=0.

cd "$(dirname "$0")/.."

LOG_DIR=${LOG_DIR:-../logs/next_step_ablation}
mkdir -p "$LOG_DIR"

export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

COMMON_ENV=(
  USE_MULTI_GPU=1
  DEVICES=${DEVICES:-0,1}
  TRAIN_STEPS=${TRAIN_STEPS:-10000}
  EVALUATION_STEPS=${EVALUATION_STEPS:-10000}
  LOG_INTERVAL=${LOG_INTERVAL:-500}
  LR=${LR:-0.00003}
  TMAX=${TMAX:-50}
  MOE_RESIDUAL_INIT=${MOE_RESIDUAL_INIT:--4.6}
  ORACLE_DEVICE=${ORACLE_DEVICE:-cuda}
  ORACLE_DEVICES=${ORACLE_DEVICES:-0,1}
  ORACLE_DTYPE=${ORACLE_DTYPE:-float16}
)

run_train_stage() {
  local name="$1"
  shift
  local log_path="${LOG_DIR}/${name}.log"
  echo "========== train: ${name} =========="
  echo "log: ${log_path}"
  env "${COMMON_ENV[@]}" "$@" bash script/pretrain_learnable_retriever_arm.sh > "$log_path" 2>&1
  echo "done: ${name}"
}

run_eval_stage() {
  local name="$1"
  shift
  local log_path="${LOG_DIR}/${name}.log"
  echo "========== eval: ${name} =========="
  echo "log: ${log_path}"
  env USE_MULTI_GPU=1 DEVICES=${DEVICES:-0,1} "$@" bash script/zeroshot_chronos.sh > "$log_path" 2>&1
  echo "done: ${name}"
}

# Priority 1a: original fixed-retriever pairs under the new stable training config.
if [ "${RUN_ORIGINAL_PAIRS_NEW_CONFIG:-1}" -eq 1 ]; then
  run_train_stage ablation_original_pairs_new_config \
    MODEL_ID=ablation_original_pairs_new_config \
    DATA_PATH=../datasets/pretrain/pretrain_pairs_ctx512 \
    JOINT_TRAIN_RETRIEVER=0 \
    RETRIEVER_CL_LAMBDA=0

  if [ "${RUN_EVAL_ABLATIONS:-0}" -eq 1 ]; then
    run_eval_stage eval_ablation_original_pairs_new_config \
      CHECKPOINT_MODEL_PATH=./checkpoints/ablation_original_pairs_new_config/model_steps9999.pth \
      SAVE_FILE_NAME=zeroshot_ablation_original_pairs_new_config.txt \
      DATASETS="${ABLATION_EVAL_DATASETS:-ETTh1}"
  fi
fi

# Priority 1b: learnable pretrain pairs, no joint CL.
if [ "${RUN_LEARNABLE_PAIRS_NO_JOINT:-1}" -eq 1 ]; then
  run_train_stage ablation_learnable_pairs_no_joint \
    MODEL_ID=ablation_learnable_pairs_no_joint \
    DATA_PATH=../datasets/pretrain/pretrain_pairs_ctx512_learnable_retriever \
    JOINT_TRAIN_RETRIEVER=0 \
    RETRIEVER_CL_LAMBDA=0

  if [ "${RUN_EVAL_ABLATIONS:-0}" -eq 1 ]; then
    run_eval_stage eval_ablation_learnable_pairs_no_joint \
      CHECKPOINT_MODEL_PATH=./checkpoints/ablation_learnable_pairs_no_joint/model_steps9999.pth \
      RETRIEVER_PROJECTOR_PATH=${LEARNABLE_PROJECTOR_PATH:-./checkpoints/learnable-retriever/learnable_retriever_ctx512_dim256_oracle/projector_final.pth} \
      RETRIEVAL_TAG=learnable_pairs_no_joint \
      SAVE_FILE_NAME=zeroshot_ablation_learnable_pairs_no_joint.txt \
      DATASETS="${ABLATION_EVAL_DATASETS:-ETTh1}"
  fi
fi

# Optional lambda ablations. Default off because each stage costs several hours.
if [ "${RUN_LAMBDA_005:-0}" -eq 1 ]; then
  run_train_stage ablation_learnable_pairs_joint_lambda005 \
    MODEL_ID=ablation_learnable_pairs_joint_lambda005 \
    DATA_PATH=../datasets/pretrain/pretrain_pairs_ctx512_learnable_retriever \
    JOINT_TRAIN_RETRIEVER=1 \
    RETRIEVER_POSITIVE_STRATEGY=oracle \
    RETRIEVER_CL_LAMBDA=0.05
fi

if [ "${RUN_LAMBDA_01:-0}" -eq 1 ]; then
  run_train_stage ablation_learnable_pairs_joint_lambda01 \
    MODEL_ID=ablation_learnable_pairs_joint_lambda01 \
    DATA_PATH=../datasets/pretrain/pretrain_pairs_ctx512_learnable_retriever \
    JOINT_TRAIN_RETRIEVER=1 \
    RETRIEVER_POSITIVE_STRATEGY=oracle \
    RETRIEVER_CL_LAMBDA=0.1
fi

echo "Selected stages finished. Logs are in ${LOG_DIR}"
