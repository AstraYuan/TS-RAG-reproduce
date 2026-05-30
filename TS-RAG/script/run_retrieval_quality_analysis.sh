#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_DIR=${OUTPUT_DIR:-results/retrieval_quality}
mkdir -p "$OUTPUT_DIR"

SEQ_LEN=${SEQ_LEN:-512}
PRED_LEN=${PRED_LEN:-64}
TOP_K=${TOP_K:-10}
EVAL_TAIL=${EVAL_TAIL:-3000}

run_quality() {
  local dataset="$1"
  local original_csv="$2"
  local official_csv="$3"
  local learnable_csv="$4"
  local output_json="${OUTPUT_DIR}/${dataset}_retrieval_quality.json"

  echo "========== retrieval quality: ${dataset} =========="
  python analyze_retrieval_quality.py \
    --original_csv "$original_csv" \
    --official_csv "$official_csv" \
    --learnable_csv "$learnable_csv" \
    --seq_len "$SEQ_LEN" \
    --pred_len "$PRED_LEN" \
    --top_k "$TOP_K" \
    --eval_tail "$EVAL_TAIL" \
    --output_json "$output_json"
}

if [ "${RUN_ETTH1:-1}" -eq 1 ]; then
  run_quality \
    ETTh1 \
    "${ETTH1_ORIGINAL_CSV:-../datasets/ETT-small/ETTh1.csv}" \
    "${ETTH1_OFFICIAL_CSV:-../datasets/ETT-small/ETTh1_retrieve_ETTh1_512_only_self_train_None.csv}" \
    "${ETTH1_LEARNABLE_CSV:-../datasets/ETT-small/ETTh1_retrieve_ETTh1_512_only_self_train_learnable_oracle_joint_stable.csv}"
fi

echo "Retrieval quality outputs are in ${OUTPUT_DIR}"
