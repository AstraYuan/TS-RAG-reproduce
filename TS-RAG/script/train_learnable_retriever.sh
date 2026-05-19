#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

run_file=train_learnable_retriever.py

model_id=learnable_retriever_ctx512_dim256
chronos_model_path=${CHRONOS_MODEL_PATH:-amazon/chronos-t5-base}

top_k=10
context_length=512
prediction_length=64
retrieval_database_path=../retrieval_database/pretrain/retrieval_database_${context_length}.parquet
data_path=../datasets/pretrain/pretrain_pairs_ctx${context_length}

input_dim=768
hidden_dim=512
output_dim=256
temperature=0.07

train_steps=10000
evaluation_steps=1000
batch_size=256
shuffle_buffer_length=10000
lr=0.0001
weight_decay=0.01
gpu_loc=0

python "$run_file" \
    --model_id "$model_id" \
    --chronos_model_path "$chronos_model_path" \
    --top_k "$top_k" \
    --context_length "$context_length" \
    --prediction_length "$prediction_length" \
    --retrieval_database_path "$retrieval_database_path" \
    --data_path "$data_path" \
    --input_dim "$input_dim" \
    --hidden_dim "$hidden_dim" \
    --output_dim "$output_dim" \
    --temperature "$temperature" \
    --train_steps "$train_steps" \
    --evaluation_steps "$evaluation_steps" \
    --batch_size "$batch_size" \
    --shuffle_buffer_length "$shuffle_buffer_length" \
    --learning_rate "$lr" \
    --weight_decay "$weight_decay" \
    --gpu_loc "$gpu_loc"
