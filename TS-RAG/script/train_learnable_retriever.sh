#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

run_file=train_learnable_retriever.py

positive_strategy=${POSITIVE_STRATEGY:-distill}
if [ -n "${MODEL_ID:-}" ]; then
    model_id="$MODEL_ID"
elif [ "$positive_strategy" = "distill" ]; then
    model_id=learnable_retriever_ctx512_dim256
else
    model_id=learnable_retriever_ctx512_dim256_${positive_strategy}
fi
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
oracle_top_m=${ORACLE_TOP_M:-1}
oracle_chunk_size=${ORACLE_CHUNK_SIZE:-4096}
oracle_query_batch_size=${ORACLE_QUERY_BATCH_SIZE:-32}
oracle_database_limit=${ORACLE_DATABASE_LIMIT:-}
oracle_device=${ORACLE_DEVICE:-cuda}
oracle_devices=${ORACLE_DEVICES:-0,1}
oracle_dtype=${ORACLE_DTYPE:-float16}

cmd=(python "$run_file" \
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
    --positive_strategy "$positive_strategy" \
    --oracle_top_m "$oracle_top_m" \
    --oracle_chunk_size "$oracle_chunk_size" \
    --oracle_query_batch_size "$oracle_query_batch_size" \
    --oracle_device "$oracle_device" \
    --oracle_devices "$oracle_devices" \
    --oracle_dtype "$oracle_dtype" \
    --train_steps "$train_steps" \
    --evaluation_steps "$evaluation_steps" \
    --batch_size "$batch_size" \
    --shuffle_buffer_length "$shuffle_buffer_length" \
    --learning_rate "$lr" \
    --weight_decay "$weight_decay" \
    --gpu_loc "$gpu_loc")

if [ -n "$oracle_database_limit" ]; then
    cmd+=(--oracle_database_limit "$oracle_database_limit")
fi

printf 'Running command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
