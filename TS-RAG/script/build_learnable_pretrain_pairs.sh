export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

run_file=build_learnable_pretrain_pairs.py

context_length=${CONTEXT_LENGTH:-512}
top_k=${TOP_K:-10}

source_data_path=${SOURCE_DATA_PATH:-../datasets/pretrain/pretrain_pairs_ctx${context_length}}
output_data_path=${OUTPUT_DATA_PATH:-../datasets/pretrain/pretrain_pairs_ctx${context_length}_learnable_retriever}
retrieval_database_path=${RETRIEVAL_DATABASE_PATH:-../retrieval_database/pretrain/retrieval_database_${context_length}.parquet}
chronos_model_path=${CHRONOS_MODEL_PATH:-./checkpoints/chronos-t5-base}
retriever_projector_path=${RETRIEVER_PROJECTOR_PATH:-./checkpoints/learnable-retriever/learnable_retriever_ctx512_dim256/projector_final.pth}

projector_output_dim=${RETRIEVER_PROJECTOR_OUTPUT_DIM:-256}
projector_similarity=${RETRIEVER_PROJECTOR_SIMILARITY:-cosine}
projector_batch_size=${RETRIEVER_PROJECTOR_BATCH_SIZE:-4096}
embedding_batch_size=${EMBEDDING_BATCH_SIZE:-128}
gpu_loc=${GPU_LOC:-0}

cmd=(python "$run_file"
    --source_data_path "$source_data_path"
    --output_data_path "$output_data_path"
    --retrieval_database_path "$retrieval_database_path"
    --chronos_model_path "$chronos_model_path"
    --retriever_projector_path "$retriever_projector_path"
    --context_length "$context_length"
    --top_k "$top_k"
    --projector_output_dim "$projector_output_dim"
    --projector_similarity "$projector_similarity"
    --projector_batch_size "$projector_batch_size"
    --embedding_batch_size "$embedding_batch_size"
    --gpu_loc "$gpu_loc")

if [ "${OVERWRITE:-0}" -eq 1 ]; then
    cmd+=(--overwrite)
fi

if [ -n "${LIMIT_FILES:-}" ]; then
    cmd+=(--limit_files "$LIMIT_FILES")
fi

"${cmd[@]}"
