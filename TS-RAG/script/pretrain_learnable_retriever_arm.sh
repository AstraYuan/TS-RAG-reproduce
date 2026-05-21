# WANDB_MODE=offline bash script/pretrain_learnable_retriever_arm.sh > ../logs/arm_pretrain_learnable_retriever.log 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

run_file=pretrain.py

# model
top_k=${TOP_K:-10}
retrieve_lookback_length=${RETRIEVE_LOOKBACK_LENGTH:-512}
retrieval_database_path=${RETRIEVAL_DATABASE_PATH:-../retrieval_database/pretrain/retrieval_database_${retrieve_lookback_length}.parquet}
augment_mode=${AUGMENT_MODE:-moe}
context_length=${CONTEXT_LENGTH:-512}
prediction_length=${PREDICTION_LENGTH:-64}

# pretrain
data_path=${DATA_PATH:-../datasets/pretrain/pretrain_pairs_ctx${retrieve_lookback_length}_learnable_retriever}
train_steps=${TRAIN_STEPS:-10000}
evaluation_steps=${EVALUATION_STEPS:-10000}
optimizer=${OPTIMIZER:-adamw}
lr=${LR:-0.0003}
weight_decay=${WEIGHT_DECAY:-0.01}
tmax=${TMAX:-20}
drop_prob=${DROP_PROB:-0.2}
batch_size=${BATCH_SIZE:-256}
shuffle_buffer_length=${SHUFFLE_BUFFER_LENGTH:-10000}

# gpu
gpu_loc=${GPU_LOC:-0}
devices=${DEVICES:-"0,1"}
use_multi_gpu=${USE_MULTI_GPU:-1}

# optional joint retriever objective
joint_train_retriever=${JOINT_TRAIN_RETRIEVER:-1}
retriever_projector_path=${RETRIEVER_PROJECTOR_PATH:-./checkpoints/learnable-retriever/learnable_retriever_ctx512_dim256_oracle/projector_final.pth}
chronos_model_path=${CHRONOS_MODEL_PATH:-./checkpoints/chronos-t5-base}
retriever_cl_lambda=${RETRIEVER_CL_LAMBDA:-0.05}
retriever_positive_strategy=${RETRIEVER_POSITIVE_STRATEGY:-oracle}
oracle_top_m=${ORACLE_TOP_M:-1}
oracle_chunk_size=${ORACLE_CHUNK_SIZE:-4096}
oracle_query_batch_size=${ORACLE_QUERY_BATCH_SIZE:-32}
oracle_database_limit=${ORACLE_DATABASE_LIMIT:-}

model_id=${MODEL_ID:-"data50m_${augment_mode}_${context_length}_pred${prediction_length}_lookback${retrieve_lookback_length}_top${top_k}_lr${lr}_drop${drop_prob}_${optimizer}_cosanneal_step${train_steps}_bs${batch_size}_learnable_retriever"}

cmd=(python "$run_file"
    --model_id "$model_id"
    --top_k "$top_k"
    --retrieve_lookback_length "$retrieve_lookback_length"
    --retrieval_database_path "$retrieval_database_path"
    --augment_mode "$augment_mode"
    --context_length "$context_length"
    --prediction_length "$prediction_length"
    --data_path "$data_path"
    --train_steps "$train_steps"
    --evaluation_steps "$evaluation_steps"
    --optimizer "$optimizer"
    --learning_rate "$lr"
    --weight_decay "$weight_decay"
    --tmax "$tmax"
    --drop_prob "$drop_prob"
    --batch_size "$batch_size"
    --shuffle_buffer_length "$shuffle_buffer_length"
    --gpu_loc "$gpu_loc"
    --devices "$devices"
    --freeze_chronos_bolt)

if [ "$use_multi_gpu" -eq 1 ]; then
    cmd+=(--use_multi_gpu)
fi

if [ "$joint_train_retriever" -eq 1 ]; then
    cmd+=(
        --joint_train_retriever
        --retriever_projector_path "$retriever_projector_path"
        --chronos_model_path "$chronos_model_path"
        --retriever_cl_lambda "$retriever_cl_lambda"
        --retriever_positive_strategy "$retriever_positive_strategy"
        --oracle_top_m "$oracle_top_m"
        --oracle_chunk_size "$oracle_chunk_size"
        --oracle_query_batch_size "$oracle_query_batch_size"
    )
    if [ -n "$oracle_database_limit" ]; then
        cmd+=(--oracle_database_limit "$oracle_database_limit")
    fi
fi

printf 'Running command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
