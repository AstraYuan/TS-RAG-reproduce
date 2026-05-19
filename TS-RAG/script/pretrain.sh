# WANDB_MODE=offline bash script/pretrain.sh > ../logs/arm_pretrain.log 2>&1

export CUDA_VISIBLE_DEVICES="0,1"

run_file=pretrain.py

# model
top_k=10
retrieve_lookback_length=512
retrieval_database_path=../retrieval_database/pretrain/retrieval_database_${retrieve_lookback_length}.parquet
augment_mode=moe
context_length=512
prediction_length=64

# pretrain
data_path=../datasets/pretrain/pretrain_pairs_ctx${retrieve_lookback_length}
train_steps=10000
evaluation_steps=10000
optimizer=adamw
lr=0.0003
weight_decay=0.01
tmax=20
# drop_prob=0.0
drop_prob=0.2
batch_size=256
shuffle_buffer_length=10000

# gpu
gpu_loc=0
devices="0,1"
use_multi_gpu=1

model_id="data50m_${augment_mode}_${context_length}_pred${prediction_length}_lookback${retrieve_lookback_length}_top${top_k}_lr${lr}_drop${drop_prob}_${optimizer}_cosanneal_step${train_steps}_bs${batch_size}_no_embeddingtuning"

cmd=(python "$run_file"
    --model_id $model_id \
    --top_k $top_k \
    --retrieve_lookback_length $retrieve_lookback_length \
    --retrieval_database_path $retrieval_database_path \
    --augment_mode $augment_mode \
    --context_length $context_length \
    --prediction_length $prediction_length \
    --data_path $data_path \
    --train_steps $train_steps \
    --evaluation_steps $evaluation_steps \
    --optimizer $optimizer \
    --learning_rate $lr \
    --weight_decay $weight_decay \
    --tmax $tmax \
    --drop_prob $drop_prob \
    --batch_size $batch_size \
    --shuffle_buffer_length $shuffle_buffer_length \
    --gpu_loc $gpu_loc \
    --devices $devices \
    --freeze_chronos_bolt)

if [ "$use_multi_gpu" -eq 1 ]; then
    cmd+=(--use_multi_gpu)
fi

"${cmd[@]}"
