#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

filename=${SAVE_FILE_NAME:-zeroshot_chronos_context_aware.txt}
model=ChronosBoltRetrieve
gpu_loc=0
run_file=zeroshot.py
seq_len=512
pred_len=64
datasets=${DATASETS:-"ETTh1 ETTh2 ETTm1 ETTm2 weather electricity exchange_rate"}
lookback_length=512
augment_mode=context
top_k=10

batch_size=256
retrieval_database_dir='../retrieval_database/'

default_context_model_id="data50m_${augment_mode}_${seq_len}_pred${pred_len}_lookback${lookback_length}_top${top_k}_lr0.0003_drop0.2_adamw_cosanneal_step10000_bs256_no_embeddingtuning"
checkpoint_model_path=${CONTEXT_AWARE_CHECKPOINT_PATH:-"./checkpoints/${default_context_model_id}/model_steps9999.pth"}

for dataset in $datasets;
do
retrieve_database_name=$dataset

if [ $dataset == 'ETTm1' ] || [ $dataset == 'ETTm2' ]; then
    data='ett_m_retrieve'
    metadata_frequency='minute'
    root_path='../datasets/ETT-small/'
elif [ $dataset == 'ETTh1' ] || [ $dataset == 'ETTh2' ]; then
    data='ett_h_retrieve'
    metadata_frequency='hour'
    root_path='../datasets/ETT-small/'
elif [ $dataset == 'electricity' ] || [ $dataset == 'exchange_rate' ]; then
    data='custom_retrieve'
    metadata_frequency='hour'
    root_path="../datasets/${dataset}/"
elif [ $dataset == 'weather' ]; then
    data='custom_retrieve'
    metadata_frequency='10minutes'
    root_path="../datasets/${dataset}/"
fi

python $run_file \
    --root_path $root_path \
    --data_path $dataset'.csv' \
    --model_id $dataset'_zeroshot_'$seq_len'_pred_'$pred_len'_'$lookback_length'_retrieve_'$pred_len \
    --data $data \
    --top_k $top_k \
    --checkpoint_model_path $checkpoint_model_path \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --lookback_length $lookback_length \
    --batch_size  $batch_size \
    --decay_fac 0.5 \
    --freq 0 \
    --percent 100 \
    --model $model \
    --gpu_loc $gpu_loc \
    --tmax 20 \
    --cos 1 \
    --save_file_name $filename \
    --retrieval_database_dir $retrieval_database_dir \
    --dimension 768 \
    --embedding_model_type chronos \
    --metadata_frequency $metadata_frequency \
    --metadata_database_name $retrieve_database_name \
    --augment_mode $augment_mode

done
