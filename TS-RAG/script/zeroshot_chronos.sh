export CUDA_VISIBLE_DEVICES="0"
filename=zeroshot_chronos.txt 
model=ChronosBoltRetrieve
gpu_loc=0
run_file=zeroshot.py
seq_len=512
pred_len=64
datasets="ETTh1"
lookback_length=512
augment_mode=moe
top_k=10

batch_size=256
retrieval_database_dir='../retrieval_database/'

checkpoint_model_path="./checkpoints/chronos-bolt/best.pth"

# Optional learnable retriever projector. Leave empty to use the original fixed
# Chronos embedding + FAISS L2 retriever.
retriever_projector_path=${RETRIEVER_PROJECTOR_PATH:-""}
retrieval_tag=${RETRIEVAL_TAG:-"learnable_retriever"}
retriever_projector_output_dim=256
retriever_projector_similarity=cosine

# top_k_h=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20) 

# for top_k in ${top_k_h[@]};
# do
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

retriever_args=()
if [ -n "$retriever_projector_path" ]; then
    retriever_args+=(
        --retriever_projector_path "$retriever_projector_path"
        --retriever_projector_output_dim "$retriever_projector_output_dim"
        --retriever_projector_similarity "$retriever_projector_similarity"
        --retrieval_tag "$retrieval_tag"
    )
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
    --augment_mode $augment_mode \
    "${retriever_args[@]}"

done
