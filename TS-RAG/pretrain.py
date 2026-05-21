import os
import time
import wandb
import torch
import random
import argparse
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from chronos import ChronosPipeline
from transformers import AutoConfig
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from models.moment import MOMENTPipelineWithRetrieval
from dataset import CustomPretrainDataset, Retriever_for_pretrain
from models.learnable_retriever import (
    LearnableProjector,
    InBatchContrastiveRetrievalLoss,
    load_projector_checkpoint,
    save_projector_checkpoint,
)
from models.ChronosBolt import ChronosBoltModelForForecasting, ChronosBoltModelForForecastingWithRetrieval
    
warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='ChronosBoltRetrieve')

parser.add_argument('--model_id', type=str, default='ChronosBoltRetrieve_Pretrain')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

# retrieve
parser.add_argument('--embedding_tuning', type=str, default=None)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--embedding_model_type', type=str, default='chronos')
parser.add_argument('--retrieve_lookback_length', type=int, default=64)
parser.add_argument('--retrieval_database_path', type=str, default='../database/pretrain/retrieval_database_512.parquet')
parser.add_argument('--joint_train_retriever', action='store_true', default=False)
parser.add_argument('--retriever_projector_path', type=str, default=None)
parser.add_argument('--retriever_projector_input_dim', type=int, default=768)
parser.add_argument('--retriever_projector_hidden_dim', type=int, default=512)
parser.add_argument('--retriever_projector_output_dim', type=int, default=256)
parser.add_argument('--retriever_projector_dropout', type=float, default=0.0)
parser.add_argument('--retriever_temperature', type=float, default=0.07)
parser.add_argument('--retriever_cl_lambda', type=float, default=0.05)
parser.add_argument('--retriever_positive_strategy', type=str, default='oracle', choices=['oracle', 'distill'])
parser.add_argument('--retriever_positive_rank', type=int, default=0)
parser.add_argument('--oracle_top_m', type=int, default=1)
parser.add_argument('--oracle_sample_top_m', action='store_true', default=False)
parser.add_argument('--oracle_chunk_size', type=int, default=4096)
parser.add_argument('--oracle_query_batch_size', type=int, default=32)
parser.add_argument('--oracle_database_limit', type=int, default=None)
parser.add_argument('--chronos_model_path', type=str, default='./checkpoints/chronos-t5-base')

# augment
parser.add_argument('--augment_mode', type=str, default='moe2')

# model
parser.add_argument('--model', type=str, default='ChronosBoltRetrieve')
parser.add_argument('--freeze_chronos_bolt', action='store_true', help="freeze the params of chronos-bolt.")
parser.add_argument('--pretrained_model_path', type=str, default='./checkpoints/base/')
parser.add_argument('--context_length', type=int, default=512)
parser.add_argument('--prediction_length', type=int, default=64)

# pretrain
parser.add_argument('--data_path', type=str, default='../datasets/pretrain/50m-with-retrieval_512', help='pretrain data path')
parser.add_argument('--train_steps', type=int, default=200_000)
parser.add_argument('--evaluation_steps', type=int, default=10_000)
parser.add_argument('--optimizer', type=str, default='adamw')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--tmax', type=int, default=20)
parser.add_argument('--drop_prob', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--shuffle_buffer_length', type=int, default=100_000)
parser.add_argument('--grad_clip_value', type=float, default=1.0)

# gpu
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--gpu_loc', type=int, default=0, help='main gpu location')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)


args = parser.parse_args()

# init wandb project
wandb.init(project=f'{args.model}_Pretrain', name=args.model_id)
wandb.config.update(args)


device = 'cuda:'+str(args.gpu_loc)
torch_device = torch.device(device if torch.cuda.is_available() else 'cpu')
print(
    "ARM pretrain runtime | "
    f"cuda_available={torch.cuda.is_available()} | "
    f"cuda_device_count={torch.cuda.device_count()} | "
    f"use_multi_gpu={args.use_multi_gpu} | "
    f"gpu_loc={args.gpu_loc} | devices={args.devices} | "
    f"joint_train_retriever={args.joint_train_retriever} | "
    f"retriever_positive_strategy={args.retriever_positive_strategy}"
)

time_now = time.time()

## load model, optimizer
config = AutoConfig.from_pretrained(args.pretrained_model_path)
if args.model == 'ChronosBolt':
    model = ChronosBoltModelForForecasting.from_pretrained(args.pretrained_model_path, config=config)
    model.load_state_dict(torch.load('./checkpoints/base/autogluon_model.pth'), strict=False)
elif args.model == 'ChronosBoltRetrieve':
    model = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(args.pretrained_model_path, config=config, augment=args.augment_mode)
    model.load_state_dict(torch.load('./checkpoints/base/autogluon_model.pth'), strict=False)
    if 'moe' in args.augment_mode:
        model.init_extra_weights([model.encode_mlp, model.mha, model.ffn, model.gate_layer])
    if 'gate' in args.augment_mode:
        model.init_extra_weights([model.gate_layer, model.gate_linear1, model.gate_linear2])
elif args.model == 'MOMENTRetrieve':
    MOMENT_MODEL_PATH = "AutonLab/MOMENT-1-large"
    model = MOMENTPipelineWithRetrieval.from_pretrained(MOMENT_MODEL_PATH,
                                           model_kwargs={
                                               'task_name': 'forecasting',
                                               'forecast_horizon': 64,
                                           })
    model.init()
    if 'moe' in args.augment_mode:
        model.init_extra_weights([model.encode_mlp, model.mha, model.ffn, model.gate_layer, model.project_before_fusion, model.project_after_fusion])
    criterion = nn.MSELoss().to(device)
else:
    print('model error')
    exit()
print(f'{args.model} model loaded')

# freeze params
if args.freeze_chronos_bolt:
    layers_to_unfreeze = ['gate_layer', 'encode_mlp', 'mha', 'ffn']
    if args.augment_mode == 'moe3':
        if args.model == 'ChronosBoltRetrieve':
            layers_to_unfreeze.append('output_patch_embedding')
        elif args.model == 'MOMENTRetrieve':
            # import pdb; pdb.set_trace()
            layers_to_unfreeze.append('head')
    elif args.augment_mode == 'gate':
        layers_to_unfreeze.append('gate_linear1')
        layers_to_unfreeze.append('gate_linear2')

    for param in model.parameters():
        param.requires_grad = False
    # unfreeze the specified layers
    for name, param in model.named_parameters():
        param.requires_grad = any(layer in name for layer in layers_to_unfreeze)

retriever_projector = None
retriever_criterion = None
chronos_embedding_model = None
retriever_database_embeddings = None
retriever_database_horizons = None

def load_joint_retriever_database(path, need_horizon=False, limit=None):
    columns = ['embedding', 'y'] if need_horizon else ['embedding']
    database = pd.read_parquet(path, columns=columns)
    if limit is not None:
        database = database.iloc[:limit]
    embeddings = np.vstack(database['embedding'].to_numpy()).astype('float32')
    horizons = None
    if need_horizon:
        horizons = np.vstack(database['y'].to_numpy()).astype('float32')
    return embeddings, horizons

def merge_top_m(best_scores, best_indices, chunk_scores, chunk_offset, top_m):
    chunk_indices = np.arange(chunk_offset, chunk_offset + chunk_scores.shape[1], dtype=np.int64)
    chunk_indices = np.broadcast_to(chunk_indices[None, :], chunk_scores.shape)
    scores = np.concatenate([best_scores, chunk_scores], axis=1)
    indices = np.concatenate([best_indices, chunk_indices], axis=1)
    selected = np.argpartition(scores, kth=top_m - 1, axis=1)[:, :top_m]
    row_ids = np.arange(scores.shape[0])[:, None]
    selected_scores = scores[row_ids, selected]
    order = np.argsort(selected_scores, axis=1)
    selected = selected[row_ids, order]
    return scores[row_ids, selected], indices[row_ids, selected]

def oracle_positive_indices(query_y, database_horizons, top_m, chunk_size, query_batch_size, sample_top_m):
    query_y = np.asarray(query_y, dtype='float32')
    if query_y.ndim == 3:
        query_y = query_y.squeeze(-1)
    output = []
    for q_start in range(0, query_y.shape[0], query_batch_size):
        q = query_y[q_start:q_start + query_batch_size]
        best_scores = np.full((q.shape[0], top_m), np.inf, dtype='float32')
        best_indices = np.full((q.shape[0], top_m), -1, dtype=np.int64)
        for db_start in range(0, database_horizons.shape[0], chunk_size):
            chunk = database_horizons[db_start:db_start + chunk_size]
            mse = ((q[:, None, :] - chunk[None, :, :]) ** 2).mean(axis=-1)
            best_scores, best_indices = merge_top_m(best_scores, best_indices, mse.astype('float32'), db_start, top_m)
        if sample_top_m and top_m > 1:
            output.extend([np.random.choice(row[row >= 0]) for row in best_indices])
        else:
            output.extend(best_indices[:, 0].tolist())
    return np.asarray(output, dtype=np.int64)

if args.joint_train_retriever:
    if args.retriever_projector_path:
        retriever_projector = load_projector_checkpoint(args.retriever_projector_path, device=torch_device)
    else:
        retriever_projector = LearnableProjector(
            input_dim=args.retriever_projector_input_dim,
            hidden_dim=args.retriever_projector_hidden_dim,
            output_dim=args.retriever_projector_output_dim,
            dropout=args.retriever_projector_dropout,
        ).to(torch_device)
    retriever_projector.train()
    retriever_criterion = InBatchContrastiveRetrievalLoss(temperature=args.retriever_temperature)
    chronos_embedding_model = ChronosPipeline.from_pretrained(
        args.chronos_model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    retriever_database_embeddings, retriever_database_horizons = load_joint_retriever_database(
        args.retrieval_database_path,
        need_horizon=args.retriever_positive_strategy == 'oracle',
        limit=args.oracle_database_limit,
    )

model.to(device)
if args.use_multi_gpu:
    args.devices = [int(i) for i in args.devices.split(',')]
    if torch.cuda.is_available() and len(args.devices) > torch.cuda.device_count():
        raise ValueError(
            f"Requested devices {args.devices}, but only {torch.cuda.device_count()} CUDA devices are visible."
        )
    model = nn.DataParallel(model, device_ids=args.devices)
    print(f"Wrapped ARM model with DataParallel on devices: {args.devices}")
else:
    print(f"Using single GPU/CPU for ARM model: {device}")

params = [p for p in model.parameters() if p.requires_grad]
if retriever_projector is not None:
    params += list(retriever_projector.parameters())

if args.optimizer == 'adam':
    model_optim = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
elif args.optimizer == 'adamw':
    model_optim = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

# retrieval already done, do not need to load the embedding model
embedding_model = None

# load retriever
retriever = Retriever_for_pretrain(
    retrieval_database_path=args.retrieval_database_path,
    dimension=768,
    embedding_model=embedding_model,
)
retriever.build_index()

## load data
dataset = CustomPretrainDataset(
    args.data_path, 
    retriever=retriever, 
    mode='training',
    drop_prob=args.drop_prob,
    context_length=args.context_length,
    prediction_length=args.prediction_length,
    retrieve_lookback_length=args.retrieve_lookback_length,
    top_k=args.top_k,
).shuffle(shuffle_buffer_length=args.shuffle_buffer_length)

train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)


## train
is_first = True 
iter_count = 0
train_loss = []
for i, batch in tqdm(enumerate(train_loader)):
    if i >= args.train_steps:
        print('training finished')
        break
    if is_first : 
        print(args.data_path, batch['x'].shape, batch['y'].shape, batch['distances'].shape, batch['indices'].shape)
        is_first=False
        
    iter_count += 1
    model_optim.zero_grad()
    retrieved_seqs = torch.tensor(retriever.whole_seq[batch['indices']])
    
    if not args.use_multi_gpu:
        batch['x'] = batch['x'].float().to(device)
        batch['y'] = batch['y'].float().to(device)
        batch['distances'] = batch['distances'].float().to(device)
        retrieved_seqs = retrieved_seqs.float().to(device)
    if args.model == 'ChronosBoltRetrieve':
        outputs = model(context = batch['x'].float(), 
                        target = batch['y'].float(),
                        retrieved_seq = retrieved_seqs.float(), 
                        distances = batch['distances'].float())                  # ChronosBoltOutput
    elif args.model == 'MOMENTRetrieve':
        outputs = model(x_enc=batch['x'].float().unsqueeze(1), retrieved_seq=retrieved_seqs.float())
        outputs = outputs.forecast.squeeze(1)                                                     
        loss = criterion(outputs, batch['y'].float())
    else:
        print('model error')
    if args.model == 'MOMENTRetrieve':
        pass
    else:
        loss = outputs.loss
    forecast_loss = loss.mean()
    cl_loss = None
    loss = forecast_loss

    if args.joint_train_retriever:
        with torch.no_grad():
            query_embeddings, _ = chronos_embedding_model.embed(batch['x'].float().detach().cpu())
            query_embeddings = query_embeddings[:, -1, :].float().to(torch_device)
            if args.retriever_positive_strategy == 'oracle':
                positive_indices = oracle_positive_indices(
                    batch['y'].float().detach().cpu().numpy(),
                    retriever_database_horizons,
                    top_m=args.oracle_top_m,
                    chunk_size=args.oracle_chunk_size,
                    query_batch_size=args.oracle_query_batch_size,
                    sample_top_m=args.oracle_sample_top_m,
                )
            else:
                rank = min(args.retriever_positive_rank, batch['indices'].shape[1] - 1)
                positive_indices = batch['indices'][:, rank].detach().cpu().numpy()
            positive_embeddings = torch.from_numpy(retriever_database_embeddings[positive_indices]).float().to(torch_device)

        query_projected = retriever_projector(query_embeddings)
        positive_projected = retriever_projector(positive_embeddings)
        cl_loss = retriever_criterion(query_projected, positive_projected)
        loss = forecast_loss + args.retriever_cl_lambda * cl_loss

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        log_payload = {
            'loss': loss.item(),
            'forecast_loss': forecast_loss.item(),
            'lr': model_optim.param_groups[0]['lr']
        }
        if cl_loss is not None:
            log_payload['retriever_cl_loss'] = cl_loss.item()
            log_payload['retriever_cl_lambda'] = args.retriever_cl_lambda
        wandb.log(log_payload)

    train_loss.append(loss.item())

    if (i + 1) % args.evaluation_steps == 0:
        print("\titers: {0} | loss: {1:.7f}".format(i + 1, sum(train_loss) / len(train_loss)))
        train_loss = []
        speed = (time.time() - time_now) / iter_count
        print('\tspeed: {:.4f}s/iter'.format(speed))
        iter_count = 0
        time_now = time.time()
        # save model and optimizer
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            save_path = os.path.join(args.checkpoints, args.model_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path,f'model_steps{i}.pth'))
            torch.save(model_optim.state_dict(), os.path.join(save_path, f'optim_steps{i}.pth'))
            if retriever_projector is not None:
                save_projector_checkpoint(os.path.join(save_path, f'projector_steps{i}.pth'), retriever_projector, args)

        # adjust learning rate
        scheduler.step()
        print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))

    loss.backward()
    clip_grad_norm_(model.parameters(), args.grad_clip_value)
    if retriever_projector is not None:
        clip_grad_norm_(retriever_projector.parameters(), args.grad_clip_value)
    model_optim.step()
                
