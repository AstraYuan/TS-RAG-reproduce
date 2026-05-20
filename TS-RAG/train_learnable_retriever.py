import os
import time
import random
import argparse
import warnings

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from chronos import ChronosPipeline
from torch.utils.data import DataLoader

from dataset import CustomPretrainDataset
from models.learnable_retriever import (
    LearnableProjector,
    InBatchContrastiveRetrievalLoss,
    save_projector_checkpoint,
)


warnings.filterwarnings("ignore")

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


class _DatasetWithoutRetriever:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Train a learnable retrieval projector")

    parser.add_argument("--model_id", type=str, default="learnable_retriever_ctx512")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/learnable-retriever")

    parser.add_argument("--data_path", type=str, default="../datasets/pretrain/pretrain_pairs_ctx512")
    parser.add_argument("--retrieval_database_path", type=str, default="../retrieval_database/pretrain/retrieval_database_512.parquet")
    parser.add_argument("--chronos_model_path", type=str, default="amazon/chronos-t5-base")

    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--prediction_length", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--positive_rank", type=int, default=0)
    parser.add_argument("--positive_strategy", type=str, default="distill", choices=["distill", "oracle"])
    parser.add_argument("--oracle_top_m", type=int, default=1)
    parser.add_argument("--oracle_sample_top_m", action="store_true", default=False)
    parser.add_argument("--oracle_chunk_size", type=int, default=4096)
    parser.add_argument("--oracle_query_batch_size", type=int, default=32)
    parser.add_argument("--oracle_database_limit", type=int, default=None)

    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--output_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--evaluation_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--shuffle_buffer_length", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip_value", type=float, default=1.0)

    parser.add_argument("--gpu_loc", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)

    return parser.parse_args()


def load_retrieval_database(path, need_horizon=False, limit=None):
    columns = ["embedding", "y"] if need_horizon else ["embedding"]
    print(f"Loading retrieval database columns {columns} from {path}")
    database = pd.read_parquet(path, columns=columns)
    if limit is not None:
        database = database.iloc[:limit]

    embeddings = np.vstack(database["embedding"].to_numpy()).astype("float32")
    horizons = None
    if need_horizon:
        horizons = np.vstack(database["y"].to_numpy()).astype("float32")

    print(f"Loaded retrieval embeddings: {embeddings.shape}")
    if horizons is not None:
        print(f"Loaded retrieval horizons: {horizons.shape}")
    return embeddings, horizons


def _merge_top_m(best_scores, best_indices, chunk_scores, chunk_offset, top_m):
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


def oracle_positive_indices(
    query_y,
    database_horizons,
    top_m=1,
    chunk_size=4096,
    query_batch_size=32,
    sample_top_m=False,
):
    query_y = np.asarray(query_y, dtype="float32")
    if query_y.ndim == 3:
        query_y = query_y.squeeze(-1)

    output = []
    for q_start in range(0, query_y.shape[0], query_batch_size):
        q = query_y[q_start:q_start + query_batch_size]
        best_scores = np.full((q.shape[0], top_m), np.inf, dtype="float32")
        best_indices = np.full((q.shape[0], top_m), -1, dtype=np.int64)

        for db_start in range(0, database_horizons.shape[0], chunk_size):
            chunk = database_horizons[db_start:db_start + chunk_size]
            mse = ((q[:, None, :] - chunk[None, :, :]) ** 2).mean(axis=-1)
            best_scores, best_indices = _merge_top_m(
                best_scores,
                best_indices,
                mse.astype("float32"),
                db_start,
                top_m,
            )

        if sample_top_m and top_m > 1:
            sampled = []
            for row in best_indices:
                valid = row[row >= 0]
                sampled.append(np.random.choice(valid))
            output.extend(sampled)
        else:
            output.extend(best_indices[:, 0].tolist())

    return np.asarray(output, dtype=np.int64)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu")

    chronos = ChronosPipeline.from_pretrained(
        args.chronos_model_path,
        device_map=str(device),
        torch_dtype=torch.bfloat16,
    )

    database_embeddings, database_horizons = load_retrieval_database(
        args.retrieval_database_path,
        need_horizon=args.positive_strategy == "oracle",
        limit=args.oracle_database_limit,
    )

    projector = LearnableProjector(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
    ).to(device)
    criterion = InBatchContrastiveRetrievalLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    dataset = CustomPretrainDataset(
        args.data_path,
        retriever=_DatasetWithoutRetriever(),
        mode="training",
        drop_prob=0.0,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        retrieve_lookback_length=args.context_length,
        top_k=args.top_k,
    ).shuffle(shuffle_buffer_length=args.shuffle_buffer_length)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    save_dir = os.path.join(args.checkpoints, args.model_id)
    os.makedirs(save_dir, exist_ok=True)

    time_now = time.time()
    losses = []
    projector.train()

    for step, batch in tqdm(enumerate(train_loader)):
        if step >= args.train_steps:
            break

        x = batch["x"].float()
        y = batch["y"].float()
        indices = batch["indices"].long()
        if args.positive_strategy == "oracle":
            positive_indices = oracle_positive_indices(
                y.cpu().numpy(),
                database_horizons,
                top_m=args.oracle_top_m,
                chunk_size=args.oracle_chunk_size,
                query_batch_size=args.oracle_query_batch_size,
                sample_top_m=args.oracle_sample_top_m,
            )
        else:
            rank = min(args.positive_rank, indices.shape[1] - 1)
            positive_indices = indices[:, rank].cpu().numpy()

        with torch.no_grad():
            query_embeddings, _ = chronos.embed(x)
            query_embeddings = query_embeddings[:, -1, :].float().to(device)
            positive_embeddings = torch.from_numpy(database_embeddings[positive_indices]).float().to(device)

        optimizer.zero_grad()
        query_projected = projector(query_embeddings)
        positive_projected = projector(positive_embeddings)
        loss = criterion(query_projected, positive_projected)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(projector.parameters(), args.grad_clip_value)
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % args.evaluation_steps == 0:
            avg_loss = sum(losses) / len(losses)
            speed = (time.time() - time_now) / len(losses)
            print(f"steps: {step + 1} | contrastive_loss: {avg_loss:.6f} | speed: {speed:.4f}s/iter")
            ckpt_path = os.path.join(save_dir, f"projector_steps{step}.pth")
            save_projector_checkpoint(ckpt_path, projector, args)
            losses = []
            time_now = time.time()

    final_path = os.path.join(save_dir, "projector_final.pth")
    save_projector_checkpoint(final_path, projector, args)
    print(f"Saved learnable retriever projector to {final_path}")


if __name__ == "__main__":
    main()
