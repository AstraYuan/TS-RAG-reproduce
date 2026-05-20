import os
import argparse
import random
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from chronos import ChronosPipeline

from dataset import Retriever_for_pretrain
from models.learnable_retriever import load_projector_checkpoint


warnings.filterwarnings("ignore")

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Build pretrain pairs with a learnable retriever")

    parser.add_argument("--source_data_path", type=str, default="../datasets/pretrain/pretrain_pairs_ctx512")
    parser.add_argument("--output_data_path", type=str, default="../datasets/pretrain/pretrain_pairs_ctx512_learnable_retriever")
    parser.add_argument("--retrieval_database_path", type=str, default="../retrieval_database/pretrain/retrieval_database_512.parquet")
    parser.add_argument("--chronos_model_path", type=str, default="amazon/chronos-t5-base")
    parser.add_argument("--retriever_projector_path", type=str, required=True)

    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--dimension", type=int, default=768)
    parser.add_argument("--projector_output_dim", type=int, default=256)
    parser.add_argument("--projector_batch_size", type=int, default=8192)
    parser.add_argument("--projector_similarity", type=str, default="cosine", choices=["l2", "cosine"])
    parser.add_argument("--embedding_batch_size", type=int, default=128)
    parser.add_argument("--gpu_loc", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1")
    parser.add_argument("--faiss_use_gpu", action="store_true", default=False)
    parser.add_argument("--faiss_gpu_devices", type=str, default="0,1")
    parser.add_argument("--limit_files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)

    return parser.parse_args()


def _target_to_array(value):
    if isinstance(value, np.ndarray):
        return value.astype("float32")
    return np.asarray(value, dtype="float32")


def _iter_parquet_files(path, limit_files=None):
    files = sorted(Path(path).glob("*.parquet"))
    if limit_files is not None:
        files = files[:limit_files]
    if not files:
        raise ValueError(f"No parquet files found in {path}")
    return files


def _embed_on_pipeline(targets, chronos, context_length):
    x = torch.tensor(np.stack([_target_to_array(v)[:context_length] for v in targets]), dtype=torch.float32)
    with torch.no_grad():
        embeddings, _ = chronos.embed(x)
    return embeddings[:, -1, :].float().cpu().numpy()


def embed_batch(targets, chronos, args):
    if not args.use_multi_gpu:
        return _embed_on_pipeline(targets, chronos, args.context_length)

    splits = np.array_split(np.asarray(list(targets), dtype=object), len(chronos))
    with ThreadPoolExecutor(max_workers=len(chronos)) as executor:
        parts = list(
            executor.map(
                lambda item: _embed_on_pipeline(item[0], item[1], args.context_length),
                [(split.tolist(), pipeline) for split, pipeline in zip(splits, chronos)],
            )
        )
    return np.concatenate(parts, axis=0)


def rebuild_file(file_path, output_path, chronos, retriever, args):
    df = pd.read_parquet(file_path)
    if "target" not in df.columns:
        raise ValueError(f"{file_path} does not contain a target column")

    all_indices = []
    all_distances = []

    for start in tqdm(range(0, len(df), args.embedding_batch_size), desc=f"rebuild {file_path.name}"):
        batch_targets = df["target"].iloc[start:start + args.embedding_batch_size].tolist()
        query_vector = embed_batch(batch_targets, chronos, args)

        indices, distances = retriever.search(query_vector, top_k=args.top_k)
        all_indices.extend(indices.astype("int64").tolist())
        all_distances.extend(distances.astype("float32").tolist())

    df["indices"] = all_indices
    df["distances"] = all_distances
    df.to_parquet(output_path, index=False)


def main():
    args = parse_args()
    source_data_path = Path(args.source_data_path)
    output_data_path = Path(args.output_data_path)
    output_data_path.mkdir(parents=True, exist_ok=True)

    device = f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu"
    args.primary_device = device
    print(
        "Build learnable pretrain pairs runtime | "
        f"cuda_available={torch.cuda.is_available()} | "
        f"cuda_device_count={torch.cuda.device_count()} | "
        f"use_multi_gpu={args.use_multi_gpu} | devices={args.devices} | "
        f"faiss_use_gpu={args.faiss_use_gpu} | faiss_gpu_devices={args.faiss_gpu_devices}"
    )
    projector_devices = None
    if args.use_multi_gpu and torch.cuda.is_available():
        projector_devices = [f"cuda:{idx}" for idx, _ in enumerate([d for d in args.devices.split(",") if d.strip()])]

    projector = load_projector_checkpoint(args.retriever_projector_path, device=device)
    args.projector_output_dim = getattr(projector, "output_dim", args.projector_output_dim)

    retriever = Retriever_for_pretrain(
        retrieval_database_path=args.retrieval_database_path,
        dimension=args.dimension,
        embedding_model=None,
        projector=projector,
        projector_device=device,
        projector_output_dim=args.projector_output_dim,
        projector_batch_size=args.projector_batch_size,
        similarity=args.projector_similarity,
        projector_devices=projector_devices,
        faiss_use_gpu=args.faiss_use_gpu,
        faiss_gpu_devices=args.faiss_gpu_devices,
    )
    retriever.build_index()

    if args.use_multi_gpu and torch.cuda.is_available():
        visible_devices = [d.strip() for d in args.devices.split(",") if d.strip()]
        chronos = [
            ChronosPipeline.from_pretrained(
                args.chronos_model_path,
                device_map=f"cuda:{idx}",
                torch_dtype=torch.bfloat16,
            )
            for idx, _ in enumerate(visible_devices)
        ]
    else:
        chronos = ChronosPipeline.from_pretrained(
            args.chronos_model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )

    for file_path in _iter_parquet_files(source_data_path, args.limit_files):
        output_path = output_data_path / file_path.name
        if output_path.exists() and not args.overwrite:
            print(f"Skip existing file: {output_path}")
            continue
        rebuild_file(file_path, output_path, chronos, retriever, args)

    print(f"Saved learnable pretrain pairs to {output_data_path}")


if __name__ == "__main__":
    main()
