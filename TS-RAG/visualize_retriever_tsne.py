import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from dataset import CustomPretrainDataset
from models.learnable_retriever import load_projector_checkpoint


class _DatasetWithoutRetriever:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize raw/projected retrieval embedding spaces with t-SNE")
    parser.add_argument("--data_path", type=str, default="../datasets/pretrain/pretrain_pairs_ctx512")
    parser.add_argument("--retrieval_database_path", type=str, default="../retrieval_database/pretrain/retrieval_database_512.parquet")
    parser.add_argument("--chronos_model_path", type=str, default="./checkpoints/chronos-t5-base")
    parser.add_argument("--retriever_projector_path", type=str, required=True)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--prediction_length", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--positive_rank", type=int, default=0)
    parser.add_argument("--sample_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu_loc", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/retrieval_quality")
    return parser.parse_args()


def collect_query_embeddings(args, device):
    chronos = ChronosPipeline.from_pretrained(
        args.chronos_model_path,
        device_map=str(device),
        torch_dtype=torch.bfloat16,
    )
    dataset = CustomPretrainDataset(
        args.data_path,
        retriever=_DatasetWithoutRetriever(),
        mode="validation",
        drop_prob=0.0,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        retrieve_lookback_length=args.context_length,
        top_k=args.top_k,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    query_embeddings = []
    positive_indices = []
    for batch in loader:
        with torch.no_grad():
            embeddings, _ = chronos.embed(batch["x"].float())
        query_embeddings.append(embeddings[:, -1, :].float().cpu().numpy())
        rank = min(args.positive_rank, batch["indices"].shape[1] - 1)
        positive_indices.extend(batch["indices"][:, rank].long().cpu().numpy().tolist())
        if len(positive_indices) >= args.sample_size:
            break

    query_embeddings = np.concatenate(query_embeddings, axis=0)[:args.sample_size].astype("float32")
    positive_indices = np.asarray(positive_indices[:args.sample_size], dtype=np.int64)
    return query_embeddings, positive_indices


def plot_tsne(points, labels, title, output_path):
    perplexity = min(30, max(5, points.shape[0] // 10))
    coords = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(points)
    plt.figure(figsize=(7, 6))
    for label, marker in [("query", "o"), ("positive", "^")]:
        mask = labels == label
        plt.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.65, label=label, marker=marker)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_loc}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_embeddings, positive_indices = collect_query_embeddings(args, device)
    database = pd.read_parquet(args.retrieval_database_path, columns=["embedding"])
    database_embeddings = np.vstack(database["embedding"].iloc[positive_indices].to_numpy()).astype("float32")

    raw_points = np.concatenate([query_embeddings, database_embeddings], axis=0)
    labels = np.asarray(["query"] * len(query_embeddings) + ["positive"] * len(database_embeddings))
    plot_tsne(raw_points, labels, "Raw Chronos embedding space", output_dir / "tsne_raw.png")

    projector = load_projector_checkpoint(args.retriever_projector_path, device=device)
    projector.eval()
    with torch.no_grad():
        query_projected = projector(torch.from_numpy(query_embeddings).float().to(device)).cpu().numpy()
        positive_projected = projector(torch.from_numpy(database_embeddings).float().to(device)).cpu().numpy()
    projected_points = np.concatenate([query_projected, positive_projected], axis=0)
    plot_tsne(projected_points, labels, "Learnable retriever projected space", output_dir / "tsne_projected.png")

    print(f"Saved t-SNE plots to {output_dir}")


if __name__ == "__main__":
    main()
