import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze retrieval overlap and oracle future quality")
    parser.add_argument("--original_csv", type=str, required=True)
    parser.add_argument("--official_csv", type=str, required=True)
    parser.add_argument("--learnable_csv", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--pred_len", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--eval_tail", type=int, default=3000)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def parse_timestamp_columns(df, variables, top_k):
    result = {}
    for variable in variables:
        cols = [f"timestamp_idx_{variable}_{k}" for k in range(top_k)]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing timestamp columns: {missing[:3]}")
        result[variable] = df[cols].to_numpy(dtype=np.int64)
    return result


def future_mse(series, query_start, retrieved_start, seq_len, pred_len):
    query_future = series[query_start + seq_len:query_start + seq_len + pred_len]
    retrieved_future = series[retrieved_start + seq_len:retrieved_start + seq_len + pred_len]
    if len(query_future) != pred_len or len(retrieved_future) != pred_len:
        return np.nan
    return float(np.mean((query_future - retrieved_future) ** 2))


def ndcg_from_union(method_mses, union_mses, top_k):
    method_gains = 1.0 / (np.asarray(method_mses, dtype=np.float64) + 1e-8)
    union_gains = 1.0 / (np.asarray(union_mses, dtype=np.float64) + 1e-8)
    discounts = 1.0 / np.log2(np.arange(2, top_k + 2))
    dcg = np.sum(method_gains[:top_k] * discounts[:len(method_gains[:top_k])])
    ideal = np.sort(union_gains)[::-1][:top_k]
    idcg = np.sum(ideal * discounts[:len(ideal)])
    if idcg <= 0:
        return np.nan
    return float(dcg / idcg)


def recall_from_union_oracle(method_indices, union_indices, union_mses, top_k):
    pairs = [
        (idx, mse)
        for idx, mse in zip(union_indices, union_mses)
        if not np.isnan(mse)
    ]
    if not pairs:
        return np.nan
    oracle = [idx for idx, _ in sorted(pairs, key=lambda x: x[1])[:top_k]]
    return len(set(method_indices[:top_k].tolist()) & set(oracle)) / min(top_k, len(oracle))


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def main():
    args = parse_args()
    original = pd.read_csv(args.original_csv)
    official = pd.read_csv(args.official_csv)
    learnable = pd.read_csv(args.learnable_csv)

    variables = list(original.columns[1:])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(original[variables].to_numpy(dtype=np.float32))
    scaled_by_var = {variable: scaled[:, idx] for idx, variable in enumerate(variables)}

    official_ts = parse_timestamp_columns(official, variables, args.top_k)
    learnable_ts = parse_timestamp_columns(learnable, variables, args.top_k)

    max_start = min(len(original), len(official), len(learnable)) - args.seq_len - args.pred_len + 1
    start = max(0, max_start - args.eval_tail)
    rows = range(start, max_start)

    overlap_values = []
    official_best_mse = []
    official_mean_mse = []
    learnable_best_mse = []
    learnable_mean_mse = []
    official_ndcg = []
    learnable_ndcg = []
    official_recall = []
    learnable_recall = []

    for row in rows:
        for variable in variables:
            official_indices = official_ts[variable][row, :args.top_k]
            learnable_indices = learnable_ts[variable][row, :args.top_k]
            overlap_values.append(
                len(set(official_indices.tolist()) & set(learnable_indices.tolist())) / args.top_k
            )

            series = scaled_by_var[variable]
            official_mses = [
                future_mse(series, row, int(idx), args.seq_len, args.pred_len)
                for idx in official_indices
            ]
            learnable_mses = [
                future_mse(series, row, int(idx), args.seq_len, args.pred_len)
                for idx in learnable_indices
            ]
            union_indices = list(dict.fromkeys(official_indices.tolist() + learnable_indices.tolist()))
            union_mses = [
                future_mse(series, row, int(idx), args.seq_len, args.pred_len)
                for idx in union_indices
            ]

            official_best_mse.append(np.nanmin(official_mses))
            official_mean_mse.append(np.nanmean(official_mses))
            learnable_best_mse.append(np.nanmin(learnable_mses))
            learnable_mean_mse.append(np.nanmean(learnable_mses))
            official_ndcg.append(ndcg_from_union(official_mses, union_mses, args.top_k))
            learnable_ndcg.append(ndcg_from_union(learnable_mses, union_mses, args.top_k))
            official_recall.append(recall_from_union_oracle(official_indices, union_indices, union_mses, args.top_k))
            learnable_recall.append(recall_from_union_oracle(learnable_indices, union_indices, union_mses, args.top_k))

    metrics = {
        "eval_rows": [start, max_start],
        "num_variables": len(variables),
        "top_k": args.top_k,
        "overlap_at_k": summarize(overlap_values),
        "official_best_future_mse": summarize(official_best_mse),
        "official_retrieval_mse_at_k": summarize(official_mean_mse),
        "learnable_best_future_mse": summarize(learnable_best_mse),
        "learnable_retrieval_mse_at_k": summarize(learnable_mean_mse),
        "official_union_ndcg_at_k": summarize(official_ndcg),
        "learnable_union_ndcg_at_k": summarize(learnable_ndcg),
        "official_union_recall_at_k": summarize(official_recall),
        "learnable_union_recall_at_k": summarize(learnable_recall),
    }

    print(json.dumps(metrics, indent=2))
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
