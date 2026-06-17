from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from dscript.models.interaction_res_bidirect import InteractionInputs
from dscript.parallel_embedding_loader import LazyEmbeddingStore, add_batch_dim_if_needed


def predict_pair(model, embeddings, n0, n1, use_cuda):
    z0 = add_batch_dim_if_needed(embeddings[n0])
    z1 = add_batch_dim_if_needed(embeddings[n1])
    if use_cuda:
        z0 = z0.cuda()
        z1 = z1.cuda()
    with torch.no_grad():
        _, logit, _ = model.map_predict(InteractionInputs(z0, z1))
        return torch.sigmoid(logit.view(-1).float()).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--outfile", default="temp/bernett_res_bidirect_swap_check.tsv")
    parser.add_argument("-d", "--device", type=int, default=-1)
    args = parser.parse_args()

    use_cuda = args.device >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.device)
    model = torch.load(
        args.model,
        weights_only=False,
        map_location="cuda" if use_cuda else torch.device("cpu"),
    )
    model.eval()
    model.use_cuda = use_cuda

    test_df = pd.read_csv(args.test, sep="\t", header=None).head(args.limit)
    proteins = sorted(set(test_df[0]).union(test_df[1]))
    embeddings = LazyEmbeddingStore(
        Path(args.embeddings),
        proteins,
        mode="pt_dir",
        cache_size=256,
        num_workers=4,
    )

    rows = []
    for i, (n0, n1, label) in enumerate(test_df.itertuples(index=False), 1):
        p01 = predict_pair(model, embeddings, n0, n1, use_cuda)
        p10 = predict_pair(model, embeddings, n1, n0, use_cuda)
        rows.append((n0, n1, label, p01, p10, p10 - p01, abs(p10 - p01)))
        if i % 10 == 0:
            print(f"done {i}/{len(test_df)}", flush=True)

    out = pd.DataFrame(
        rows,
        columns=["p1", "p2", "label", "pred_p1_p2", "pred_p2_p1", "delta", "abs_delta"],
    )
    out.to_csv(args.outfile, sep="\t", index=False)
    print(out["abs_delta"].describe().to_string())
    print("max_abs_delta_row:")
    print(out.loc[out["abs_delta"].idxmax()].to_string())


if __name__ == "__main__":
    main()
