"""
Evaluate a trained interaction_clear model.
"""

from __future__ import annotations

import argparse
import datetime
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from ..fasta import parse_dict
from ..foldseek import build_backbone_vocab, fold_vocab, get_foldseek_onehot
from ..models.interaction_clear import InteractionInputs
from ..parallel_embedding_loader import EmbeddingLoader, add_batch_dim_if_needed
from ..utils import log

matplotlib.use("Agg")


class EvaluateArguments(NamedTuple):
    cmd: str
    device: int
    model: str
    embeddings: str
    test: str
    func: Callable[[EvaluateArguments], None]


def add_args(parser):
    parser.add_argument("--model", required=True, type=str, help="Path to trained model")
    parser.add_argument("--test", required=True, help="Test data TSV")
    parser.add_argument(
        "--embeddings",
        required=True,
        help="Directory containing per-protein `.pt` embeddings or an HDF5 file",
    )
    parser.add_argument("-o", "--outfile", help="Output prefix for predictions and plots")
    parser.add_argument("-d", "--device", type=int, default=-1, help="Compute device to use")
    parser.add_argument(
        "--load_proc",
        type=int,
        default=4,
        help="Number of workers to use when loading directory embeddings",
    )
    parser.add_argument("--log_wandb", action="store_true", help="Log metrics to Weights and Biases")
    parser.add_argument("--wandb-entity", default=None, help="Weights and Biases entity name")
    parser.add_argument("--wandb-project", default=None, help="Weights and Biases project name")
    parser.add_argument(
        "--allow_foldseek",
        default=False,
        action="store_true",
        help="If set, add the foldseek one-hot representation",
    )
    parser.add_argument("--foldseek_fasta", help="FASTA file containing the foldseek representation")
    parser.add_argument(
        "--allow_backbone3di",
        default=False,
        action="store_true",
        help="If set, add the backbone 3Di one-hot representation",
    )
    parser.add_argument("--backbone3di_fasta", help="FASTA file containing the backbone 3Di representation")
    return parser


def plot_eval_predictions(labels, predictions, path="figure"):
    pos_phat = predictions[labels == 1]
    neg_phat = predictions[labels == 0]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Distribution of Predictions")
    ax1.hist(pos_phat)
    ax1.set_xlim(0, 1)
    ax1.set_title("Positive")
    ax1.set_xlabel("p-hat")
    ax2.hist(neg_phat)
    ax2.set_xlim(0, 1)
    ax2.set_title("Negative")
    ax2.set_xlabel("p-hat")
    plt.savefig(path + ".phat_dist.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(labels, predictions)
    aupr = average_precision_score(labels, predictions)
    log(f"AUPR: {aupr}")

    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f"Precision-Recall (AUPR: {aupr:.3})")
    plt.savefig(path + ".aupr.png")
    plt.close()

    fpr, tpr, _ = roc_curve(labels, predictions)
    auroc = roc_auc_score(labels, predictions)
    log(f"AUROC: {auroc}")

    plt.step(fpr, tpr, color="b", alpha=0.2, where="post")
    plt.fill_between(fpr, tpr, step="post", alpha=0.2, color="b")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f"Receiver Operating Characteristic (AUROC: {auroc:.3})")
    plt.savefig(path + ".auroc.png")
    plt.close()


def log_eval_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    out_path_prefix: str,
    threshold: float = 0.5,
    split_name: str = "test",
    inference_seconds: float | None = None,
    wandb_run=None,
) -> None:
    n = int(labels.shape[0])

    if n == 0:
        loss = float("nan")
        p_prob = np.array([], dtype=np.float32)
    else:
        logits_t = torch.from_numpy(logits).float()
        y_t = torch.from_numpy(labels).float()
        loss = float(F.binary_cross_entropy_with_logits(logits_t, y_t, reduction="mean").item())
        p_prob = torch.sigmoid(logits_t).cpu().numpy()

    if n == 0:
        aupr = auroc = acc = prec = rec = f1 = mse = float("nan")
    else:
        y_true_int = labels.astype(int)
        y_pred = (p_prob >= threshold).astype(int)
        aupr = float(average_precision_score(y_true_int, p_prob))
        auroc = float(roc_auc_score(y_true_int, p_prob)) if len(np.unique(y_true_int)) > 1 else float("nan")
        acc = float(accuracy_score(y_true_int, y_pred))
        prec = float(precision_score(y_true_int, y_pred, zero_division=0))
        rec = float(recall_score(y_true_int, y_pred, zero_division=0))
        f1 = float(f1_score(y_true_int, y_pred, zero_division=0))
        mse = float(mean_squared_error(y_true_int, p_prob))

    with open(out_path_prefix + "_metrics.txt", "w+") as f:
        inference_text = (
            f"\n[{split_name}] inference_seconds: {inference_seconds:.6f}"
            if inference_seconds is not None
            else ""
        )
        log(
            f"[{split_name}] n: {n}\n"
            f"[{split_name}] threshold: {threshold}\n"
            f"[{split_name}] loss: {loss:.6f}\n"
            f"[{split_name}] AUPR: {aupr:.6f}\n"
            f"[{split_name}] AUROC: {auroc:.6f}\n"
            f"[{split_name}] accuracy: {acc:.6f}\n"
            f"[{split_name}] mse: {mse:.6f}\n"
            f"[{split_name}] precision: {prec:.6f}\n"
            f"[{split_name}] recall: {rec:.6f}\n"
            f"[{split_name}] f1: {f1:.6f}"
            f"{inference_text}",
            file=f,
        )

    if wandb_run is not None:
        payload = {
            "test/loss": loss,
            "test/accuracy": acc,
            "test/mse": mse,
            "test/precision": prec,
            "test/recall": rec,
            "test/f1": f1,
            "test/aupr": aupr,
            "test/auroc": auroc,
        }
        if inference_seconds is not None:
            payload["test/inference_seconds"] = inference_seconds
        wandb_run.log(payload)


def load_records(enabled=False, fasta_path=""):
    if not enabled:
        return {}
    assert fasta_path is not None
    return parse_dict(fasta_path)


def main(args):
    wandb_run = None
    if args.log_wandb:
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            config=vars(args),
        )

    allow_foldseek = args.allow_foldseek
    allow_backbone3di = args.allow_backbone3di
    fold_record = load_records(allow_foldseek, args.foldseek_fasta)
    backbone_record = load_records(allow_backbone3di, args.backbone3di_fasta)
    backbone_vocab = build_backbone_vocab()

    device = args.device
    use_cuda = (device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        log(f"Using CUDA device {device} - {torch.cuda.get_device_name(device)}")
    else:
        log("Using CPU")

    model_path = args.model
    if use_cuda:
        model = torch.load(model_path, weights_only=False, map_location="cuda")
        model.use_cuda = True
    else:
        model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False).cpu()
        model.use_cuda = False

    emb_path = Path(args.embeddings)
    if emb_path.is_dir():
        embedding_mode = "pt_dir"
        log(f"Embedding path is a directory: {emb_path}")
    elif emb_path.is_file() and h5py.is_hdf5(str(emb_path)):
        embedding_mode = "hdf5"
        log(f"Embedding path is an HDF5 file: {emb_path}")
    else:
        raise FileNotFoundError(f"Embedding path does not exist or is unsupported: {emb_path}")

    test_df = pd.read_csv(args.test, sep="\t", header=None)
    out_path = args.outfile or datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    out_file = open(out_path + ".predictions.tsv", "w+")

    all_proteins = sorted(list(set(test_df[0]).union(test_df[1])))
    embeddings: dict[str, torch.Tensor] = {}
    if embedding_mode == "pt_dir":
        embedding_loader = EmbeddingLoader(
            embedding_dir_name=emb_path,
            protein_names=all_proteins,
            num_workers=args.load_proc,
        )
        embeddings = embedding_loader.embeddings_cpu
    else:
        with h5py.File(emb_path, "r") as h5fi:
            for prot_name in tqdm(all_proteins, desc="Loading HDF5 embeddings"):
                embeddings[prot_name] = torch.from_numpy(h5fi[prot_name][:, :])

    model.eval()
    logits = []
    labels = []
    probs = []

    inference_start = time.perf_counter()
    with torch.no_grad():
        for _, (n0, n1, label) in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting pairs"):
            p0 = add_batch_dim_if_needed(embeddings[n0])
            p1 = add_batch_dim_if_needed(embeddings[n1])

            if use_cuda:
                p0 = p0.cuda()
                p1 = p1.cuda()

            f_a = f_b = b_a = b_b = None

            def build_struct_embedding(name, length, record, vocab):
                enc = get_foldseek_onehot(name, length, record, vocab).unsqueeze(0)
                return enc.cuda() if use_cuda else enc

            if allow_foldseek:
                f_a = build_struct_embedding(n0, p0.shape[1], fold_record, fold_vocab)
                f_b = build_struct_embedding(n1, p1.shape[1], fold_record, fold_vocab)

            if allow_backbone3di:
                b_a = build_struct_embedding(n0, p0.shape[1], backbone_record, backbone_vocab)
                b_b = build_struct_embedding(n1, p1.shape[1], backbone_record, backbone_vocab)

            interaction_inputs = InteractionInputs(
                p0,
                p1,
                embed_foldseek=allow_foldseek,
                f0=f_a,
                f1=f_b,
                embed_backbone=allow_backbone3di,
                b0=b_a,
                b1=b_b,
            )
            _, logit, _ = model.map_predict(interaction_inputs)

            logit_val = logit.view(-1).float().item()
            prob_val = torch.sigmoid(logit.view(-1).float()).item()
            logits.append(logit_val)
            probs.append(prob_val)
            labels.append(label)
            out_file.write(f"{n0}\t{n1}\t{label}\t{prob_val:.5f}\n")
    inference_seconds = time.perf_counter() - inference_start
    log(f"Inference time: {inference_seconds:.6f}s")

    logits = np.array(logits, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    probs = np.array(probs, dtype=np.float32)

    log_eval_metrics(
        labels=labels,
        logits=logits,
        out_path_prefix=out_path,
        threshold=0.5,
        split_name="test",
        inference_seconds=inference_seconds,
        wandb_run=wandb_run,
    )
    plot_eval_predictions(labels, probs, out_path)
    out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
