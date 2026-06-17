"""
Evaluate a trained model.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from collections.abc import Callable
from typing import NamedTuple
import torch.nn.functional as F
import wandb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
)
import csv
from tqdm import tqdm
from pathlib import Path
from dscript.loading import LoadingPool

from dscript.models.interaction import InteractionInputs
from dscript.models.interaction import ModelInteraction
from dscript.models.contact import ContactCNN
from dscript.models.embedding import FullyConnectedEmbed

from ..foldseek import get_foldseek_onehot, build_backbone_vocab
from ..parallel_embedding_loader import EmbeddingLoader, add_batch_dim_if_needed
from ..fasta import parse_dict
from ..utils import log
import h5py

matplotlib.use("Agg")


class EvaluateArguments(NamedTuple):
    cmd: str
    device: int
    model: str
    embedding: str
    test: str
    func: Callable[[EvaluateArguments], None]


def add_args(parser):
    """
    Create parser for command line utility.

    :meta private:
    """

    parser.add_argument(
        "--model",
        default="samsl/topsy_turvy_human_v1",
        type=str,
        help="Pretrained Model. If this is a `.sav` or `.pt` file, it will be loaded. Otherwise, we will try to load `[model]` from HuggingFace hub [default: samsl/topsy_turvy_human_v1]",
    )
    parser.add_argument("--test", help="Test Data", required=True)
    parser.add_argument(
        "--embeddings",
        help="directory containing per-protein `.pt` embeddings or HDF5 file with embeddings",
        required=True,
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=1280,
        help="input embedding dimension used to initialize the original model when loading a state_dict checkpoint (default: 6165)",
    )
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=100,
        help="projection dimension used to initialize the original model when loading a state_dict checkpoint (default: 100)",
    )
    parser.add_argument(
        "--dropout-p",
        type=float,
        default=0.5,
        help="embedding dropout used to initialize the original model when loading a state_dict checkpoint (default: 0.5)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=50,
        help="contact hidden dimension used to initialize the original model when loading a state_dict checkpoint (default: 50)",
    )
    parser.add_argument(
        "--kernel-width",
        type=int,
        default=7,
        help="contact kernel width used to initialize the original model when loading a state_dict checkpoint (default: 7)",
    )
    parser.add_argument("-o", "--outfile", help="Output file to write results")
    parser.add_argument(
        "-d", "--device", type=int, default=-1, help="Compute device to use"
    )
    parser.add_argument(
        "--load_proc",
        type=int,
        default=16,
        help="Number of processes to use when loading embeddings (-1 = # of available CPUs, default=16). Because loading is IO-bound, values larger that the # of CPUs are allowed.",
    )

     # wandb
  
    parser.add_argument(
        "--log_wandb", action="store_true", help="Log metrics to Weights and Biases"
    )
    parser.add_argument(
        "--wandb-entity", default=None, help="Weights and Biases entity name"
    )
    parser.add_argument(
        "--wandb-project", default=None, help="Weights and Biases project name"
    )


    ## Foldseek arguments
    parser.add_argument(
        "--allow_foldseek",
        default=False,
        action="store_true",
        help="If set to true, adds the foldseek one-hot representation",
    )
    parser.add_argument(
        "--foldseek_fasta",
        help="foldseek fasta file containing the foldseek representation",
    )
    parser.add_argument(
        "--foldseek_vocab",
        help="foldseek vocab json file mapping foldseek alphabet to json",
    )

    parser.add_argument(
        "--add_foldseek_after_projection",
        default=False,
        action="store_true",
        help="If set to true, adds the fold seek embedding after the projection layer",
    )

    ## Backbone arguments
    parser.add_argument(
        "--allow_backbone3di",
        default=False,
        action="store_true",
        help="If set to true, adds the 12 state one-hot representation",
    )
    parser.add_argument(
        "--backbone3di_fasta",
        help="FASTA file containing the 12 state representation",
    )
    parser.add_argument(
        "--no-w",
        action="store_true",
        help="disable the weight matrix when instantiating the original model for state_dict checkpoints",
    )
    parser.add_argument(
        "--no-sigmoid",
        action="store_true",
        help="disable the final sigmoid activation when instantiating the original model for state_dict checkpoints",
    )
    parser.add_argument(
        "--do-pool",
        action="store_true",
        help="enable max-pool when instantiating the original model for state_dict checkpoints",
    )
    parser.add_argument(
        "--pool-width",
        type=int,
        default=9,
        help="pool width when instantiating the original model for state_dict checkpoints (default: 9)",
    )

    return parser


def _build_original_model(args, use_cuda: bool, fold_vocab=None, backbone_vocab=None):
    projection_dim = args.projection_dim
    embedding_model = FullyConnectedEmbed(
        args.input_dim,
        projection_dim,
        dropout=args.dropout_p,
    )

    contact_in_dim = projection_dim
    if args.allow_foldseek:
        contact_in_dim += len(fold_vocab or {})
    if args.allow_backbone3di:
        contact_in_dim += len(backbone_vocab or {})

    contact_model = ContactCNN(contact_in_dim, args.hidden_dim, args.kernel_width)
    model = ModelInteraction(
        embedding_model,
        contact_model,
        use_cuda,
        do_w=not args.no_w,
        pool_size=args.pool_width,
        do_pool=args.do_pool,
        do_sigmoid=not args.no_sigmoid,
    )
    model.use_cuda = use_cuda
    return model


def plot_eval_predictions(labels, predictions, path="figure"):
    """
    Plot histogram of positive and negative predictions, precision-recall curve, and receiver operating characteristic curve.

    :param y: Labels
    :type y: np.ndarray
    :param phat: Predicted probabilities
    :type phat: np.ndarray
    :param path: File prefix for plots to be saved to [default: figure]
    :type path: str
    """

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

    precision, recall, pr_thresh = precision_recall_curve(labels, predictions)
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

    fpr, tpr, roc_thresh = roc_curve(labels, predictions)
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
    phats: np.ndarray,
    out_path_prefix: str,
    threshold: float = 0.5,
    split_name: str = "test",
    inference_seconds: float | None = None,
    wandb_run=None,
) -> None:

    # labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    # phats = np.asarray(phats, dtype=np.float32).reshape(-1)

    n = int(labels.shape[0])
    inference_seconds_per_pair = (
        inference_seconds / n if inference_seconds is not None and n > 0 else None
    )

    # Loss (BCE over probabilities)
    if n == 0:
        loss = float("nan")
    else:
        logits = torch.from_numpy(phats).float()  # [n]
        y = torch.from_numpy(labels).float()
        loss = float(
            F.binary_cross_entropy_with_logits(logits, y, reduction="mean").item()
        )
        p_prob = torch.sigmoid(logits).cpu().numpy()

    # Other metrics
    if n == 0:
        aupr = auroc = acc = prec = rec = f1 = mse = float("nan")
    else:
        y_true_int = labels.astype(int)

        y_pred = (p_prob >= threshold).astype(int)

        aupr = float(average_precision_score(y_true_int, p_prob))
        auroc = (
            float(roc_auc_score(y_true_int, p_prob))
            if len(np.unique(y_true_int)) > 1
            else float("nan")
        )

        acc = float(accuracy_score(y_true_int, y_pred))
        prec = float(precision_score(y_true_int, y_pred, zero_division=0))
        rec = float(recall_score(y_true_int, y_pred, zero_division=0))
        f1 = float(f1_score(y_true_int, y_pred, zero_division=0))
        mse = float(mean_squared_error(y_true_int, p_prob))

    with open(out_path_prefix + "_metrics.txt", "w+") as f:
        inference_text = (
            f"\n[{split_name}] inference_seconds: {inference_seconds:.6f}"
            f"\n[{split_name}] inference_seconds_per_pair: {inference_seconds_per_pair:.6f}"
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
        }
        if inference_seconds is not None:
            payload["test/inference_seconds"] = inference_seconds
        if inference_seconds_per_pair is not None:
            payload["test/inference_seconds_per_pair"] = inference_seconds_per_pair
        wandb_run.log(payload)


def main(args):
    """
    Run model evaluation from arguments.

    :meta private:
    """
    wandb_run = None
    if args.log_wandb:
        wandb_run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity=args.wandb_entity,
            # Set the wandb project where this run will be logged.
            project=args.wandb_project,
            # Track hyperparameters and run metadata.
            config=vars(args),
        )
    ########## Foldseek code #########################
    allow_foldseek = args.allow_foldseek
    fold_fasta_file = args.foldseek_fasta
    fold_vocab_file = args.foldseek_vocab
    fold_record = {}
    fold_vocab = None
    if allow_foldseek:
        assert fold_fasta_file is not None and fold_vocab_file is not None
        fold_fasta = parse_dict(fold_fasta_file)
        for rec_k, rec_v in fold_fasta.items():
            fold_record[rec_k] = rec_v
        with open(fold_vocab_file) as fv:
            fold_vocab = json.load(fv)
    ########## Backbone code #########################
    allow_backbone = args.allow_backbone3di
    backbone_fasta_file = args.backbone3di_fasta
    backbone_record = {}
    backbone_vocab = None
    if allow_backbone:
        assert backbone_fasta is not None
        backbone_fasta = parse_dict(backbone_fasta_file)
        for rec_k, rec_v in backbone_fasta.items():
            backbone_record[rec_k] = rec_v
        backbone_vocab = build_backbone_vocab()

    ##################################################

    # Set Device
    device = args.device
    use_cuda = (device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        log(f"Using CUDA device {device} - {torch.cuda.get_device_name(device)}")
    else:
        log("Using CPU")

    # Load Model
    model_path = args.model
    map_location = "cuda" if use_cuda else torch.device("cpu")
    checkpoint_obj = torch.load(model_path, weights_only=False, map_location=map_location)

    if isinstance(checkpoint_obj, torch.nn.Module):
        model = checkpoint_obj
        if not use_cuda:
            model = model.cpu()
        model.use_cuda = use_cuda
        log(f"Loaded full model object from {model_path}")
    else:
        state_dict = (
            checkpoint_obj["state_dict"]
            if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj
            else checkpoint_obj
        )
        if not isinstance(state_dict, dict):
            raise TypeError(
                f"Unsupported checkpoint format in {model_path}: {type(checkpoint_obj)}"
            )

        model = _build_original_model(
            args,
            use_cuda,
            fold_vocab=fold_vocab,
            backbone_vocab=backbone_vocab,
        )
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            log(f"Missing keys when loading state_dict: {missing_keys}")
        if unexpected_keys:
            log(f"Unexpected keys when loading state_dict: {unexpected_keys}")
        log(f"Loaded state_dict checkpoint from {model_path}")

    if use_cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    emb_path = Path(args.embeddings)

    if emb_path.is_dir():
        embedding_mode = "pt_dir"
        log(f"Embedding path is a directory: {emb_path}")
    elif emb_path.is_file():
        # Could be HDF5 or something else
        if h5py.is_hdf5(str(emb_path)):
            embedding_mode = "hdf5"
            log(f"Embedding path is an HDF5 file: {emb_path}")
        else:
            raise ValueError(
                f"Embedding file is not HDF5 and not a directory: {emb_path}"
            )
    else:
        raise FileNotFoundError(f"Embedding path does not exist: {emb_path}")

    # Load Pairs
    test_fi = args.test

    test_df = pd.read_csv(test_fi, sep="\t", header=None)

    if args.outfile is None:
        outPath = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    else:
        outPath = args.outfile
    outFile = open(outPath + ".predictions.tsv", "w+")

    allProteins = sorted(list(set(test_df[0]).union(test_df[1])))

    # Load embeddings
    embeddings: dict[str, torch.Tensor] = {}
    if embedding_mode == "pt_dir":
        embedding_loader = EmbeddingLoader(
            embedding_dir_name=emb_path, protein_names=allProteins, num_workers=4
        )
        embeddings = embedding_loader.embeddings_cpu
    elif embedding_mode == "hdf5":
        with h5py.File(emb_path, "r") as h5fi:
            for prot_name in tqdm(allProteins, desc="Loading HDF5 embeddings"):
                embeddings[prot_name] = torch.from_numpy(h5fi[prot_name][:, :])

    # Evaluate

    model.eval()
    inference_start = time.perf_counter()
    with torch.no_grad():
        logits = []
        labels = []
        probs = []
        for _, (n0, n1, label) in tqdm(
            test_df.iterrows(), total=len(test_df), desc="Predicting pairs"
        ):
            try:

                p0 = embeddings[n0]
                p1 = embeddings[n1]

                # Ensure 3D [B, L, D]
                p0 = add_batch_dim_if_needed(p0)
                p1 = add_batch_dim_if_needed(p1)

                if use_cuda:
                    p0 = p0.cuda()
                    p1 = p1.cuda()

                f_a = f_b = b_a = b_b = None

                def build_struct_embedding(n, length, record, vocab):
                    e = get_foldseek_onehot(n, length, record, vocab).unsqueeze(0)
                    if use_cuda:
                        e = e.cuda()
                    return e

                if allow_foldseek:
                    f_a = build_struct_embedding(
                        n0, p0.shape[1], fold_record, fold_vocab
                    )
                    f_b = build_struct_embedding(
                        n1, p1.shape[1], fold_record, fold_vocab
                    )

                if allow_backbone:
                    b_a = build_struct_embedding(
                        n0, p0.shape[1], backbone_record, backbone_vocab
                    )
                    b_b = build_struct_embedding(
                        n1, p1.shape[1], backbone_record, backbone_vocab
                    )

                interactionInputs = InteractionInputs(
                    p0,
                    p1,
                    embed_foldseek=allow_foldseek,
                    f0=f_a,
                    f1=f_b,
                    embed_backbone=allow_backbone,
                    b0=b_a,
                    b1=b_b,
                )
                _, logit = model.map_predict(interactionInputs)

                logit_val = logit.view(-1).float().item()
                prob_val = torch.sigmoid(logit.view(-1).float()).item()

                logits.append(logit_val)
                probs.append(prob_val)
                labels.append(label)

                outFile.write(f"{n0}\t{n1}\t{label}\t{prob_val:.5}\n")
            except Exception as e:
                sys.stderr.write(f"{n0} x {n1} - {e}")
    inference_seconds = time.perf_counter() - inference_start
    log(f"Inference time: {inference_seconds:.6f}s")
    if len(labels) > 0:
        log(f"Inference time per pair: {inference_seconds / len(labels):.6f}s")

    logits = np.array(logits, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    probs = np.array(probs, dtype=np.float32)

    log_eval_metrics(
        labels=labels,
        phats=logits,
        out_path_prefix=outPath,
        threshold=0.5,
        split_name="test",
        inference_seconds=inference_seconds,
        wandb_run=wandb_run,
    )
    

    plot_eval_predictions(labels, probs, outPath)

    outFile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
