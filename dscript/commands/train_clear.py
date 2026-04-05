"""
Train an interaction_clear model with classifier-side fused-map processing.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_optimizer as optim
import wandb
from sklearn.metrics import average_precision_score as average_precision
from torch.autograd import Variable
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from .. import __version__
from ..fasta import parse_dict
from ..foldseek import (
    Foldseek3diContext,
    build_backbone_vocab,
    fold_vocab,
    get_foldseek_onehot,
)
from ..glider import glide_compute_map, glider_score
from ..models.contact import ContactCNN
from ..models.embedding import FullyConnectedEmbed
from ..models.interaction_clear import InteractionInputs, ModelInteraction
from ..parallel_embedding_loader import EmbeddingLoader, add_batch_dim_if_needed
from ..utils import (
    PairedDataset,
    collate_paired_sequences,
    log,
)


class TrainArguments(NamedTuple):
    cmd: str
    device: int
    train: str
    test: str
    embedding: str
    no_augment: bool
    input_dim: int
    projection_dim: int
    dropout: float
    hidden_dim: int
    kernel_width: int
    grid_size: int
    classifier_d_model: int
    no_w: bool
    no_sigmoid: bool
    do_pool: bool
    pool_width: int
    num_epochs: int
    batch_size: int
    weight_decay: float
    lr: float
    run_tt: bool
    glider_weight: float
    glider_thresh: float
    outfile: str | None
    save_prefix: str | None
    checkpoint: str | None
    seed: int | None
    use_lookahead: bool
    func: Callable[[TrainArguments], None]


def _grad_l2_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def _module_grad_norm(named_params, pattern: str) -> float:
    total = 0.0
    for name, param in named_params:
        if pattern in name and param.grad is not None:
            total += param.grad.data.norm(2).item() ** 2
    return total ** 0.5


def add_args(parser):
    """
    Create parser for command line utility.

    :meta private:
    """

    data_grp = parser.add_argument_group("Data")
    proj_grp = parser.add_argument_group("Projection Module")
    contact_grp = parser.add_argument_group("Contact Module")
    inter_grp = parser.add_argument_group("Interaction Module")
    train_grp = parser.add_argument_group("Training")
    misc_grp = parser.add_argument_group("Output and Device")
    foldseek_grp = parser.add_argument_group("Foldseek related commands")

    # Data
    data_grp.add_argument("--train", required=True, help="list of training pairs")
    data_grp.add_argument(
        "--test", required=True, help="list of validation/testing pairs"
    )
    # Embedding Directory
    data_grp.add_argument(
        "--embedding",
        required=True,
        help="directory containing per-protein `.pt` embeddings or HDF5 file with embeddings",
    )
    data_grp.add_argument(
        "--no-augment",
        action="store_true",
        help="data is automatically augmented by adding (B A) for all pairs (A B). Set this flag to not augment data",
    )

    # Embedding model
    proj_grp.add_argument(
        "--input-dim",
        type=int,
        default=1280,
        help="dimension of input language model embedding (per amino acid) (default: 1280), ESM-2 650M: 1280;ESM-C 600M: 1152",
    )
    proj_grp.add_argument(
        "--projection-dim",
        type=int,
        default=100,
        help="dimension of embedding projection layer (default: 100)",
    )
    proj_grp.add_argument(
        "--dropout-p",
        type=float,
        default=0.5,
        help="parameter p for embedding dropout layer (default: 0.5)",
    )

    # Contact model
    contact_grp.add_argument(
        "--hidden-dim",
        type=int,
        default=50,
        help="number of hidden units for comparison layer in contact prediction (default: 50)",
    )
    contact_grp.add_argument(
        "--kernel-width",
        type=int,
        default=7,
        help="width of convolutional filter for contact prediction (default: 7)",
    )
    inter_grp.add_argument(
        "--grid-size",
        type=int,
        default=100,
        help="pooled grid size for the clean map classifier (default: 100)",
    )
    inter_grp.add_argument(
        "--classifier-d-model",
        type=int,
        default=64,
        help="hidden width for PairClassifierTokens (default: 64)",
    )

    # Interaction Model
    inter_grp.add_argument(
        "--no-w",
        action="store_true",
        help="no use of weight matrix in interaction prediction model",
    )
    inter_grp.add_argument(
        "--no-sigmoid",
        action="store_true",
        help="no use of sigmoid activation at end of interaction model",
    )
    inter_grp.add_argument(
        "--do-pool",
        action="store_true",
        help="use max pool layer in interaction prediction model",
    )
    inter_grp.add_argument(
        "--pool-width",
        type=int,
        default=9,
        help="size of max-pool in interaction model (default: 9)",
    )

    # Training
    train_grp.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="number of epochs (default: 10)",
    )

    train_grp.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="minibatch size (default: 25)",
    )
    train_grp.add_argument(
        "--weight-decay",
        type=float,
        default=0,
        help="L2 regularization (default: 0)",
    )
    train_grp.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate (default: 0.001)",
    )
    train_grp.add_argument(
        "--use-lookahead",
        action="store_true",
        help="wrap the base AdamW optimizer with Lookahead",
    )
    # Topsy-Turvy
    train_grp.add_argument(
        "--topsy-turvy",
        dest="run_tt",
        action="store_true",
        help="run in Topsy-Turvy mode -- use top-down GLIDER scoring to guide training",
    )
    train_grp.add_argument(
        "--glider-weight",
        dest="glider_weight",
        type=float,
        default=0.2,
        help="weight on the GLIDER accuracy objective (default: 0.2)",
    )
    train_grp.add_argument(
        "--glider-thresh",
        dest="glider_thresh",
        type=float,
        default=0.925,
        help="threshold beyond which GLIDER scores treated as positive edges (0 < gt < 1) (default: 0.925)",
    )

    # Output
    misc_grp.add_argument("-o", "--outfile", help="output file path (default: stdout)")
    misc_grp.add_argument("--save-prefix", help="path prefix for saving models")
    misc_grp.add_argument(
        "-d", "--device", type=int, default=-1, help="compute device to use"
    )
    misc_grp.add_argument(
        "--checkpoint", help="checkpoint model to start training from"
    )
    misc_grp.add_argument("--seed", help="Set random seed", type=int)
    misc_grp.add_argument(
        "--log_wandb", action="store_true", help="Log metrics to Weights and Biases"
    )
    misc_grp.add_argument(
        "--wandb-entity", default=None, help="Weights and Biases entity name"
    )
    misc_grp.add_argument(
        "--wandb-project", default=None, help="Weights and Biases project name"
    )

    ## Foldseek arguments
    foldseek_grp.add_argument(
        "--allow_foldseek",
        default=False,
        action="store_true",
        help="If set to true, adds the foldseek one-hot representation",
    )
    foldseek_grp.add_argument(
        "--foldseek_fasta",
        help="foldseek fasta file containing the foldseek representation",
    )
    foldseek_grp.add_argument(
        "--allow_backbone3di",
        default=False,
        action="store_true",
        help="If set to true, adds the 12 state one-hot representation",
    )
    foldseek_grp.add_argument(
        "--backbone3di_fasta",
        help="FASTA file containing the 12 state representation",
    )

    return parser


def predict_cmap_interaction(
    model,
    n0,
    n1,
    tensors,
    use_cuda,
    structural_context=None,
    require_grad=False,
):
    b = len(n0)

    # mode + grad context BEFORE running the model
    if require_grad:
        model.train()
        ctx = torch.enable_grad()
    else:
        model.eval()
        ctx = torch.no_grad()

    def _maybe_cuda(x):
        return x.cuda() if (use_cuda and x is not None) else x

    def _predict_one(i):
        z_a = add_batch_dim_if_needed(tensors[n0[i]])
        z_b = add_batch_dim_if_needed(tensors[n1[i]])
        z_a, z_b = _maybe_cuda(z_a), _maybe_cuda(z_b)

        f_a = f_b = b_a = b_b = None

        if structural_context is not None and structural_context.allow_foldseek:
            f_a = get_foldseek_onehot(
                n0[i], z_a.shape[1],
                structural_context.fold_record,
                structural_context.fold_vocab,
            ).unsqueeze(0)
            f_b = get_foldseek_onehot(
                n1[i], z_b.shape[1],
                structural_context.fold_record,
                structural_context.fold_vocab,
            ).unsqueeze(0)
            f_a, f_b = _maybe_cuda(f_a), _maybe_cuda(f_b)

        if structural_context is not None and structural_context.allow_backbone3di:
            b_a = get_foldseek_onehot(
                n0[i], z_a.shape[1],
                structural_context.backbone_record,
                structural_context.backbone_vocab,
            ).unsqueeze(0)
            b_b = get_foldseek_onehot(
                n1[i], z_b.shape[1],
                structural_context.backbone_record,
                structural_context.backbone_vocab,
            ).unsqueeze(0)
            b_a, b_b = _maybe_cuda(b_a), _maybe_cuda(b_b)

        _, logit, feat = model.map_predict(
            InteractionInputs(
                z_a, z_b,
                embed_foldseek=(structural_context is not None and structural_context.allow_foldseek),
                f0=f_a, f1=f_b,
                embed_backbone=(structural_context is not None and structural_context.allow_backbone3di),
                b0=b_a, b1=b_b,
            )
        )
        return logit.reshape(-1), feat

    with ctx:
        pair_outputs = tuple(_predict_one(i) for i in range(b))
        logits, feat = (
            torch.cat(parts, dim=0) for parts in zip(*pair_outputs)
        )
        aux = {}

    return logits, feat, aux


def predict_interaction(model, n0, n1, tensors, use_cuda, structural_context=None):
    logits, _, _ = predict_cmap_interaction(model, n0, n1, tensors, use_cuda, structural_context)
    return logits


def smooth_labels(labels, smoothing=0.1):
    return labels * (1 - smoothing) + 0.5 * smoothing


def mixup_feat(feat: torch.Tensor, y: torch.Tensor, alpha: float = 0.4,
               lam_min: float | None = None, lam_max: float | None = None):
    """
    feat: [B, ...]
    y:    [B] (0/1) or [B,1]
    returns feat_mix, y_mix
    """
    if feat.size(0) < 2:
        return feat, y

    B = feat.size(0)
    device = feat.device

    y = y.to(device=device, dtype=torch.float32).view(B)

    # sample lambda
    if (lam_min is not None) and (lam_max is not None):
        lam = torch.empty(B, device=device, dtype=torch.float32).uniform_(lam_min, lam_max)
    else:
        lam = torch.distributions.Beta(alpha, alpha).sample((B,)).to(device=device, dtype=torch.float32)

    index = torch.randperm(B, device=device)

    # broadcast lambda over feature dims
    lam_feat = lam.view(B, *([1] * (feat.dim() - 1))).to(dtype=feat.dtype)

    feat_mix = lam_feat * feat + (1.0 - lam_feat) * feat[index]
    y_mix = lam * y + (1.0 - lam) * y[index]   # keep y_mix float32

    return feat_mix, y_mix




def interaction_grad(
    model,
    n0,
    n1,
    y,
    tensors,
    run_tt=False,
    glider_weight=0,
    glider_map=None,
    glider_mat=None,
    use_cuda=True,
    structural_context=None,
):
    logits, _, _ = predict_cmap_interaction(
        model,
        n0,
        n1,
        tensors,
        use_cuda,
        structural_context,
        require_grad=True,
    )

    b = len(n0)
    y = (y.cuda() if use_cuda else y).float().view(-1)

    logits_used = logits.view(-1).float()
    bce_loss = F.binary_cross_entropy_with_logits(logits_used, y)
    if run_tt:
        g_score = torch.tensor(
            [glider_score(n0[i], n1[i], glider_map, glider_mat) for i in range(b)],
            dtype=torch.float32,
            device=logits_used.device,
        )
        glider_loss = F.binary_cross_entropy_with_logits(logits_used, g_score)
        bce_loss = glider_weight * glider_loss + (1.0 - glider_weight) * bce_loss

    loss = bce_loss

    # -----------------------------
    # Metrics (use NON-mixup logits for interpretability)
    # -----------------------------
    with torch.no_grad():
        p_prob = torch.sigmoid(logits_used.view(-1))  # [B]
        y_metrics = y.to(logits_used.device)
        correct = ((p_prob > 0.5).float() == y_metrics).sum().item()
        mse = ((y_metrics - p_prob) ** 2).mean().item()
        assert torch.isfinite(logits_used).all()

    loss_terms = {
        "bce_loss": float(bce_loss.detach().item()),
    }

    return loss, correct, mse, b, logits_used, loss_terms




def interaction_eval(
    model,
    test_iterator,
    tensors,
    use_cuda,
    ### Foldseek added here
    structural_context=None,
    ###
):
    """
    Evaluate test data set performance.

    :param model: Model to be trained
    :type model: dscript.models.interaction.ModelInteraction
    :param test_iterator: Test data iterator
    :type test_iterator: torch.utils.data.DataLoader
    :param tensors: Dictionary of protein names to embeddings
    :type tensors: dict[str, torch.Tensor]
    :param use_cuda: Whether to use GPU
    :type use_cuda: bool

    :return: (Loss, number correct, mean square error, precision, recall, F1 Score, AUPR)
    :rtype: (torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
    """
    p_hat_list = []
    y_list = []

    for n0, n1, y in test_iterator:
        # predict_interaction should return logits shaped [B] or [B,1]
        logits= predict_interaction(model, n0, n1, tensors, use_cuda, structural_context)
        p_hat_list.append(logits.view(-1))   # force [B]
        y_list.append(y.view(-1))            # force [B]

    logits = torch.cat(p_hat_list, dim=0)    # [N]
    y = torch.cat(y_list, dim=0).float()     # [N]

    device = logits.device
    y = y.to(device)

    # --- Loss: logits + targets
    
    loss = F.binary_cross_entropy_with_logits(logits, y)

    with torch.no_grad():
        p = torch.sigmoid(logits)            # [N] probabilities

        pred = (p > 0.5).float()
        correct = (pred == y).sum().item()

        # MSE should be on probabilities vs labels (not logits)
        mse = torch.mean((y - p) ** 2).item()

        tp = torch.sum(pred * y).item()
        fp = torch.sum(pred * (1 - y)).item()
        fn = torch.sum((1 - pred) * y).item()
        

        pr = tp / (tp + fp + 1e-8)
        re = tp / (tp + fn + 1e-8)
        f1 = 2 * pr * re / (pr + re + 1e-8)
        

    # --- AUPR: use probs (recommended)
    y_np = y.detach().cpu().numpy()
    p_np = p.detach().cpu().numpy()
    aupr = average_precision(y_np, p_np)

    return loss, correct, mse, pr, re, f1, aupr

def split_params_by_prefix(model, prefix: str):
    hi, lo = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (hi if n.startswith(prefix) else lo).append(p)
    return hi, lo

class LRRestarter:
    def __init__(
        self, optimizer, initial_T_max, restart_T_max, patience, start_restart_epoch
    ):
        self.optimizer = optimizer
        self.initial_T_max = initial_T_max
        self.restart_T_max = restart_T_max
        self.patience = patience
        self.start_restart_epoch = start_restart_epoch

        self.scheduler = CosineAnnealingLR(
            optimizer, T_max=self.initial_T_max, eta_min=5e-5
        )
        self.best_score = None
        self.epochs_since_improvement = 0
        self.last_restart_epoch = 0

    def step(self):
        self.scheduler.step()

    def update(self, val_score, epoch):
        if self.best_score is None or val_score > self.best_score + 0.001:
            self.best_score = val_score
            self.epochs_since_improvement = 0
            return False  # no restart
        else:
            self.epochs_since_improvement += 1

            if (
                epoch >= self.start_restart_epoch
                and self.epochs_since_improvement >= self.patience
            ):
                print(f"🔁 Restarting LR at epoch {epoch+1} due to no improvement")
                self.scheduler = CosineAnnealingLR(
                    self.optimizer, T_max=self.restart_T_max, eta_min=5e-5
                )
                self.epochs_since_improvement = 0
                self.last_restart_epoch = epoch
                return True  # did restart

        return False  


def _count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def _sync_cuda(use_cuda):
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


def _format_duration(seconds):
    total_seconds = max(float(seconds), 0.0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


def train_model(args, output):
    overall_start_time = time.perf_counter()
    total_inference_time = 0.0

    if args.log_wandb:
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity=args.wandb_entity,
            # Set the wandb project where this run will be logged.
            project=args.wandb_project,
            # Track hyperparameters and run metadata.
            config=vars(args),
        )

    # Create data sets
    batch_size = args.batch_size
    use_cuda = (args.device > -1) and torch.cuda.is_available()
    train_fi = args.train
    test_fi = args.test
    no_augment = args.no_augment

    emb_path = Path(args.embedding)

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

    ########## Foldseek code #########################

    def load_records(enabled=False, fasta_path=""):
        if not enabled:
            return {}
        assert fasta_path is not None
        return parse_dict(fasta_path)

    allow_foldseek = args.allow_foldseek
    allow_backbone3di = args.allow_backbone3di

    fold_record = load_records(allow_foldseek, args.foldseek_fasta)
    backbone_record = load_records(allow_backbone3di, args.backbone3di_fasta)

    backbone_vocab = build_backbone_vocab()

    foldseek3dicontext = Foldseek3diContext(
        # foldseek info
        allow_foldseek=allow_foldseek,
        fold_record=fold_record,
        fold_vocab=fold_vocab,
        # backbone info
        allow_backbone3di=allow_backbone3di,
        backbone_record=backbone_record,
        backbone_vocab=backbone_vocab,
    )

    ##################################################

    train_df = pd.read_csv(train_fi, sep="\t", header=None)
    train_df.columns = ["prot1", "prot2", "label"]

    if no_augment:
        train_p1 = train_df["prot1"]
        train_p2 = train_df["prot2"]
        train_y = torch.from_numpy(train_df["label"].values)
    else:
        train_p1 = pd.concat(
            (train_df["prot1"], train_df["prot2"]), axis=0
        ).reset_index(drop=True)
        train_p2 = pd.concat(
            (train_df["prot2"], train_df["prot1"]), axis=0
        ).reset_index(drop=True)
        train_y = torch.from_numpy(
            pd.concat((train_df["label"], train_df["label"])).values
        )

    train_dataset = PairedDataset(train_p1, train_p2, train_y)
    train_iterator = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_paired_sequences,
        shuffle=True,
    )

    log(f"Loaded {len(train_p1)} training pairs", file=output)
    output.flush()

    test_df = pd.read_csv(test_fi, sep="\t", header=None)
    test_df.columns = ["prot1", "prot2", "label"]
    test_p1 = test_df["prot1"]
    test_p2 = test_df["prot2"]
    test_y = torch.from_numpy(test_df["label"].values)

    test_dataset = PairedDataset(test_p1, test_p2, test_y)
    test_iterator = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_paired_sequences,
        shuffle=False,
    )

    log(f"Loaded {len(test_p1)} test pairs", file=output)
    log("Loading embeddings...", file=output)
    output.flush()
    test_size = len(test_dataset)

    all_proteins = set(train_p1).union(train_p2).union(test_p1).union(test_p2)

    # Load embeddings
    embeddings: dict[str, torch.Tensor] = {}
    if embedding_mode == "pt_dir":
        embedding_loader = EmbeddingLoader(
            embedding_dir_name=emb_path, protein_names=all_proteins, num_workers=4
        )
        embeddings = embedding_loader.embeddings_cpu
    elif embedding_mode == "hdf5":
        with h5py.File(emb_path, "r") as h5fi:
            for prot_name in tqdm(all_proteins, desc="Loading HDF5 embeddings"):
                embeddings[prot_name] = torch.from_numpy(h5fi[prot_name][:, :])

    # Topsy-Turvy
    run_tt = args.run_tt
    glider_weight = args.glider_weight
    glider_thresh = args.glider_thresh * 100

    if run_tt:
        log("Running D-SCRIPT Topsy-Turvy:", file=output)
        log(f"\tglider_weight: {glider_weight}", file=output)
        log(f"\tglider_thresh: {glider_thresh}th percentile", file=output)
        log("Computing GLIDER matrix...", file=output)
        output.flush()

        glider_mat, glider_map = glide_compute_map(
            train_df[train_df.iloc[:, 2] == 1], thres_p=glider_thresh
        )
    else:
        glider_mat, glider_map = (None, None)

    # Create embedding model
    input_dim = args.input_dim

    projection_dim = args.projection_dim

    dropout_p = args.dropout_p
    embedding_model = FullyConnectedEmbed(input_dim, projection_dim, dropout=dropout_p)
    log("Initializing embedding model with:", file=output)
    log(f"\tprojection_dim: {projection_dim}", file=output)
    log(f"\tdropout_p: {dropout_p}", file=output)

    # Create contact model
    hidden_dim = args.hidden_dim
    kernel_width = args.kernel_width
    log("Initializing contact model with:", file=output)
    log(f"\thidden_dim: {hidden_dim}", file=output)
    log(f"\tkernel_width: {kernel_width}", file=output)

    proj_dim = projection_dim
    if allow_foldseek:
        proj_dim += len(fold_vocab)
    if allow_backbone3di:
        proj_dim += len(backbone_vocab)
    contact_model = ContactCNN(proj_dim, hidden_dim, kernel_width)

    # Create the full model
    do_w = not args.no_w
    do_pool = args.do_pool
    pool_width = args.pool_width
    grid_size = args.grid_size
    classifier_d_model = args.classifier_d_model
    do_sigmoid = not args.no_sigmoid
    log("Initializing interaction model with:", file=output)
    log(f"\tdo_poool: {do_pool}", file=output)
    log(f"\tpool_width: {pool_width}", file=output)
    log(f"\tgrid_size: {grid_size}", file=output)
    log(f"\tclassifier_d_model: {classifier_d_model}", file=output)
    log(f"\tdo_w: {do_w}", file=output)
    log(f"\tdo_sigmoid: {do_sigmoid}", file=output)
    model = ModelInteraction(
        embedding_model,
        contact_model,
        use_cuda,
        dropout_p,
        grid_size=grid_size,
        classifier_d_model=classifier_d_model,
        do_w=do_w,
        pool_size=pool_width,
        do_pool=do_pool,
        do_sigmoid=do_sigmoid,
    )
    model.use_cuda = use_cuda

    log(model, file=output)
    total_params, trainable_params = _count_parameters(model)
    log(
        f"Model parameters: total={total_params:,}, trainable={trainable_params:,}, frozen={total_params - trainable_params:,}",
        file=output,
    )

    if args.checkpoint is not None:
        log(
            f"Loading model from checkpoint {args.checkpoint}",
            file=output,
        )
        state_dict = torch.load(args.checkpoint)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            log(
                "Warning: Loading model with strict=False due to mismatch in state_dict keys",
                file=output,
            )
            model.load_state_dict(state_dict, strict=False)

    if use_cuda:
        model.cuda()

    # Train the model
    lr = args.lr
    wd = args.weight_decay
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    save_prefix = args.save_prefix

    
    tok_params = []
    base_params = []
    scalar_params = []
    scalar_names = ()

    tok_names = (
        # Legacy token classifier names
        "map_proj",
        "map_encoder",
        # CLEAR classifier names
        "map_stage1",
        "g_inject",
        "map_stage2",
        # Shared heads
        "feat_proj",
        "clf_head",
    )
        
    for n, p in model.named_parameters():
        if any(k in n for k in scalar_names):
            scalar_params.append(p)
            log(f"  scalar  {n}  {tuple(p.shape)}", file=output)
        elif any(k in n for k in tok_names):
            tok_params.append(p)
            log(f"  tok     {n}  {tuple(p.shape)}", file=output)
        else:
            base_params.append(p)
            log(f"  base    {n}  {tuple(p.shape)}", file=output)

    base_optim = torch.optim.AdamW(
        [
            {"params": base_params,   "lr": lr,        "weight_decay": wd},
            {"params": tok_params,    "lr": lr * 0.3,  "weight_decay": wd * 2.0},
            {"params": scalar_params, "lr": lr * 2.0,  "weight_decay": 0.0},
        ],
        eps=1e-6,
        betas=(0.9, 0.999),
    )


    optimizer = (
        optim.Lookahead(base_optim, k=5, alpha=0.5)
        if getattr(args, "use_lookahead", False)
        else base_optim
    )


    lr_manager = LRRestarter(
        base_optim,
        initial_T_max=8,  # full cycle initially
        restart_T_max=8,  # shorter cycles after restarts
        patience=4,  # wait 3 epochs of no improvement
        start_restart_epoch=8,  # don't restart before epoch 8
    )
    

        
    log(f'Using save prefix "{save_prefix}"', file=output)
    log(f"Training with Adam: lr={lr}, weight_decay={wd}", file=output)
    log(f"\tuse_lookahead: {getattr(args, 'use_lookahead', False)}", file=output)
    log(f"\tnum_epochs: {num_epochs}", file=output)
    log(f"\tbatch_size: {batch_size}", file=output)
    output.flush()

    batch_report_fmt = "[{}/{}] training {:.1%}: Loss={:.6}, Accuracy={:.3%}, MSE={:.6}"
    epoch_report_fmt = "Finished Epoch {}/{}: Loss={:.6}, Accuracy={:.3%}, MSE={:.6}, Precision={:.6}, Recall={:.6}, F1={:.6}, AUPR={:.6}"
    
    
    best_aupr = float("-inf")
    best_epoch = -1
    patience = 8
    min_delta = 1e-4
    bad_epochs = 0

    best_model_path = None
    if save_prefix is not None:
        best_model_path = save_prefix + "_best_model.sav"
    else:
        best_model_path = "best_model.sav"
       
    N = len(train_iterator) * batch_size
    global_step = 0
    steps_per_epoch = len(train_iterator)
    for epoch in range(num_epochs):
        epoch_start_time = time.perf_counter()
        model.train()

        n = 0
        loss_accum = 0
        acc_accum = 0
        mse_accum = 0
        bce_accum = 0
        
       
        # Train batches
        for batch_idx, (z0, z1, y) in enumerate(train_iterator):
            optimizer.zero_grad(set_to_none=True)
            loss, correct, mse, b, logits_used, loss_terms = interaction_grad(
                model,
                z0,
                z1,
                y,
                embeddings,
                run_tt=run_tt,
                glider_weight=glider_weight,
                glider_map=glider_map,
                glider_mat=glider_mat,
                use_cuda=use_cuda,
                structural_context=foldseek3dicontext,
            )
            loss.backward()

            loss_val = float(loss.detach().item())
            n += b
            loss_accum += b * (loss_val  - loss_accum) / n                
            acc_accum  += (correct - b * acc_accum) / n                 
            mse_accum  += b * (mse       - mse_accum)  / n
            bce_accum += b * (loss_terms["bce_loss"] - bce_accum) / n
            
            report = (n - b) // 100 < n // 100


            named_params = list(model.named_parameters())
            total_norm = _grad_l2_norm([p for _, p in named_params])
            grad_map_stage1 = _module_grad_norm(named_params, "clf.map_stage1")
            grad_g_inject = _module_grad_norm(named_params, "clf.g_inject")
            grad_map_stage2 = _module_grad_norm(named_params, "clf.map_stage2")
            grad_feat_proj = _module_grad_norm(named_params, "clf.feat_proj")
            grad_clf_head = _module_grad_norm(named_params, "clf.clf_head")

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.clip()
           
            global_step += 1

            if report:
                tokens = [
                    epoch + 1,
                    num_epochs,
                    n / N,
                    loss_accum,
                    acc_accum,
                    mse_accum,
                ]
                log(batch_report_fmt.format(*tokens), file=output)
                train_terms = [
                    f"bce={bce_accum:.6f}",
                    f"gnorm={total_norm:.4f}",
                    f"map1={grad_map_stage1:.4f}",
                    f"ginj={grad_g_inject:.4f}",
                    f"map2={grad_map_stage2:.4f}",
                    f"fproj={grad_feat_proj:.4f}",
                    f"head={grad_clf_head:.4f}",
                    f"lr_base={base_optim.param_groups[0]['lr']:.2e}",
                    f"lr_tok={base_optim.param_groups[1]['lr']:.2e}",
                ]
                log("train terms: " + " ".join(train_terms), file=output)

                
                if args.log_wandb:
                    log_payload = {
                        "train/loss": loss_accum,
                        "train/accuracy": acc_accum,
                        "train/mse": mse_accum,
                        "train/grad_norm": total_norm,
                        "train/bce_loss": bce_accum,
                        "train/grad_norm_map_stage1": grad_map_stage1,
                        "train/grad_norm_g_inject": grad_g_inject,
                        "train/grad_norm_map_stage2": grad_map_stage2,
                        "train/grad_norm_feat_proj": grad_feat_proj,
                        "train/grad_norm_clf_head": grad_clf_head,
                        "train/lr_base": base_optim.param_groups[0]["lr"],
                        "train/lr_tok": base_optim.param_groups[1]["lr"],
                    }
                    run.log(log_payload)

                output.flush()

        model.eval()
        lr_now = base_optim.param_groups[0]["lr"]   # ← check base_optim, not optimizer
        lr_tok = base_optim.param_groups[1]["lr"]
        lr_scalar = base_optim.param_groups[2]["lr"]
        log(
            f"Epoch {epoch+1}: lr_base={lr_now:.3e} lr_tok={lr_tok:.3e} lr_scalar={lr_scalar:.3e}",
            file=output,
        )
        epoch_terms = [
            f"bce={bce_accum:.6f}",
        ]
        log("Epoch train terms: " + " ".join(epoch_terms), file=output)

        _sync_cuda(use_cuda)
        inference_start_time = time.perf_counter()
        with torch.no_grad():
       
            
            (
                inter_loss,
                inter_correct,
                inter_mse,
                inter_pr,
                inter_re,
                inter_f1,
                inter_aupr,
            ) = interaction_eval(
                model, test_iterator, embeddings, use_cuda,foldseek3dicontext
            )
        _sync_cuda(use_cuda)
        inference_time = time.perf_counter() - inference_start_time
        total_inference_time += inference_time
            
        lr_manager.step()
        lr_manager.update(inter_aupr, epoch)  
        
        with torch.no_grad():  
            
            tokens = [
                epoch + 1,
                num_epochs,
                inter_loss,
                inter_correct / test_size,
                inter_mse,
                inter_pr,
                inter_re,
                inter_f1,
                inter_aupr,
            ]
            log(epoch_report_fmt.format(*tokens), file=output)
            epoch_runtime = time.perf_counter() - epoch_start_time
            log(
                f"Epoch {epoch+1} timing: runtime={_format_duration(epoch_runtime)}, inference={_format_duration(inference_time)}",
                file=output,
            )

            if args.log_wandb:
                run.log(
                    {
                        "val/loss": inter_loss,
                        "val/accuracy": inter_correct / test_size,
                        "val/mse": inter_mse,
                        "val/precision": inter_pr,
                        "val/recall": inter_re,
                        "val/f1": inter_f1,
                        "val/aupr": inter_aupr,
                    }
                )

            # ---- Early stopping on val AUPR (save only best)
            val_aupr = float(
                inter_aupr.item() if hasattr(inter_aupr, "item") else inter_aupr
            )

            val_aupr = float(inter_aupr.item() if hasattr(inter_aupr, "item") else inter_aupr)

            if val_aupr > best_aupr + min_delta:
                best_aupr = val_aupr
                best_epoch = epoch + 1
                bad_epochs = 0

                
                torch.save(model, best_model_path)
                log(f"[BEST] epoch {best_epoch}: val AUPR={best_aupr:.6f} -> saved {best_model_path}", file=output)
            else:
                bad_epochs += 1
                log(f"[BEST] no improvement (best epoch {best_epoch}, AUPR={best_aupr:.6f}) bad_epochs={bad_epochs}/{patience}", file=output)

            output.flush()

            if bad_epochs >= patience:
                log(f"[EarlyStop] stop at epoch {epoch+1}. best epoch {best_epoch}, best AUPR={best_aupr:.6f}", file=output)
                break

    total_runtime = time.perf_counter() - overall_start_time
    log(
        f"Training summary timing: runtime={_format_duration(total_runtime)}, total_inference={_format_duration(total_inference_time)}",
        file=output,
    )

    if save_prefix is not None:
        if args.log_wandb:
            # Upload trained model as artifact
            artifact = wandb.Artifact(
                name="trained-model",
                type="model",
                description="D-SCRIPT trained interaction model",
            )
            artifact.add_file(best_model_path)
            run.log_artifact(artifact)
            run.finish()

def main(args):
    """
    Run training from arguments.

    :meta private:
    """

    output = args.outfile
    if output is None:
        output = sys.stdout
    else:
        output = open(output, "w")

    log(f"D-SCRIPT Version {__version__}", file=output, print_also=True)
    log(f"Called as: {' '.join(sys.argv)}", file=output, print_also=True)

    # Set the device
    device = args.device
    use_cuda = (device > -1) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        log(
            f"Using CUDA device {device} - {torch.cuda.get_device_name(device)}",
            file=output,
            print_also=True,
        )
    else:
        log("Using CPU", file=output, print_also=True)
        device = "cpu"

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    train_model(args, output)

    output.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
