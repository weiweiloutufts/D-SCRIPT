"""Train the torch residual interaction model with EMA soft targets."""

from __future__ import annotations

import argparse
import copy
import re
import shutil
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
from ..models.embedding import FullyConnectedEmbed
from ..models.interaction_res_torch_soft_contact_not import (
    InteractionInputs,
    ModelInteraction as SoftContactModelInteraction,
)
from ..models.contact_torch import ContactCNN as TorchContactCNN


class ModelInteraction(SoftContactModelInteraction):
    """Adapt the soft-contact model to the shared training-loop constructor."""

    def __init__(self, *args, spiral_turns=None, **kwargs):
        super().__init__(*args, **kwargs)


from ..parallel_embedding_loader import (
    add_batch_dim_if_needed,
    LazyEmbeddingStore,
)
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
    spiral_turns: float
    noise_std: float
    attn_pool_size: int
    attn_noise_std: float
    attn_dropout: float
    contact_model: str
    map_out_channels: int
    no_w: bool
    no_sigmoid: bool
    do_pool: bool
    pool_width: int
    num_epochs: int
    early_stop_patience: int
    early_stop_min_delta: float
    negative_center_loss_weight: float
    negative_center_loss_min_count: int
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
    allow_prostt5: bool
    prostt5_dir: str | None
    prostt5_dim: int
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


def _get_contact_cnn(contact_model: str):
    if contact_model != "contact_torch":
        raise ValueError("--contact-model must be contact_torch")
    return TorchContactCNN


def _contact_attention_learned_r(model) -> float | None:
    """Safely retrieve the contact-side spiral r when available.

    Supports contact-map attention (``contact.cmap_attn.sa``) and
    embedding-level attention (``contact.emb_attn.sa``). Returns None if the
    model does not have either block.
    """
    contact = getattr(model, "contact", None)
    attn_block = getattr(contact, "cmap_attn", None)
    if attn_block is None:
        attn_block = getattr(contact, "emb_attn", None)
    spiral_attn = getattr(attn_block, "sa", None)
    if spiral_attn is None or not hasattr(spiral_attn, "get_learned_r"):
        return None
    return spiral_attn.get_learned_r()


def _extract_auprs_from_log(path: Path) -> list[float]:
    pattern = re.compile(r"(?:^|[^A-Za-z])AUPR\s*=\s*([0-9]*\.?[0-9]+)", re.I)
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return []
    return [float(match.group(1)) for match in pattern.finditer(text)]


def _past_best_aupr(results_root: Path, current_outfile: str | None) -> float:
    current_path = Path(current_outfile).resolve() if current_outfile else None
    best = float("-inf")
    if not results_root.is_dir():
        return best
    for log_path in results_root.rglob("results.log"):
        try:
            if current_path is not None and log_path.resolve() == current_path:
                continue
        except OSError:
            continue
        auprs = _extract_auprs_from_log(log_path)
        if auprs:
            best = max(best, max(auprs))
    return best


def _save_global_best_code_snapshot(
    *,
    save_prefix: str | None,
    best_model_path: str,
    epoch: int,
    aupr: float,
    args: TrainArguments,
) -> tuple[str | None, float]:
    """Copy source files when this run beats all past results in its family."""
    repo_root = Path(__file__).resolve().parents[2]
    if save_prefix is not None:
        base = Path(save_prefix)
    else:
        base = Path(best_model_path).with_suffix("")

    run_dir = base.parent
    results_root = run_dir.parent
    past_best = _past_best_aupr(results_root, args.outfile)
    if aupr <= past_best:
        return None, past_best

    snapshot_root = results_root / "_global_best_code"
    snapshot_dir = snapshot_root / f"global_best_epoch{epoch:03d}_aupr{aupr:.6f}_{run_dir.name}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Soft-target trainers delegate their training loop to this module. Infer
    # the entry point from the recorded command so their snapshots contain the
    # code that actually defined the run, rather than only the base trainer.
    command_name = Path(str(args.cmd).split()[0]).stem
    variant_files = {
        "train_res_torch_soft": [
            "dscript/commands/train_res_torch_soft.py",
            "dscript/models/interaction_res_torch_soft.py",
            "bash/train_res_torch_soft_bernett.sh",
        ],
        "train_res_torch_soft_contact": [
            "dscript/commands/train_res_torch_soft_contact.py",
            "dscript/models/contact_torch.py",
            "dscript/models/interaction_res_torch_soft_contact.py",
            "bash/train_res_torch_soft_contact_bernett.sh",
        ],
        "train_res_torch_soft_ema": [
            "dscript/commands/train_res_torch_soft_ema.py",
            "dscript/models/interaction_res_torch_soft_ema.py",
            "dscript/models/interaction_torch_soft_ema.py",
            "bash/train_res_torch_soft_ema_bernett.sh",
        ],
    }
    files_to_copy = [
        "dscript/commands/train_res_torch_pos.py",
        "dscript/models/contact_torch_pos.py",
        "dscript/models/interaction_res_torch_pos.py",
        "dscript/models/spiral_attention.py",
        "spiral_attention/spiral_attention.py",
    ]
    files_to_copy.extend(
        variant_files.get(command_name, ["bash/train_res_torch_pos_bernett.sh"])
    )

    for rel in files_to_copy:
        src = repo_root / rel
        if src.is_file():
            dst = snapshot_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    manifest = snapshot_dir / "MANIFEST.txt"
    with manifest.open("w", encoding="utf-8") as fh:
        fh.write("snapshot_type: global_best_across_past_results\n")
        fh.write(f"results_root: {results_root}\n")
        fh.write(f"previous_best_aupr: {past_best:.6f}\n")
        fh.write(f"best_epoch: {epoch}\n")
        fh.write(f"best_aupr: {aupr:.6f}\n")
        fh.write(f"best_model_path: {best_model_path}\n")
        fh.write(f"command: {args.cmd}\n")
        fh.write(f"seed: {args.seed}\n")
        fh.write(f"spiral_turns: {args.spiral_turns}\n")
        fh.write(f"batch_size: {args.batch_size}\n")
        fh.write(f"classifier_d_model: {args.classifier_d_model}\n")
        fh.write(f"hidden_dim: {args.hidden_dim}\n")
        fh.write(f"map_out_channels: {args.map_out_channels}\n")
        fh.write(f"attn_dropout: {args.attn_dropout}\n")
        if hasattr(args, "soft_target_lambda"):
            fh.write(f"soft_target_lambda: {args.soft_target_lambda}\n")
            fh.write(f"soft_target_ema_decay: {args.soft_target_ema_decay}\n")
            fh.write(
                "soft_target_positive_only: "
                f"{args.soft_target_positive_only}\n"
            )

    latest = snapshot_root / "LATEST_GLOBAL_BEST.txt"
    with latest.open("w", encoding="utf-8") as fh:
        fh.write(f"best_aupr: {aupr:.6f}\n")
        fh.write(f"best_epoch: {epoch}\n")
        fh.write(f"snapshot_dir: {snapshot_dir}\n")
        fh.write(f"best_model_path: {best_model_path}\n")
        fh.write(f"command: {args.cmd}\n")

    return str(snapshot_dir), past_best


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
    prostt5_grp = parser.add_argument_group("ProstT5 related commands")

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
        help="pooled grid size for the residual map classifier (default: 100)",
    )
    inter_grp.add_argument(
        "--classifier-d-model",
        type=int,
        default=64,
        help="hidden width for PairClassifierTokens (default: 64)",
    )
    inter_grp.add_argument(
        "--spiral-turns",
        type=float,
        default=1.0,
        help=(
            "compatibility argument retained from spiral experiments; "
            "contact_torch_pos uses standard torch attention (default: 1.0)"
        ),
    )
    inter_grp.add_argument(
        "--noise-std",
        type=float,
        default=0.05,
        help=(
            "compatibility argument retained from spiral experiments; "
            "not used by contact_torch_pos (default: 0.05)"
        ),
    )
    inter_grp.add_argument(
        "--attn-pool-size",
        type=int,
        default=16,
        help=(
            "spatial pool size for ContactMapAttention before self-attention; "
            "attn_pool_size×attn_pool_size tokens are used (default: 16 → 256 tokens)"
        ),
    )
    inter_grp.add_argument(
        "--attn-noise-std",
        type=float,
        default=0.05,
        help=(
            "compatibility argument retained for ContactCNN constructor; "
            "not used by torch attention (default: 0.05)"
        ),
    )
    inter_grp.add_argument(
        "--attn-dropout",
        type=float,
        default=0.0,
        help=(
            "dropout probability inside ContactMapAttention "
            "(default: 0.0)"
        ),
    )
    inter_grp.add_argument(
        "--contact-model",
        choices=("contact_torch",),
        default="contact_torch",
        help=(
            "ContactCNN implementation to use. This command is fixed to "
            "contact_torch (default: contact_torch)"
        ),
    )
    inter_grp.add_argument(
        "--map-out-channels",
        type=int,
        default=8,
        help=(
            "number of output channels from ContactCNN; these flow directly "
            "into PairClassifierTokens as map_in_channels (default: 8)"
        ),
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
        "--early-stop-patience",
        type=int,
        default=3,
        help=(
            "stop after this many validation epochs without AUPR improvement "
            "(default: 3)"
        ),
    )
    train_grp.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help=(
            "minimum validation AUPR increase required to reset early-stop "
            "patience (default: 1e-4)"
        ),
    )
    train_grp.add_argument(
        "--negative-center-loss-weight",
        type=float,
        default=0.0,
        help=(
            "auxiliary weight for pulling negative pair features toward their "
            "batch centroid; 0 disables it (default: 0)"
        ),
    )
    train_grp.add_argument(
        "--negative-center-loss-min-count",
        type=int,
        default=2,
        help=(
            "minimum number of negative examples required in a batch before "
            "applying negative-center loss (default: 2)"
        ),
    )
    train_grp.add_argument(
        "--mixup-weight",
        type=float,
        default=0.0,
        help=(
            "weight for mixed-feature BCE when mixup is enabled; the remaining "
            "weight stays on ordinary BCE; 0 disables mixup (default: 0)"
        ),
    )
    train_grp.add_argument(
        "--mixup-lam-min",
        type=float,
        default=None,
        help="optional lower bound for uniform mixup lambda; use with --mixup-lam-max",
    )
    train_grp.add_argument(
        "--mixup-lam-max",
        type=float,
        default=None,
        help="optional upper bound for uniform mixup lambda; use with --mixup-lam-min",
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
        # FIX 7: was 0 — small non-zero default prevents overfitting
        default=1e-4,
        help="L2 regularization (default: 1e-4)",
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

    ## ProstT5 arguments
    prostt5_grp.add_argument(
        "--allow_prostt5",
        default=False,
        action="store_true",
        help="If set to true, use ProstT5 embeddings instead of ESM2",
    )
    prostt5_grp.add_argument(
        "--prostt5_dir",
        help="Directory containing ProstT5 embeddings for proteins",
    )
    prostt5_grp.add_argument(
        "--prostt5_dim",
        type=int,
        default=512,
        help="Dimension of ProstT5 embeddings (default: 512), used as input to FullyConnectedEmbed",
    )

    train_grp.add_argument(
        "--soft-target-lambda",
        type=float,
        default=0.8,
        help=(
            "label contribution to the soft target: lambda * label + "
            "(1 - lambda) * EMA-teacher prediction (default: 0.8)"
        ),
    )
    train_grp.add_argument(
        "--soft-target-ema-decay",
        type=float,
        default=0.999,
        help="EMA decay used to update the teacher model (default: 0.999)",
    )
    train_grp.add_argument(
        "--soft-target-positive-only",
        action="store_true",
        help="apply EMA soft targets only to positive labels",
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

        interaction_inputs = InteractionInputs(
            z_a, z_b,
            embed_foldseek=(structural_context is not None and structural_context.allow_foldseek),
            f0=f_a, f1=f_b,
            embed_backbone=(structural_context is not None and structural_context.allow_backbone3di),
            b0=b_a, b1=b_b,
        )
        try:
            outputs = model.map_predict(interaction_inputs)
            if use_cuda:
                torch.cuda.synchronize()
        except RuntimeError:
            def _shape(x):
                return None if x is None else tuple(x.shape)

            print(
                "[train_res_torch_soft_contact] forward failed for "
                f"pair_index={i}, n0={n0[i]}, n1={n1[i]}, "
                f"z0_shape={tuple(z_a.shape)}, z1_shape={tuple(z_b.shape)}, "
                f"z0_dtype={z_a.dtype}, z1_dtype={z_b.dtype}, "
                f"z0_device={z_a.device}, z1_device={z_b.device}, "
                f"foldseek_shapes={_shape(f_a)}/{_shape(f_b)}, "
                f"backbone_shapes={_shape(b_a)}/{_shape(b_b)}",
                file=sys.stderr,
            )
            raise
        _, logit, feat = outputs
        return logit.reshape(-1), feat

    with ctx:
        pair_outputs = tuple(_predict_one(i) for i in range(b))
        logits, feat = (
            torch.cat(parts, dim=0) for parts in zip(*pair_outputs)
        )

    return logits, feat, {}

def predict_interaction(model, n0, n1, tensors, use_cuda, structural_context=None):
    logits, _, _ = predict_cmap_interaction(
        model, n0, n1, tensors, use_cuda, structural_context
    )
    return logits


def smooth_labels(labels: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """Apply label smoothing: pushes hard 0/1 labels toward 0.5.

    Args:
        labels: Float tensor of binary labels in {0, 1}.
        smoothing: Smoothing factor in [0, 1). 0 = no smoothing.

    Returns:
        Smoothed label tensor of the same shape and dtype.
    """
    return labels * (1.0 - smoothing) + 0.5 * smoothing


def mixup_feat(
    feat: torch.Tensor,
    y: torch.Tensor,
    lam_min: float,
    lam_max: float,
):
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

    lam = torch.empty(B, device=device, dtype=torch.float32).uniform_(lam_min, lam_max)

    index = torch.randperm(B, device=device)

    # broadcast lambda over feature dims
    lam_feat = lam.view(B, *([1] * (feat.dim() - 1))).to(dtype=feat.dtype)

    feat_mix = lam_feat * feat + (1.0 - lam_feat) * feat[index]
    y_mix = lam * y + (1.0 - lam) * y[index]   # keep y_mix float32

    return feat_mix, y_mix


def negative_center_loss(
    feat: torch.Tensor,
    y: torch.Tensor,
    min_count: int = 2,
) -> torch.Tensor:
    neg_mask = y.view(-1).to(device=feat.device) < 0.5
    if int(neg_mask.sum().item()) < min_count:
        return feat.new_zeros(())

    neg_feat = F.normalize(feat[neg_mask].float(), dim=1)
    center = neg_feat.mean(dim=0, keepdim=True)
    return ((neg_feat - center) ** 2).sum(dim=1).mean()


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
    label_smoothing: float = 0.05,
    negative_center_loss_weight: float = 0.0,
    negative_center_loss_min_count: int = 2,
    mixup_weight: float = 0.0,
    mixup_lam_min: float | None = None,
    mixup_lam_max: float | None = None,
    soft_target_lambda: float = 0.8,
    teacher_model=None,
    soft_target_positive_only: bool = False,
):
    logits, feat, _ = predict_cmap_interaction(
        model,
        n0,
        n1,
        tensors,
        use_cuda,
        structural_context,
        require_grad=True,
    )
    b = len(n0)
    labels = (y.cuda() if use_cuda else y).float().view(-1)
    pred = logits.view(-1).float()

    if teacher_model is None:
        raise ValueError("an EMA teacher_model is required for soft targets")

    with torch.no_grad():
        teacher_logits, _, _ = predict_cmap_interaction(
            teacher_model,
            n0,
            n1,
            tensors,
            use_cuda,
            structural_context,
            require_grad=False,
        )
        teacher_probability = torch.sigmoid(teacher_logits.view(-1).float())

    lam = float(soft_target_lambda)
    blended_target = lam * labels + (1.0 - lam) * teacher_probability
    if soft_target_positive_only:
        positive_mask = labels > 0.5
        soft_target = torch.where(positive_mask, blended_target, labels)
    else:
        soft_target = blended_target

    soft_target = smooth_labels(soft_target, smoothing=label_smoothing)
    cls_loss = F.binary_cross_entropy_with_logits(pred, soft_target)

    if run_tt:
        glider_target = torch.tensor(
            [glider_score(n0[i], n1[i], glider_map, glider_mat) for i in range(b)],
            dtype=torch.float32,
            device=pred.device,
        )
        glider_loss = F.binary_cross_entropy_with_logits(pred, glider_target)
        cls_loss = glider_weight * glider_loss + (1.0 - glider_weight) * cls_loss

    mixup_loss = pred.new_zeros(())
    if (
        mixup_weight > 0.0
        and mixup_lam_min is not None
        and mixup_lam_max is not None
        and feat.size(0) >= 2
    ):
        feat_mix, y_mix = mixup_feat(
            feat,
            labels,
            lam_min=mixup_lam_min,
            lam_max=mixup_lam_max,
        )
        mixup_logits = model.clf.classify_from_feat(feat_mix).view(-1).float()
        y_mix = y_mix.to(mixup_logits.device)
        mixup_soft = smooth_labels(y_mix, smoothing=label_smoothing)
        mixup_loss = F.binary_cross_entropy_with_logits(mixup_logits, mixup_soft)
        cls_loss = (1.0 - mixup_weight) * cls_loss + mixup_weight * mixup_loss

    neg_center_loss_value = pred.new_zeros(())
    if negative_center_loss_weight > 0:
        neg_center_loss_value = negative_center_loss(
            feat,
            labels,
            min_count=negative_center_loss_min_count,
        )

    loss = cls_loss + negative_center_loss_weight * neg_center_loss_value

    with torch.no_grad():
        probability = torch.sigmoid(pred)
        y_metrics = labels.to(pred.device)
        correct = ((probability > 0.5).float() == y_metrics).sum().item()
        mse = ((y_metrics - probability) ** 2).mean().item()
        assert torch.isfinite(pred).all()

    loss_terms = {
        "cls_loss": float(cls_loss.detach().item()),
        "mixup_loss": float(mixup_loss.detach().item()),
        "neg_center_loss": float(neg_center_loss_value.detach().item()),
    }
    return loss, correct, mse, b, pred, loss_terms


class EMASoftTargetObjective:
    """Stateful loss callable whose teacher follows the optimized student."""

    def __init__(
        self,
        *,
        soft_target_lambda: float,
        ema_decay: float,
        positive_only: bool,
        negative_center_loss_weight: float,
        negative_center_loss_min_count: int,
        mixup_weight: float,
        mixup_lam_min: float | None,
        mixup_lam_max: float | None,
    ):
        self.soft_target_lambda = float(soft_target_lambda)
        self.ema_decay = float(ema_decay)
        self.positive_only = bool(positive_only)
        self.negative_center_loss_weight = float(negative_center_loss_weight)
        self.negative_center_loss_min_count = int(negative_center_loss_min_count)
        self.mixup_weight = float(mixup_weight)
        self.mixup_lam_min = mixup_lam_min
        self.mixup_lam_max = mixup_lam_max
        self.teacher_model = None

    @torch.no_grad()
    def _prepare_teacher(self, model):
        if self.teacher_model is None:
            self.teacher_model = copy.deepcopy(model)
            self.teacher_model.requires_grad_(False)
        else:
            teacher_state = self.teacher_model.state_dict()
            student_state = model.state_dict()
            for name, teacher_value in teacher_state.items():
                student_value = student_state[name]
                if teacher_value.is_floating_point():
                    teacher_value.mul_(self.ema_decay).add_(
                        student_value,
                        alpha=1.0 - self.ema_decay,
                    )
                else:
                    teacher_value.copy_(student_value)
        self.teacher_model.eval()

    def __call__(self, model, *args, **kwargs):
        self._prepare_teacher(model)
        kwargs.setdefault("negative_center_loss_weight", self.negative_center_loss_weight)
        kwargs.setdefault("negative_center_loss_min_count", self.negative_center_loss_min_count)
        kwargs.setdefault("mixup_weight", self.mixup_weight)
        kwargs.setdefault("mixup_lam_min", self.mixup_lam_min)
        kwargs.setdefault("mixup_lam_max", self.mixup_lam_max)
        return interaction_grad(
            model,
            *args,
            **kwargs,
            teacher_model=self.teacher_model,
            soft_target_lambda=self.soft_target_lambda,
            soft_target_positive_only=self.positive_only,
        )


def interaction_eval(
    model,
    test_iterator,
    tensors,
    use_cuda,
    structural_context=None,
):
    """
    Evaluate test data set performance.

    :param model: Model to be trained
    :type model: dscript.models.interaction_res.ModelInteraction
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
        logits = predict_interaction(model, n0, n1, tensors, use_cuda, structural_context)
        p_hat_list.append(logits.view(-1))
        y_list.append(y.view(-1))

    logits = torch.cat(p_hat_list, dim=0)
    y = torch.cat(y_list, dim=0).float()

    device = logits.device
    y = y.to(device)

    # Eval loss uses hard labels (no smoothing) for honest reporting
    loss = F.binary_cross_entropy_with_logits(logits, y)

    with torch.no_grad():
        p = torch.sigmoid(logits)

        pred = (p > 0.5).float()
        correct = (pred == y).sum().item()

        mse = torch.mean((y - p) ** 2).item()

        tp = torch.sum(pred * y).item()
        fp = torch.sum(pred * (1 - y)).item()
        fn = torch.sum((1 - pred) * y).item()

        pr = tp / (tp + fp + 1e-8)
        re = tp / (tp + fn + 1e-8)
        f1 = 2 * pr * re / (pr + re + 1e-8)

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


def _log_parameter_summary(model, output):
    total_params, trainable_params = _count_parameters(model)
    log(
        f"Model parameters: total={total_params:,}, trainable={trainable_params:,}, frozen={total_params - trainable_params:,}",
        file=output,
    )

    for module_name, module in model.named_children():
        module_total, module_trainable = _count_parameters(module)
        log(
            f"\t{module_name}: total={module_total:,}, trainable={module_trainable:,}",
            file=output,
        )


def _sync_cuda(use_cuda):
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


def _format_duration(seconds):
    total_seconds = max(float(seconds), 0.0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


def train_model(args, output, interaction_grad_fn=None):
    if interaction_grad_fn is None:
        interaction_grad_fn = interaction_grad

    overall_start_time = time.perf_counter()
    total_inference_time = 0.0

    if args.log_wandb:
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
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
        if h5py.is_hdf5(str(emb_path)):
            embedding_mode = "hdf5"
            log(f"Embedding path is an HDF5 file: {emb_path}")
        else:
            raise ValueError(
                f"Embedding file is not HDF5 and not a directory: {emb_path}"
            )
    else:
        raise FileNotFoundError(f"Embedding path does not exist: {emb_path}")

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
        allow_foldseek=allow_foldseek,
        fold_record=fold_record,
        fold_vocab=fold_vocab,
        allow_backbone3di=allow_backbone3di,
        backbone_record=backbone_record,
        backbone_vocab=backbone_vocab,
    )

    allow_prostt5 = args.allow_prostt5

    if allow_prostt5:
        if args.prostt5_dir is None:
            raise ValueError("--prostt5_dir must be specified when --allow_prostt5 is set")
        prostt5_path = Path(args.prostt5_dir)
        if not prostt5_path.is_dir():
            raise FileNotFoundError(f"ProstT5 directory does not exist: {prostt5_path}")
        log(f"ProstT5 embeddings will be loaded from: {prostt5_path}", file=output)

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

    if allow_prostt5:
        log(f"Validating ProstT5 embeddings from {args.prostt5_dir}...", file=output)
        embeddings = LazyEmbeddingStore(
            prostt5_path,
            all_proteins,
            mode="pt_dir",
            cache_size=256,
            num_workers=4,
        )
    else:
        if embedding_mode == "pt_dir":
            log(f"Validating embeddings from {emb_path} for lazy loading...", file=output)
            embeddings = LazyEmbeddingStore(
                emb_path,
                all_proteins,
                mode="pt_dir",
                cache_size=256,
                num_workers=4,
            )
        elif embedding_mode == "hdf5":
            log(f"Validating HDF5 embeddings from {emb_path} for lazy loading...", file=output)
            embeddings = LazyEmbeddingStore(
                emb_path,
                all_proteins,
                mode="hdf5",
                cache_size=256,
            )

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
    projection_dim = args.projection_dim
    dropout_p = args.dropout_p

    if allow_prostt5:
        input_dim = args.prostt5_dim
        log("Using ProstT5 embeddings - projecting from ProstT5 dimension", file=output)
        log(f"\tInput dimension (ProstT5): {input_dim}", file=output)
    else:
        input_dim = args.input_dim
        log("Using ESM2 embeddings - projecting from ESM2 dimension", file=output)
        log(f"\tInput dimension (ESM2): {input_dim}", file=output)

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

    attn_pool_size = args.attn_pool_size
    attn_noise_std = args.attn_noise_std
    attn_dropout = args.attn_dropout
    spiral_turns = args.spiral_turns
    contact_model_name = args.contact_model
    map_out_channels = args.map_out_channels
    ContactCNN = _get_contact_cnn(contact_model_name)
    log(f"\tcontact_model: {contact_model_name}", file=output)
    log(f"\tattn_pool_size: {attn_pool_size}", file=output)
    log(f"\tattn_noise_std: {attn_noise_std} (compatibility; unused)", file=output)
    log(f"\tattn_dropout: {attn_dropout}", file=output)
    log(f"\tattn_spiral_turns: {spiral_turns} (compatibility; unused)", file=output)
    log(f"\tmap_out_channels: {map_out_channels}", file=output)

    contact_kwargs = dict(
        out_channels=map_out_channels,
        attn_pool_size=attn_pool_size,
        attn_dropout=attn_dropout,
        attn_noise_std=attn_noise_std,
        attn_spiral_turns=spiral_turns,
    )

    contact_model = ContactCNN(
        proj_dim,
        hidden_dim,
        kernel_width,
        **contact_kwargs,
    )

    # Create the full model
    do_w = not args.no_w
    do_pool = args.do_pool
    pool_width = args.pool_width
    grid_size = args.grid_size
    classifier_d_model = args.classifier_d_model
    noise_std = args.noise_std
    do_sigmoid = not args.no_sigmoid
    prostt5_dim = args.prostt5_dim
    log("Initializing interaction model with:", file=output)
    log(f"\tdo_pool: {do_pool}", file=output)
    log(f"\tpool_width: {pool_width}", file=output)
    log(f"\tgrid_size: {grid_size}", file=output)
    log(f"\tclassifier_d_model: {classifier_d_model}", file=output)
    log(f"\tspiral_turns: {spiral_turns} (compatibility; unused)", file=output)
    log(f"\tnoise_std: {noise_std} (compatibility; unused)", file=output)
    log(f"\tdo_w: {do_w}", file=output)
    log(f"\tdo_sigmoid: {do_sigmoid}", file=output)
    log(f"\tallow_prostt5: {allow_prostt5}", file=output)
    log(f"\tprostt5_dim: {prostt5_dim}", file=output)
    if allow_prostt5:
        log(f"\tEmbedding source: ProstT5 (projected from {args.prostt5_dim} -> {projection_dim})", file=output)
    else:
        log(f"\tEmbedding source: ESM2 (projected from {args.input_dim} -> {projection_dim})", file=output)

    log(f"\tmap_out_channels: {map_out_channels}", file=output)
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
        spiral_turns=spiral_turns,
        noise_std=noise_std,
        map_out_channels=map_out_channels,
    )
    model.use_cuda = use_cuda

    log(model, file=output)
    _log_parameter_summary(model, output)

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

    for n, p in model.named_parameters():
        if p.requires_grad:
            log(f"  train   {n}  {tuple(p.shape)}", file=output)
        else:
            log(f"  frozen  {n}  {tuple(p.shape)}", file=output)

    base_optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
        eps=1e-6,
        betas=(0.9, 0.999),
    )

    # FIX 8: warn when Lookahead is combined with LR restarts — the restarter
    # creates a new CosineAnnealingLR that wraps base_optim directly, which
    # can desync Lookahead's slow-weight buffer after a restart.
    use_lookahead = getattr(args, "use_lookahead", False)
    if use_lookahead:
        log(
            "Warning: LR restarts (LRRestarter) may desync Lookahead slow-weight "
            "buffers after a restart. Consider disabling --use-lookahead if you "
            "observe instability after epoch restarts.",
            file=output,
        )

    optimizer = (
        optim.Lookahead(base_optim, k=5, alpha=0.5)
        if use_lookahead
        else base_optim
    )

    lr_manager = LRRestarter(
        base_optim,
        initial_T_max=8,
        restart_T_max=8,
        patience=2,
        # FIX 9: was 8 — with default num_epochs=10 this left almost no time
        # for restarts to have effect. Lowered to 4 so restarts can trigger
        # in the middle of a standard 10-epoch run. Lowered again to 2 for
        # short sweeps where validation often peaks around epoch 2.
        start_restart_epoch=2,
    )

    log(f'Using save prefix "{save_prefix}"', file=output)
    log(f"Training with Adam: lr={lr}, weight_decay={wd}", file=output)
    log(f"\tuse_lookahead: {use_lookahead}", file=output)
    log(f"\tnum_epochs: {num_epochs}", file=output)
    log(f"\tearly_stop_patience: {args.early_stop_patience}", file=output)
    log(f"\tearly_stop_min_delta: {args.early_stop_min_delta}", file=output)
    log(
        f"\tnegative_center_loss_weight: {args.negative_center_loss_weight}",
        file=output,
    )
    log(
        f"\tnegative_center_loss_min_count: {args.negative_center_loss_min_count}",
        file=output,
    )
    log(f"\tmixup_weight: {args.mixup_weight}", file=output)
    log(f"\tmixup_lam_min: {args.mixup_lam_min}", file=output)
    log(f"\tmixup_lam_max: {args.mixup_lam_max}", file=output)
    log(f"\tbatch_size: {batch_size}", file=output)
    output.flush()

    batch_report_fmt = "[{}/{}] training {:.1%}: Loss={:.6}, Accuracy={:.3%}, MSE={:.6}"
    epoch_report_fmt = "Finished Epoch {}/{}: Loss={:.6}, Accuracy={:.3%}, MSE={:.6}, Precision={:.6}, Recall={:.6}, F1={:.6}, AUPR={:.6}"

    best_aupr = float("-inf")
    best_epoch = -1
    patience = args.early_stop_patience
    min_delta = args.early_stop_min_delta
    if patience < 1:
        raise ValueError("--early-stop-patience must be >= 1")
    if min_delta < 0:
        raise ValueError("--early-stop-min-delta must be >= 0")
    if args.negative_center_loss_weight < 0:
        raise ValueError("--negative-center-loss-weight must be >= 0")
    if args.negative_center_loss_min_count < 1:
        raise ValueError("--negative-center-loss-min-count must be >= 1")
    if not 0.0 <= args.mixup_weight <= 1.0:
        raise ValueError("--mixup-weight must be between 0 and 1")
    if (args.mixup_lam_min is None) != (args.mixup_lam_max is None):
        raise ValueError("--mixup-lam-min and --mixup-lam-max must be set together")
    if args.mixup_lam_min is not None:
        if not 0.0 <= args.mixup_lam_min <= args.mixup_lam_max <= 1.0:
            raise ValueError("mixup lambda bounds must satisfy 0 <= min <= max <= 1")
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
        cls_accum = 0
        mixup_accum = 0
        neg_center_accum = 0

        for batch_idx, (z0, z1, y) in enumerate(train_iterator):
            optimizer.zero_grad(set_to_none=True)
            grad_kwargs = {}
            if interaction_grad_fn is interaction_grad:
                grad_kwargs = {
                    "negative_center_loss_weight": args.negative_center_loss_weight,
                    "negative_center_loss_min_count": args.negative_center_loss_min_count,
                    "mixup_weight": args.mixup_weight,
                    "mixup_lam_min": args.mixup_lam_min,
                    "mixup_lam_max": args.mixup_lam_max,
                }
            loss, correct, mse, b, logits, loss_terms = interaction_grad_fn(
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
                **grad_kwargs,
            )
            loss.backward()

            loss_val = float(loss.detach().item())
            n += b
            loss_accum += b * (loss_val  - loss_accum) / n
            acc_accum  += (correct - b * acc_accum) / n
            mse_accum  += b * (mse       - mse_accum)  / n
            cls_accum += b * (loss_terms["cls_loss"] - cls_accum) / n
            mixup_accum += b * (loss_terms.get("mixup_loss", 0.0) - mixup_accum) / n
            neg_center_accum += b * (
                loss_terms.get("neg_center_loss", 0.0) - neg_center_accum
            ) / n

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
                current_learned_r = _contact_attention_learned_r(model)
                train_terms = [
                    f"cls={cls_accum:.6f}",
                    f"gnorm={total_norm:.4f}",
                    f"map1={grad_map_stage1:.4f}",
                    f"ginj={grad_g_inject:.4f}",
                    f"map2={grad_map_stage2:.4f}",
                    f"fproj={grad_feat_proj:.4f}",
                    f"head={grad_clf_head:.4f}",
                    f"lr={base_optim.param_groups[0]['lr']:.2e}",
                ]
                if current_learned_r is not None:
                    train_terms.append(f"contact_learned_r={current_learned_r:.4f}")
                if args.negative_center_loss_weight > 0:
                    train_terms.append(f"neg_center={neg_center_accum:.6f}")
                if args.mixup_weight > 0:
                    train_terms.append(f"mixup={mixup_accum:.6f}")
                log("train terms: " + " ".join(train_terms), file=output)

                if args.log_wandb:
                    wandb_payload = {
                        "train/loss": loss_accum,
                        "train/accuracy": acc_accum,
                        "train/mse": mse_accum,
                        "train/grad_norm": total_norm,
                        "train/cls_loss": cls_accum,
                        "train/grad_norm_map_stage1": grad_map_stage1,
                        "train/grad_norm_g_inject": grad_g_inject,
                        "train/grad_norm_map_stage2": grad_map_stage2,
                        "train/grad_norm_feat_proj": grad_feat_proj,
                        "train/grad_norm_clf_head": grad_clf_head,
                        "train/lr": base_optim.param_groups[0]["lr"],
                    }
                    if args.negative_center_loss_weight > 0:
                        wandb_payload["train/neg_center_loss"] = neg_center_accum
                    if args.mixup_weight > 0:
                        wandb_payload["train/mixup_loss"] = mixup_accum
                    if current_learned_r is not None:
                        wandb_payload["train/contact_learned_r"] = current_learned_r
                    run.log(wandb_payload)

                output.flush()

        model.eval()
        lr_now = base_optim.param_groups[0]["lr"]
        # Log the learned contact-map spiral r via the helper so this works
        # whether or not ContactMapAttention is present in the model.
        learned_r = _contact_attention_learned_r(model)
        r_term = (
            f" contact_learned_r={learned_r:.4f}"
            if learned_r is not None
            else ""
        )
        log(
            f"Epoch {epoch+1}: lr={lr_now:.3e}{r_term}",
            file=output,
        )
        epoch_terms = [f"cls={cls_accum:.6f}"]
        if args.negative_center_loss_weight > 0:
            epoch_terms.append(f"neg_center={neg_center_accum:.6f}")
        if args.mixup_weight > 0:
            epoch_terms.append(f"mixup={mixup_accum:.6f}")
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
                model, test_iterator, embeddings, use_cuda, foldseek3dicontext
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
                wandb_payload = {
                    "val/loss": inter_loss,
                    "val/accuracy": inter_correct / test_size,
                    "val/mse": inter_mse,
                    "val/precision": inter_pr,
                    "val/recall": inter_re,
                    "val/f1": inter_f1,
                    "val/aupr": inter_aupr,
                }
                if learned_r is not None:
                    wandb_payload["val/contact_learned_r"] = learned_r
                run.log(wandb_payload)

            val_aupr = float(
                inter_aupr.item() if hasattr(inter_aupr, "item") else inter_aupr
            )

            if val_aupr > best_aupr + min_delta:
                best_aupr = val_aupr
                best_epoch = epoch + 1
                bad_epochs = 0

                torch.save(model, best_model_path)
                log(
                    f"[BEST] epoch {best_epoch}: val AUPR={best_aupr:.6f} -> saved {best_model_path}",
                    file=output,
                )
            else:
                bad_epochs += 1
                log(f"[BEST] no improvement (best epoch {best_epoch}, AUPR={best_aupr:.6f}) bad_epochs={bad_epochs}/{patience}", file=output)

            output.flush()

            if bad_epochs >= patience:
                log(f"[EarlyStop] stop at epoch {epoch+1}. best epoch {best_epoch}, best AUPR={best_aupr:.6f}", file=output)
                break

    if best_epoch > 0 and best_model_path is not None and os.path.isfile(best_model_path):
        code_snapshot_path, past_best_aupr = _save_global_best_code_snapshot(
            save_prefix=save_prefix,
            best_model_path=best_model_path,
            epoch=best_epoch,
            aupr=best_aupr,
            args=args,
        )
        if code_snapshot_path is not None:
            log(
                f"[GLOBAL BEST] best epoch {best_epoch}: val AUPR={best_aupr:.6f} beat past AUPR={past_best_aupr:.6f} -> code {code_snapshot_path}",
                file=output,
            )
        else:
            log(
                f"[GLOBAL BEST] no code snapshot; run best AUPR={best_aupr:.6f}, past global AUPR={past_best_aupr:.6f}",
                file=output,
            )
        output.flush()

    total_runtime = time.perf_counter() - overall_start_time
    log(
        f"Training summary timing: runtime={_format_duration(total_runtime)}, total_inference={_format_duration(total_inference_time)}",
        file=output,
    )

    if args.log_wandb:
        if save_prefix is not None and os.path.isfile(best_model_path):
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
    if not 0.0 <= args.soft_target_lambda <= 1.0:
        raise ValueError("--soft-target-lambda must be between 0 and 1")
    if not 0.0 <= args.soft_target_ema_decay < 1.0:
        raise ValueError("--soft-target-ema-decay must be in [0, 1)")

    objective = EMASoftTargetObjective(
        soft_target_lambda=args.soft_target_lambda,
        ema_decay=args.soft_target_ema_decay,
        positive_only=args.soft_target_positive_only,
        negative_center_loss_weight=args.negative_center_loss_weight,
        negative_center_loss_min_count=args.negative_center_loss_min_count,
        mixup_weight=args.mixup_weight,
        mixup_lam_min=args.mixup_lam_min,
        mixup_lam_max=args.mixup_lam_max,
    )
    train_model(args, output, interaction_grad_fn=objective)

    output.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
