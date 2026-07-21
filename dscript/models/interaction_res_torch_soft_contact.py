"""
Symmetric residual interaction model with torch contact-map attention and
pre-contact positional enrichment.

This variant uses the residual contact-map classifier but enforces pair-order
invariance inside a single forward pass:

* both proteins are processed by the same sequence modules;
* global pair features are symmetric only;
* the map classifier is applied to both map orientations with shared weights.

The sequence self-attention block uses standard nn.MultiheadAttention.

The contact module adds position channels to the hidden pair tensor C before
torch MultiheadAttention and the contact CNN. The downstream map classifier does
not add grid RoPE, so location information is injected in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from .contact_torch import ContactCNN
from .embedding import FullyConnectedEmbed


@dataclass
class InteractionInputs:
    z0: torch.Tensor
    z1: torch.Tensor

    f0: torch.Tensor = None
    f1: torch.Tensor = None

    b0: torch.Tensor = None
    b1: torch.Tensor = None

    embed_foldseek: bool = False
    embed_backbone: bool = False

    def __post_init__(self):
        if self.embed_foldseek:
            assert self.f0 is not None and self.f1 is not None
            assert isinstance(self.f0, torch.Tensor) and isinstance(self.f1, torch.Tensor)
            assert (
                self.z0.get_device() == self.f0.get_device()
                and self.z0.get_device() == self.f1.get_device()
            )
            assert (
                self.f0.shape[1] == self.z0.shape[1]
                and self.f1.shape[1] == self.z1.shape[1]
            )
        if self.embed_backbone:
            assert self.b0 is not None and self.b1 is not None
            assert isinstance(self.b0, torch.Tensor) and isinstance(self.b1, torch.Tensor)
            assert (
                self.z0.get_device() == self.b0.get_device()
                and self.z0.get_device() == self.b1.get_device()
            )
            assert (
                self.b0.shape[1] == self.z0.shape[1]
                and self.b1.shape[1] == self.z1.shape[1]
            )


class LogisticActivation(nn.Module):
    def __init__(self, x0=0, k=1, train=False):
        super().__init__()
        self.x0 = x0
        self.k = nn.Parameter(torch.FloatTensor([float(k)]))
        self.k.requires_grad = train

    def forward(self, x):
        return torch.clamp(1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1)

    def clip(self):
        self.k.data.clamp_(min=0)


class ResBlock(nn.Module):
    def __init__(self, d: int, p_drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(p_drop),
        )

    def forward(self, x):
        return x + self.net(x)


class ResidualFFN(nn.Module):
    def __init__(self, d: int, p_drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(4 * d, d),
            nn.Dropout(p_drop),
        )

    def forward(self, x):
        return self.net(x)


class PairClassifierTokens(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        grid: int = 16,
        p_drop: float = 0.5,
        use_input_norm: bool = True,
        clamp_in: float | None = 5.0,
        g_fused_dim: int = 256,
        map_in_channels: int = 8,
    ):
        super().__init__()
        self.use_input_norm = bool(use_input_norm)
        self.clamp_in = clamp_in
        self.grid = int(grid)

        seq_dim = d_model * 2
        proj_dim = g_fused_dim
        cond_hid = max(64, g_fused_dim // 2)

        # map_in_channels matches ContactCNN.out_channels — the contact map
        # arrives with this many channels, so in_norm and the conv layers are
        # sized accordingly. map_global_channels are added on top via map_g_proj.
        self.map_in_channels  = int(map_in_channels)
        self.map_global_channels = 8
        map_total_channels = self.map_in_channels + self.map_global_channels

        self.in_norm = nn.GroupNorm(
            num_groups=self.map_in_channels,
            num_channels=self.map_in_channels,
            eps=1e-6,
            affine=True,
        )

        self.map_stage1 = nn.Sequential(
            nn.Conv2d(map_total_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )
        self.map_skip = nn.Conv2d(
            map_total_channels, d_model, kernel_size=1, bias=False
        )

        self.map_g_proj = nn.Sequential(
            nn.Linear(g_fused_dim, cond_hid),
            nn.GELU(),
            nn.LayerNorm(cond_hid),
            nn.Dropout(p_drop),
            nn.Linear(cond_hid, self.map_global_channels),
        )

        self.g_inject = nn.Sequential(
            nn.Linear(g_fused_dim, cond_hid),
            nn.GELU(),
            nn.LayerNorm(cond_hid),
            nn.Dropout(p_drop),
            nn.Linear(cond_hid, d_model),
        )

        self.map_stage2 = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )

        self.feat_proj = nn.Linear(seq_dim, proj_dim)

        clf_in = seq_dim + proj_dim * 3
        self.clf_head = nn.Sequential(
            nn.Linear(clf_in, d_model * 4),
            nn.LayerNorm(d_model * 4),
            nn.GELU(),
            nn.Dropout(p_drop),
            ResBlock(d_model * 4, p_drop),
            nn.Linear(d_model * 4, 1),
        )
        nn.init.normal_(self.clf_head[-1].weight, std=0.01)
        nn.init.constant_(self.clf_head[-1].bias, 0.0)
        nn.init.zeros_(self.map_g_proj[-1].weight)
        nn.init.zeros_(self.map_g_proj[-1].bias)
        nn.init.zeros_(self.map_skip.weight)

    def _apply_1d_rope(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        dim = x.shape[-1]
        if dim % 2 != 0:
            raise ValueError(f"RoPE expects an even feature dim, got {dim}")

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim, device=x.device, dtype=torch.float32)
        inv_freq = 10000.0 ** (-2.0 * freq_seq / dim)
        angles = coords.to(dtype=torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
        sin = angles.sin().unsqueeze(0).to(dtype=x.dtype)
        cos = angles.cos().unsqueeze(0).to(dtype=x.dtype)

        x_pair = x.reshape(*x.shape[:-1], half_dim, 2)
        x0 = x_pair[..., 0]
        x1 = x_pair[..., 1]

        out = torch.empty_like(x_pair)
        out[..., 0] = x0 * cos - x1 * sin
        out[..., 1] = x0 * sin + x1 * cos
        return out.flatten(start_dim=-2)

    def _apply_grid_rope(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        _, t, d = tokens.shape
        if t != height * width:
            raise ValueError(f"Expected T==height*width, got T={t}, H={height}, W={width}")

        row_dim = d // 2
        col_dim = d - row_dim
        row_dim -= row_dim % 2
        col_dim -= col_dim % 2
        rope_dim = row_dim + col_dim

        rows = torch.arange(height, device=tokens.device).repeat_interleave(width)
        cols = torch.arange(width, device=tokens.device).repeat(height)

        rope_parts = []
        if row_dim > 0:
            rope_parts.append(self._apply_1d_rope(tokens[..., :row_dim], rows))
        if col_dim > 0:
            start = row_dim
            rope_parts.append(self._apply_1d_rope(tokens[..., start:start + col_dim], cols))
        if rope_dim < d:
            rope_parts.append(tokens[..., rope_dim:])

        return torch.cat(rope_parts, dim=-1)

    def _extract_map_summary(self, map_input: torch.Tensor, g_fused: torch.Tensor):
        g = self.grid
        batch_size = map_input.shape[0]
        x_pool = F.adaptive_avg_pool2d(map_input, (g, g))
        g_map = self.map_g_proj(g_fused)[:, :, None, None].expand(
            batch_size, self.map_global_channels, g, g
        )
        x_pool = torch.cat([x_pool, 0.5 * g_map], dim=1)
        map_feat = self.map_stage1(x_pool)
        map_feat = map_feat + self.map_skip(x_pool)

        batch_size, channels, _, _ = map_feat.shape
        g_ctx = self.g_inject(g_fused)[:, :, None, None]
        map_feat = F.gelu(map_feat + g_ctx)
        map_feat = map_feat + self.map_stage2(map_feat)

        feat_avg = F.adaptive_avg_pool2d(map_feat, 1).flatten(1)
        feat_max = F.adaptive_max_pool2d(map_feat, 1).flatten(1)
        feat = torch.cat([feat_avg, feat_max], dim=1)
        feat_proj = self.feat_proj(feat)
        return feat, feat_proj, map_feat

    def extract_feat(self, yhat: torch.Tensor, g_fused: torch.Tensor):
        x = self.in_norm(yhat) if self.use_input_norm else yhat
        if self.clamp_in is not None:
            x = x.clamp(-float(self.clamp_in), float(self.clamp_in))

        feat, feat_proj, stage_map = self._extract_map_summary(x, g_fused)

        assert feat_proj.shape == g_fused.shape, (
            f"feat_proj {feat_proj.shape} vs g_fused {g_fused.shape}"
        )

        feat = torch.cat(
            [feat, g_fused, feat_proj + g_fused, feat_proj * g_fused], dim=1
        )

        return {
            "feat": feat,
            "stage_map": stage_map,
        }

    def classify_from_feat(self, feat: torch.Tensor) -> torch.Tensor:
        return self.clf_head(feat)

    def forward(self, yhat: torch.Tensor, g: torch.Tensor):
        aux = self.extract_feat(yhat, g)
        logits = self.classify_from_feat(aux["feat"])
        return logits, aux["feat"], aux["stage_map"]


class ModelInteraction(nn.Module):
    def __init__(
        self,
        embedding,
        contact,
        use_cuda,
        dropout,
        grid_size=100,
        classifier_d_model=64,
        do_w=True,
        do_sigmoid=True,
        do_pool=False,
        pool_size=9,
        theta_init=1,
        lambda_init=0,
        gamma_init=0,
        noise_std=0.05,
        map_out_channels=8,
    ):
        super().__init__()
        self.use_cuda = use_cuda
        self.do_w = do_w
        self.do_sigmoid = do_sigmoid
        self.classifier_d_model = classifier_d_model
        self.map_out_channels = int(map_out_channels)
        # noise_std retained in signature for backward compatibility but is no
        # longer forwarded to _encode_sequence (which uses standard attention).
        if do_sigmoid:
            self.activation = LogisticActivation(x0=0.5, k=20)

        self.embedding = embedding
        self.contact = contact

        if self.do_w:
            self.theta = nn.Parameter(torch.FloatTensor([theta_init]))
            self.lambda_ = nn.Parameter(torch.FloatTensor([lambda_init]))

        self.do_pool = do_pool
        self.pool_size = pool_size
        self.maxPool = nn.MaxPool2d(pool_size, padding=pool_size // 2)
        self.gamma = nn.Parameter(torch.FloatTensor([gamma_init]))
        self.register_buffer("xx", torch.arange(2000))

        d_proj = self.embedding.nout
        h = 16
        # Capture short-range residue patterns before recurrent sequence
        # encoding. Padding preserves the original sequence length.
        self.seq_cnn = nn.Sequential(
            nn.Conv1d(d_proj, d_proj, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.seq_bilstm = nn.LSTM(
            input_size=d_proj,
            hidden_size=h,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        d = 2 * h
        heads = 1
        # Standard dot-product multi-head self-attention for sequence encoding.
        self.sa = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.sa_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(d)
        self.ln_2 = nn.LayerNorm(d)
        self.ff = ResidualFFN(d, dropout)
        self.pool = nn.Linear(d, 1)
        self.attn_dropout = nn.Dropout(dropout)
        self.pool_dropout = nn.Dropout(dropout)

        # Symmetric global features only: p0+p1, p0*p1, abs(p0-p1).
        # map_in_channels matches contact.out_channels so the contact map
        # channels flow directly into PairClassifierTokens without expansion.
        self.clf = PairClassifierTokens(
            d_model=self.classifier_d_model,
            grid=grid_size,
            p_drop=dropout,
            use_input_norm=True,
            clamp_in=5.0,
            g_fused_dim=3 * d,
            map_in_channels=self.map_out_channels,
        )
        self.clip()

    def clip(self):
        self.contact.clip()
        with torch.no_grad():
            if self.do_w:
                self.theta.clamp_(0, 1)
                self.lambda_.clamp_(min=0)
            self.gamma.clamp_(0, 3)

    def embed(self, x):
        if self.embedding is None:
            return x
        x = x.to(dtype=next(self.embedding.parameters()).dtype)
        return self.embedding(x)

    def _encode_sequence(self, seq_e):
        # Conv1d uses [batch, channels, length], whereas the embedding and
        # BiLSTM use [batch, length, channels].
        seq_e = self.seq_cnn(seq_e.transpose(1, 2)).transpose(1, 2)
        h, _ = self.seq_bilstm(seq_e)
        # Standard multi-head self-attention (nn.MultiheadAttention).
        # need_weights=False skips the attention weight averaging for speed.
        attn, _ = self.sa(h, h, h, need_weights=False)
        x = self.ln_1(h + self.attn_dropout(attn))
        x = self.ln_2(x + self.ff(x))
        # FIX 6: pool_dropout applied AFTER softmax to avoid distorting the
        # attention distribution before normalisation.
        weights = self.pool_dropout(
            torch.softmax(self.pool(x), dim=1)
        )
        return (weights * x).sum(dim=1)

    def cpred(self, inputs):
        e0 = self.embed(inputs.z0)
        e1 = self.embed(inputs.z1)
        seq_e0 = e0
        seq_e1 = e1

        if inputs.embed_foldseek:
            e0 = torch.concat([e0, inputs.f0], dim=2)
            e1 = torch.concat([e1, inputs.f1], dim=2)

        if inputs.embed_backbone:
            e0 = torch.concat([e0, inputs.b0], dim=2)
            e1 = torch.concat([e1, inputs.b1], dim=2)

        if e0.shape[-1] != e1.shape[-1]:
            raise ValueError(
                "Contact embedding dim mismatch: "
                f"{tuple(e0.shape)} vs {tuple(e1.shape)}"
            )
        c_map = self.contact.predict(self.contact.cmap(e0, e1))

        p0 = self._encode_sequence(seq_e0)
        p1 = self._encode_sequence(seq_e1)

        g_add = p0 + p1
        g_mul = p0 * p1
        g_abs = (p0 - p1).abs()
        return c_map, g_add, g_mul, g_abs

    def _build_interaction_inputs(
        self,
        z0,
        z1,
        embed_foldseek=False,
        f0=None,
        f1=None,
        embed_backbone=False,
        b0=None,
        b1=None,
    ):
        return InteractionInputs(
            z0,
            z1,
            embed_foldseek=embed_foldseek,
            f0=f0,
            f1=f1,
            embed_backbone=embed_backbone,
            b0=b0,
            b1=b1,
        )

    def _apply_weight(self, c_map):
        if not self.do_w:
            return c_map

        n, m = c_map.shape[2:]
        device = c_map.device
        xx_n = torch.arange(n, device=device, dtype=c_map.dtype)
        xx_m = torch.arange(m, device=device, dtype=c_map.dtype)

        x1 = -1 * torch.square((xx_n + 1 - ((n + 1) / 2)) / (-1 * ((n + 1) / 2)))
        x2 = -1 * torch.square((xx_m + 1 - ((m + 1) / 2)) / (-1 * ((m + 1) / 2)))
        x1 = torch.exp(self.lambda_ * x1)
        x2 = torch.exp(self.lambda_ * x2)

        w = x1.unsqueeze(1) * x2
        w = (1 - self.theta) * w + self.theta
        return c_map * w

    def map_predict(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], InteractionInputs):
            cpred_inputs = args[0]
        elif len(args) >= 2:
            cpred_inputs = self._build_interaction_inputs(*args, **kwargs)
        else:
            raise TypeError("map_predict expects InteractionInputs or z0 and z1")

        c_map, g_add, g_mul, g_abs = self.cpred(cpred_inputs)
        g = torch.cat([g_add, g_mul, g_abs], dim=1)
        yhat = self._apply_weight(c_map)

        logit, feat, stage_map = self.clf(yhat, g)
        return stage_map, logit, feat

    def predict(
        self,
        z0,
        z1,
        embed_foldseek=False,
        f0=None,
        f1=None,
        embed_backbone=False,
        b0=None,
        b1=None,
    ):
        _, phat, _ = self.map_predict(
            z0,
            z1,
            embed_foldseek=embed_foldseek,
            f0=f0,
            f1=f1,
            embed_backbone=embed_backbone,
            b0=b0,
            b1=b1,
        )
        return phat

    def forward(
        self,
        z0,
        z1,
        embed_foldseek=False,
        f0=None,
        f1=None,
        embed_backbone=False,
        b0=None,
        b1=None,
    ):
        return self.predict(
            z0,
            z1,
            embed_foldseek=embed_foldseek,
            f0=f0,
            f1=f1,
            embed_backbone=embed_backbone,
            b0=b0,
            b1=b1,
        )


class DSCRIPTModel(ModelInteraction, PyTorchModelHubMixin):
    def __init__(
        self,
        emb_nin,
        emb_nout,
        emb_dropout,
        con_embed_dim,
        con_hidden_dim,
        con_width,
        use_cuda,
        dropout_p,
        emb_activation=nn.ReLU(),
        con_activation=nn.Sigmoid(),
        do_w=True,
        do_sigmoid=True,
        do_pool=False,
        pool_size=9,
        classifier_d_model=64,
        theta_init=1,
        lambda_init=0,
        gamma_init=0,
        noise_std=0.05,
        map_out_channels=8,
    ):
        embedding = FullyConnectedEmbed(emb_nin, emb_nout, emb_dropout, emb_activation)
        # out_channels matches map_out_channels so ContactCNN produces the right
        # number of feature maps for PairClassifierTokens directly.
        contact = ContactCNN(
            con_embed_dim,
            con_hidden_dim,
            con_width,
            out_channels=map_out_channels,
            activation=con_activation,
        )
        super().__init__(
            embedding=embedding,
            contact=contact,
            use_cuda=use_cuda,
            dropout=dropout_p,
            classifier_d_model=classifier_d_model,
            do_w=do_w,
            do_sigmoid=do_sigmoid,
            do_pool=do_pool,
            pool_size=pool_size,
            theta_init=theta_init,
            lambda_init=lambda_init,
            gamma_init=gamma_init,
            noise_std=noise_std,
            map_out_channels=map_out_channels,
        )
