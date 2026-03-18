import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .contact import ContactCNN
from .embedding import FullyConnectedEmbed

from dataclasses import dataclass
import torch.nn.functional as F
import math

import sys

import matplotlib.pyplot as plt
from typing import Optional


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
            assert isinstance(self.f0, torch.Tensor) and isinstance(
                self.f1, torch.Tensor
            )
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
            assert isinstance(self.b0, torch.Tensor) and isinstance(
                self.b1, torch.Tensor
            )
            assert (
                self.z0.get_device() == self.b0.get_device()
                and self.z0.get_device() == self.b1.get_device()
            )
            assert (
                self.b0.shape[1] == self.z0.shape[1]
                and self.b1.shape[1] == self.z1.shape[1]
            )


class LogisticActivation(nn.Module):
    """
    Implementation of Generalized Sigmoid
    Applies the element-wise function:

    :math:`\\sigma(x) = \\frac{1}{1 + \\exp(-k(x-x_0))}`

    :param x0: The value of the sigmoid midpoint
    :type x0: float
    :param k: The slope of the sigmoid - trainable -  :math:`k \\geq 0`
    :type k: float
    :param train: Whether :math:`k` is a trainable parameter
    :type train: bool
    """

    def __init__(self, x0=0, k=1, train=False):
        super().__init__()
        self.x0 = x0
        self.k = nn.Parameter(torch.FloatTensor([float(k)]))
        self.k.requires_grad = train

    def forward(self, x):
        """
        Applies the function to the input elementwise

        :param x: :math:`(N \\times *)` where :math:`*` means, any number of additional dimensions
        :type x: torch.Tensor
        :return: :math:`(N \\times *)`, same shape as the input
        :rtype: torch.Tensor
        """
        o = torch.clamp(1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1)
        return o

    def clip(self):
        """
        Restricts sigmoid slope :math:`k` to be greater than or equal to 0, if :math:`k` is trained.

        :meta private:
        """
        self.k.data.clamp_(min=0)


class ContactPatchTokenizer(nn.Module):
    """
    Conv patchify -> feature map (h,w) -> adaptive pool to (grid,grid) -> tokens [B,T,D]
    Also returns x_pool_1ch [B,1,grid,grid] for global pooling (ph).

    """

    def __init__(self, d_model=64, patch=8, stride=4, grid=16, noise_std=0.02):
        super().__init__()
        self.grid = int(grid)
        self.noise_std = float(noise_std)
        self.patch_k = int(patch)
        self.patch_s = int(stride)

        # bias=True helps gradient flow early in training
        self.patch = nn.Conv2d(
            1, d_model, kernel_size=self.patch_k, stride=self.patch_s, bias=True
        )
        self.norm = nn.GroupNorm(1, d_model, eps=1e-6, affine=True)
        self.act = nn.SiLU()

        # small residual refine block after pooling — improves gradient flow
        # through the avg pool bottleneck
        self.refine = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, d_model, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
        )

        # PPM fusion norm
        self.ppm_gn = nn.GroupNorm(8, d_model)

    def _ppm_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pyramid pooling: fuse 1x1, g//3, g//2, g scales → [B, D, g, g]"""
        g  = self.grid
        g2 = max(1, g // 3)
        g3 = max(1, g // 2)

        p1 = F.adaptive_avg_pool2d(x, (1,  1 ))  # global context
        p2 = F.adaptive_avg_pool2d(x, (g2, g2))  # coarse
        p3 = F.adaptive_avg_pool2d(x, (g3, g3))  # medium
        p4 = F.adaptive_avg_pool2d(x, (g,  g ))  # target scale

        # upsample all to g,g
        up = lambda t: F.interpolate(t, size=(g, g), mode='bilinear', align_corners=False)
        return self.ppm_gn(up(p1) + up(p2) + up(p3) + p4)              # [B, D, g, g]

    def forward(self, x: torch.Tensor):
        # x: [B,1,H,W]
        x = self.act(self.norm(self.patch(x)))  # [B,D,h,w]
        if self.grid < 32:
            # intermediate pool to stable mid-scale before PPM
            x_mid  = F.adaptive_avg_pool2d(x, (32, 32))                    # [B, D, 32, 32]
        else:
            x_mid = x

        x_mid  = x_mid + self.refine(x_mid)  
        
        # PPM → fixed grid
        x_pool = self._ppm_pool(x_mid)                                  # [B, D, g, g]

        # tokens
        tokens = x_pool.flatten(2).transpose(1, 2)                      # [B, T, D] T=g*g

        if self.training and self.noise_std > 0:
            rms = tokens.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            tokens = tokens + torch.randn_like(tokens) * (self.noise_std * rms)

        # mean over channels — no extra parameters, cleaner gradient
        x_pool_d = x_pool # [B,D,g,g]

        return tokens, x_pool_d

        
# Group Normalization with 64 groups
class GN64(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.gn = nn.GroupNorm(64, num_channels)
    
    def forward(self, x):
        return self.gn(x)


class ResBlock(nn.Module):
    def __init__(self, d: int, p_drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(p_drop),
        )
    def forward(self, x): return x + self.net(x)


class PairClassifierTokens(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        patch: int = 8,
        stride: int = 4,
        grid: int = 16,
        p_drop: float = 0.2,
        use_input_norm: bool = True,
        clamp_in: float | None = 5.0,
        noise_std: float = 0.1,
        init_tau_ph: float = 5.0,
        init_tau_tok: float = 1.5,
        init_pos_scale: float = 0.05,
        target_tok_ent_frac: float = 0.9,
        min_tau_tok: float = 1.3,
        max_score_scale: float = 6.0,
        g_fused_dim: int = 256,
    ):
        super().__init__()
        self.use_input_norm      = bool(use_input_norm)
        self.clamp_in            = clamp_in
        self.grid                = int(grid)
        self.target_tok_ent_frac = float(target_tok_ent_frac)
        self.min_tau_tok         = float(min_tau_tok)
        self.max_score_scale     = float(max_score_scale)

        seq_dim  = d_model * 6
        proj_dim = g_fused_dim

        self.in_norm = nn.GroupNorm(1, 1, eps=1e-6, affine=True)

        self.tokenizer = ContactPatchTokenizer(
            d_model=d_model, patch=patch, stride=stride,
            grid=grid, noise_std=noise_std,
        )

        self.pos_row   = nn.Embedding(self.grid, d_model)
        self.pos_col   = nn.Embedding(self.grid, d_model)
        self.pos_drop  = nn.Dropout(p_drop)
        self.pos_scale = nn.Parameter(torch.tensor(float(init_pos_scale)))

        self.log_tau_ph  = nn.Parameter(torch.tensor(float(init_tau_ph)).log())
        self.log_tau_tok = nn.Parameter(torch.tensor(float(init_tau_tok)).log())
        self.score_scale = nn.Parameter(torch.tensor(3.0))   # start with signal, not 1.0

        # token scoring — hidden layer, no redundant LayerNorm
        self.token_score = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.GELU(),
            nn.Linear(d_model // 2, 1, bias=True),
        )
        # zero-init final layer — sharpens gradually as score_scale amplifies
        nn.init.zeros_(self.token_score[-1].weight)
        nn.init.zeros_(self.token_score[-1].bias)

        # ph head
        self.ph_head = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        for m in self.ph_head:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.1)
                nn.init.constant_(m.bias, 0.0)

        self.ph_gate = nn.Linear(2, 1, bias=True)
        nn.init.constant_(self.ph_gate.weight, 0.0)
        nn.init.constant_(self.ph_gate.bias,   0.0)
        with torch.no_grad():
            self.ph_gate.weight[0, 0] = 1.0
            self.ph_gate.weight[0, 1] = 0.1

        # grid — GroupNorm after each conv
        self.grid_post_gn = nn.GroupNorm(8, d_model)
        self.grid_refine  = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1, bias=False),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1, bias=False),
            nn.GroupNorm(8, d_model),
        )

        # cross attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # clf_attn — slim 2-layer scorer
        self.clf_attn = nn.Sequential(
            nn.Linear(seq_dim, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # feat_proj tied to g_fused_dim
        self.feat_proj = nn.Linear(seq_dim, proj_dim)

        # clf_head with residual block
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

    # -------------------------------------------------------------------------

    def _compute_ph(self, x_1ch: torch.Tensor) -> torch.Tensor:
        tau  = self.log_tau_ph.exp().clamp(0.8, 6.0)
        x    = (x_1ch * tau).clamp(-50, 50)
        K    = x.shape[-2] * x.shape[-1]
        phat = (torch.logsumexp(x, dim=(1, 2, 3)) - math.log(K)) / tau
        return phat.unsqueeze(1)

    @staticmethod
    def _entropy(p: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
        return -(p * (p + eps).log()).sum(dim=dim)

    def _add_grid_pos(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T, D = tokens.shape
        g = self.tokenizer.grid
        if T != g * g:
            raise ValueError(f"Expected T==g*g, got T={T}, g={g}")

        rows = torch.arange(g, device=tokens.device).repeat_interleave(g)
        cols = torch.arange(g, device=tokens.device).repeat(g)
        pos  = (self.pos_row(rows) + self.pos_col(cols)).unsqueeze(0).to(dtype=tokens.dtype)

        tokens = tokens + torch.tanh(self.pos_scale) * pos
        if self.training:
            tokens = self.pos_drop(tokens)
        return tokens

    # -------------------------------------------------------------------------

    def extract_feat(self, yhat_fused: torch.Tensor, g_fused: torch.Tensor):
        x = self.in_norm(yhat_fused) if self.use_input_norm else yhat_fused
        if self.clamp_in is not None:
            x = x.clamp(-float(self.clamp_in), float(self.clamp_in))

        tokens, x_pool_D = self.tokenizer(x)
        tokens = self._add_grid_pos(tokens)

        B, T, D = tokens.shape
        g   = self.tokenizer.grid
        tau = self.log_tau_tok.exp().clamp(self.min_tau_tok, 5.0)

        # --- token branch ---
        tokens_n = F.layer_norm(tokens, (D,))

        if self.training:
            # noise aug
            tokens_n   = tokens_n + torch.randn_like(tokens_n) * 0.05
            tok_logits = self.token_score(tokens_n)
            scaled     = tok_logits * self.score_scale.abs().clamp(0.5, self.max_score_scale)  # (B, T, 1)

            # ── clean main branch ─────────────────────────────────────────
            token_w  = torch.softmax(scaled / tau, dim=1)
            feat_tok = (tokens_n * token_w).sum(dim=1)              # (B, D)

            # ── masked aug branch ─────────────────────────────────────────
            mask2        = (torch.rand_like(scaled) > 0.25)
            token_w_aug  = torch.softmax(scaled.masked_fill(~mask2, -1e9) / tau, dim=1)
            feat_tok_aug = (tokens_n * token_w_aug).sum(dim=1)      # (B, D)

            # ── JSD + cosine cons_loss ────────────────────────────────────
            w1    = token_w.squeeze(-1).clamp(min=1e-8)
            w2    = token_w_aug.squeeze(-1).clamp(min=1e-8)
            m     = (0.5 * (w1 + w2)).clamp(min=1e-8)
            log_m = m.log()
            kl1   = F.kl_div(log_m, w1, reduction='none').sum(dim=1)
            kl2   = F.kl_div(log_m, w2, reduction='none').sum(dim=1)
            cons_loss = (0.5 * (kl1 + kl2) +
                         0.5 * (1 - F.cosine_similarity(feat_tok, feat_tok_aug, dim=1)))

        else:
            tok_logits   = self.token_score(tokens_n)
            scaled       = tok_logits * self.score_scale.abs().clamp(0.5, self.max_score_scale)
            token_w      = torch.softmax(scaled / tau, dim=1)
            feat_tok     = (tokens_n * token_w).sum(dim=1)
            feat_tok_aug = feat_tok
            cons_loss    = 0.0

        # --- grid branch --- (p1 removed — was identity)
        grid2d = tokens_n.transpose(1, 2).reshape(B, D, g, g)
        grid2d    = self.grid_post_gn(
            F.avg_pool2d(grid2d, 3, stride=1, padding=1) +
            F.max_pool2d(grid2d, 3, stride=1, padding=1)
        )
        grid_feat = grid2d + self.grid_refine(grid2d)               # (B, D, g, g)

        # --- cross attention: grid queries token ---
        q = self.q_proj(grid2d.flatten(2).transpose(1, 2)).unsqueeze(1)  # (B, 1, g*g, D)

        k = self.k_proj(feat_tok).unsqueeze(1).unsqueeze(1)
        v = self.v_proj(feat_tok).unsqueeze(1).unsqueeze(1)
        grid_ctx_2d = F.scaled_dot_product_attention(q, k, v) \
                        .squeeze(1).transpose(1, 2).reshape(B, D, g, g)

        if self.training:
            k_aug = self.k_proj(feat_tok_aug).unsqueeze(1).unsqueeze(1)
            v_aug = self.v_proj(feat_tok_aug).unsqueeze(1).unsqueeze(1)
            grid_ctx_2d_aug = F.scaled_dot_product_attention(q, k_aug, v_aug) \
                                .squeeze(1).transpose(1, 2).reshape(B, D, g, g)
        else:
            grid_ctx_2d_aug = grid_ctx_2d - grid_feat               

        x_pool = x_pool_D.mean(dim=1, keepdim=True)                # [B, 1, g, g]
        # --- feat_seq [B, 6D, g*g] ---
        def to_seq(t): return t.flatten(2)
        
        feat_seq = torch.cat([
            to_seq(grid_feat),                          # spatial structure
            to_seq(x_pool_D),                           # original signal
            to_seq(grid_feat * x_pool_D),               # agreement
            to_seq(torch.abs(grid_feat - x_pool_D)),    # disagreement
            to_seq(grid_ctx_2d),                        # semantically grounded
            to_seq(grid_ctx_2d_aug),                    # aug grounded / residual
        ], dim=1)                                       # (B, 6D, g*g)

        if self.training:
            rms      = feat_seq.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
            feat_seq = feat_seq + torch.randn_like(feat_seq) * (0.05 * rms)
            feat_seq = F.dropout(feat_seq, p=0.1)

        attn_w    = torch.softmax(self.clf_attn(feat_seq.transpose(1, 2)), dim=1)
        feat      = (feat_seq.transpose(1, 2) * attn_w).sum(dim=1) # (B, 6D)
        feat_proj = self.feat_proj(feat)                            # (B, proj_dim)

        assert feat_proj.shape == g_fused.shape, \
            f"feat_proj {feat_proj.shape} vs g_fused {g_fused.shape}"

        feat = torch.cat(
            [feat, g_fused, feat_proj + g_fused, feat_proj * g_fused], dim=1
        )

        tok_ent        = self._entropy(token_w.squeeze(-1), dim=1)
        target_tok_ent = math.log(T) * float(self.target_tok_ent_frac)

        return {
            "feat":                 feat,
            "x_pool":               x_pool,
            "tok_logits":           tok_logits,
            "token_weights":        token_w,
            "feat_tok":             feat_tok,
            "grid_feat":            grid_feat,
            "grid":                 g,
            "token_entropy":        tok_ent,
            "target_token_entropy": target_tok_ent,
            "cons_loss":            cons_loss,
        }

    # -------------------------------------------------------------------------

    def classify_from_feat(self, feat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        logit    = self.clf_head(feat)
        ph_logit = self._compute_ph(x)
        ph       = self.ph_head(ph_logit)
        logit    = self.ph_gate(torch.cat([logit, ph], dim=-1))
        return logit, ph_logit

    def forward(self, yhat_fused: torch.Tensor, g: torch.Tensor):
        aux              = self.extract_feat(yhat_fused, g)
        logits, ph_logit = self.classify_from_feat(aux["feat"], aux["x_pool"])
        aux["ph_logit"]  = ph_logit
        return logits, aux["feat"], aux

class ModelInteraction(nn.Module):
    def __init__(
        self,
        embedding,
        contact,
        use_cuda,
        dropout,
        do_w=True,
        # language_mod_size=25,
        do_sigmoid=True,
        do_pool=False,
        pool_size=9,
        theta_init=1,
        lambda_init=0,
        gamma_init=0,
    ):
        """
        Main D-SCRIPT model. Contains an embedding and contact model and offers access to those models. Computes pooling operations on contact map to generate interaction probability.

        :param embedding: Embedding model
        :type embedding: dscript.models.embedding.FullyConnectedEmbed
        :param contact: Contact model
        :type contact: dscript.models.contact.ContactCNN
        :param use_cuda: Whether the model should be run on GPU
        :type use_cuda: bool
        :param do_w: whether to use the weighting matrix [default: True]
        :type do_w: bool
        :param do_sigmoid: whether to use a final sigmoid activation [default: True]
        :type do_sigmoid: bool
        :param do_pool: whether to do a local max-pool prior to the global pool
        :type do_pool: bool
        :param pool_size: width of max-pool [default 9]
        :type pool_size: bool
        :param theta_init: initialization value of :math:`\\theta` for weight matrix [default: 1]
        :type theta_init: float
        :param lambda_init: initialization value of :math:`\\lambda` for weight matrix [default: 0]
        :type lambda_init: float
        :param gamma_init: initialization value of :math:`\\gamma` for global pooling [default: 0]
        :type gamma_init: float

        """
        super().__init__()
        self.use_cuda = use_cuda
        self.do_w = do_w
        self.do_sigmoid = do_sigmoid
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

        self.clip()

        self.register_buffer("xx", torch.arange(2000))

        ## added aug
        k = 8
        ## need to adjust after set the dims of foldseek and backbone embedding
        D = self.embedding.nout  # = 100
        h = 32
        self.seq_bilstm = nn.LSTM(
            input_size=D,
            hidden_size=h,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        d = 2 * h  # global vector dim after BiLSTM
        in_dim = 4 * d

        heads = 1  # must divide d; use 2 or 1 if needed

        self.sa0 = nn.MultiheadAttention(embed_dim=d, num_heads=heads, batch_first=True)
        self.sa1 = nn.MultiheadAttention(embed_dim=d, num_heads=heads, batch_first=True)

        self.ln0_1 = nn.LayerNorm(d)
        self.ln0_2 = nn.LayerNorm(d)
        self.ln1_1 = nn.LayerNorm(d)
        self.ln1_2 = nn.LayerNorm(d)

        self.ff0 = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        self.ff1 = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))

        # attention pooling heads
        self.pool0 = nn.Linear(d, 1)
        self.pool1 = nn.Linear(d, 1)

        # prepare for the cls
        hid = max(64, in_dim // 2)

        self.g_proj = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.LayerNorm(hid),
            nn.Linear(hid, k),
        )

        self.log_gk_scale = nn.Parameter(torch.log(torch.tensor(0.1)))

        in_ch = 2 + k
        mid = 32
        self.yhat_fuse = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=1, bias=False),
            nn.GroupNorm(1, mid),  # stable for batch=1
            nn.GELU(),
            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.yhat_fuse[-1].weight)
        nn.init.zeros_(self.yhat_fuse[-1].bias)

        self.clf = PairClassifierTokens(
            d_model=128,
            patch=8,
            stride=4,
            grid=8,
            p_drop=dropout,
            use_input_norm=True,
            clamp_in=5.0,
            g_fused_dim = in_dim,
        )

    def clip(self):
        """
        Clamp model values

        :meta private:
        """
        self.contact.clip()

        with torch.no_grad():
            if self.do_w:
                self.theta.clamp_(0, 1)
                self.lambda_.clamp_(min=0)

            self.gamma.clamp_(0, 3)

    def embed(self, x):
        """
        Project down input language model embeddings into low dimension using projection module

        :param z: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z: torch.Tensor
        :return: D-SCRIPT projection :math:`(b \\times N \\times d)`
        :rtype: torch.Tensor
        """
        if self.embedding is None:
            return x
        else:
            x = x.to(dtype=next(self.embedding.parameters()).dtype)
            return self.embedding(x)

    def cpred(self, inputs):
        """
        Project down input language model embeddings into low dimension using projection module

        :param z0: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z0: torch.Tensor
        :param z1: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z1: torch.Tensor
        :return: Predicted contact map :math:`(b \\times N \\times M)`
        :rtype: torch.Tensor
        """
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

        Bmap = self.contact.cmap(e0, e1)
        C = self.contact.predict(Bmap)
        ###added augment
        # print("e0 shape:", e0.shape)
        # print("e1 shape:", e1.shape)
        # Keep the sequence encoder on projected LM embeddings only so optional
        # structural channels can widen the contact input without breaking the LSTM.
        h0, _ = self.seq_bilstm(seq_e0)  # [1,N,d]
        h1, _ = self.seq_bilstm(seq_e1)  # [1,M,d]

        # --- self-attention block for seq0
        x0_attn, _ = self.sa0(h0, h0, h0, need_weights=False)  # [1,N,d]
        x0 = self.ln0_1(h0 + x0_attn)
        x0_ff = self.ff0(x0)
        x0 = self.ln0_2(x0 + x0_ff)  # [1,N,d]

        # --- self-attention block for seq1
        x1_attn, _ = self.sa1(h1, h1, h1, need_weights=False)  # [1,M,d]
        x1 = self.ln1_1(h1 + x1_attn)
        x1_ff = self.ff1(x1)
        x1 = self.ln1_2(x1 + x1_ff)  # [1,M,d]

        # --- attention pooling (learned weighted sum)
        a0 = self.pool0(x0)  # [1,N,1]
        a1 = self.pool1(x1)  # [1,M,1]
        w0 = torch.softmax(a0, dim=1)  # [1,N,1]
        w1 = torch.softmax(a1, dim=1)  # [1,M,1]
        p0 = (w0 * x0).sum(dim=1)  # [1,d]
        p1 = (w1 * x1).sum(dim=1)  # [1,d]

        int_add = p0 + p1  # [B,d]
        int_mul = p0 * p1  # [B,d]
        int_abs = (p0 - p1).abs()  # [B,d]
        int_sub = p0 - p1  # [B,d]

        # print("int0, int1 shape:", int0.shape, int1.shape)
        ### added return int0, int1
        return C, int_add, int_mul, int_abs, int_sub

    # TODO: Temporaru overload to allow downstream (post train/evaluate) methods to work.
    def _build_interaction_inputs(
        self,
        z0,
        z1,
        ### Foldseek embedding added
        embed_foldseek=False,
        f0=None,
        f1=None,
        ### Backbone embedding added
        embed_backbone=False,
        b0=None,
        b1=None,
    ):
        """
        Project down input language model embeddings into low dimension using projection module

        :param z0: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z0: torch.Tensor
        :param z1: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z1: torch.Tensor
        :return: Predicted contact map, predicted probability of interaction :math:`(b \\times N \\times d_0), (1)`
        :rtype: torch.Tensor, torch.Tensor
        """
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

    def map_predict(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], InteractionInputs):
            cpredInputs = args[0]

        if len(args) >= 2:
            cpredInputs = self._build_interaction_inputs(*args, **kwargs)

        C, g_add, g_mul, int_abs, int_sub = self.cpred(cpredInputs)

        if self.training and not hasattr(self, "_printed_batch_B"):
            self._printed_batch_B = True
            print("[DEBUG map_predict] C shape =", tuple(C.shape))

        # Ensure g_add/g_mul are [B,D]
        if g_add.ndim == 3:  # [B,1,D]
            g_add = g_add.squeeze(1)
            g_mul = g_mul.squeeze(1)
            int_abs = int_abs.squeeze(1)
            int_sub = int_sub.squeeze(1)

        if self.do_w:
            N, M = C.shape[2:]
            device = C.device

            xx_N = torch.arange(N, device=device, dtype=C.dtype)
            xx_M = torch.arange(M, device=device, dtype=C.dtype)

            x1 = -1 * torch.square((xx_N + 1 - ((N + 1) / 2)) / (-1 * ((N + 1) / 2)))

            x2 = -1 * torch.square((xx_M + 1 - ((M + 1) / 2)) / (-1 * ((M + 1) / 2)))

            x1 = torch.exp(self.lambda_ * x1)
            x2 = torch.exp(self.lambda_ * x2)

            W = x1.unsqueeze(1) * x2
            W = (1 - self.theta) * W + self.theta

            yhat = C * W

        else:
            yhat = C

        # ---- fuse global interaction into map (BEFORE pooling is usually better)
        B, _, N, M = yhat.shape

        g = torch.cat([g_add, g_mul, int_abs, int_sub], dim=1)  # [B,4D]
        gk = self.g_proj(g)  # [B,k]

        gk_map = gk[:, :, None, None].expand(B, gk.shape[1], N, M)  # [B,k,N,M]

        gk_scale = torch.sigmoid(self.log_gk_scale) * 0.5  # ∈ (0, 0.5)

        eps = 1e-4
        y_logit = torch.log(yhat.clamp_min(eps)) - torch.log(
            (1 - yhat).clamp_min(eps)
        )  # [B,1,N,M]

        yhat_cat = torch.cat([yhat, y_logit, gk_scale * gk_map], dim=1)

        delta = self.yhat_fuse(yhat_cat)  # [B,1,N,M]
        if self.training:
            delta = delta + 0.02 * torch.randn_like(delta)
        yhat_fused = y_logit + delta

        logit, feat, aux = self.clf(yhat_fused,g)

        return yhat_fused, logit, feat, aux 

    # INTERNAL
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
        """
        Project down input language model embeddings into low dimension using projection module

        :param z0: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z0: torch.Tensor
        :param z1: Language model embedding :math:`(b \\times N \\times d_0)`
        :type z1: torch.Tensor
        :return: Predicted probability of interaction
        :rtype: torch.Tensor, torch.Tensor
        """
        _, phat, _, _ = self.map_predict(
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
        """
        :meta private:
        """
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
        theta_init=1,
        lambda_init=0,
        gamma_init=0,
    ):
        embedding = FullyConnectedEmbed(emb_nin, emb_nout, emb_dropout, emb_activation)
        contact = ContactCNN(con_embed_dim, con_hidden_dim, con_width, con_activation)
        super().__init__(
            embedding=embedding,
            contact=contact,
            use_cuda=use_cuda,
            dropout=dropout_p,
            do_w=do_w,
            do_sigmoid=do_sigmoid,
            do_pool=do_pool,
            pool_size=pool_size,
            theta_init=theta_init,
            lambda_init=lambda_init,
            gamma_init=gamma_init,
        )
