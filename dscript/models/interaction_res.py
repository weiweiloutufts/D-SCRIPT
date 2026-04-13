import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .contact import ContactCNN
from .embedding import FullyConnectedEmbed

from dataclasses import dataclass
import torch.nn.functional as F


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
    ):
        super().__init__()
        self.use_input_norm      = bool(use_input_norm)
        self.clamp_in            = clamp_in
        self.grid                = int(grid)

        seq_dim  = d_model * 2
        proj_dim = g_fused_dim
        cond_hid = max(64, g_fused_dim // 2)

        self.in_norm = nn.GroupNorm(1, 1, eps=1e-6, affine=True)
        self.map_global_channels = 8

        self.map_stage1 = nn.Sequential(
            nn.Conv2d(1 + self.map_global_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )
        self.map_skip = nn.Conv2d(
            1 + self.map_global_channels, d_model, kernel_size=1, bias=False
        )

        self.map_g_proj = nn.Sequential(
            nn.Linear(g_fused_dim, cond_hid),
            nn.GELU(),
            nn.LayerNorm(cond_hid),
            nn.Linear(cond_hid, self.map_global_channels),
        )

        self.g_inject = nn.Sequential(
            nn.Linear(g_fused_dim, cond_hid),
            nn.GELU(),
            nn.LayerNorm(cond_hid),
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
        nn.init.zeros_(self.map_g_proj[-1].weight)
        nn.init.zeros_(self.map_g_proj[-1].bias)
        nn.init.zeros_(self.map_skip.weight)

    # -------------------------------------------------------------------------

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
        _, T, D = tokens.shape
        if T != height * width:
            raise ValueError(f"Expected T==height*width, got T={T}, H={height}, W={width}")

        row_dim = D // 2
        col_dim = D - row_dim
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
        if rope_dim < D:
            rope_parts.append(tokens[..., rope_dim:])

        return torch.cat(rope_parts, dim=-1)
    # -------------------------------------------------------------------------

    def _extract_map_summary(self, map_input: torch.Tensor, g_fused: torch.Tensor):
        g = self.grid
        B, _, H, W = map_input.shape
        g_map = self.map_g_proj(g_fused)[:, :, None, None].expand(
            B, self.map_global_channels, H, W
        )
        g_scale = 0.1
        map_cat = torch.cat([map_input, g_scale * g_map], dim=1)
        x_pool = F.adaptive_avg_pool2d(map_cat, (g, g))             # [B, 1+K, g, g]
        map_feat = self.map_stage1(x_pool)                          # [B, D, g, g]
        map_feat = map_feat + self.map_skip(x_pool)                 # [B, D, g, g]

        B, D, _, _ = map_feat.shape
        tokens = map_feat.flatten(2).transpose(1, 2)                # [B, g*g, D]
        tokens = self._apply_grid_rope(tokens, g, g)
        map_feat = tokens.transpose(1, 2).reshape(B, D, g, g)       # [B, D, g, g]

        g_ctx = self.g_inject(g_fused)[:, :, None, None]            # [B, D, 1, 1]
        map_feat = F.gelu(map_feat + g_ctx)
        map_feat = map_feat + self.map_stage2(map_feat)             # [B, D, g, g]

        feat_avg = F.adaptive_avg_pool2d(map_feat, 1).flatten(1)    # [B, D]
        feat_max = F.adaptive_max_pool2d(map_feat, 1).flatten(1)    # [B, D]
        feat = torch.cat([feat_avg, feat_max], dim=1)               # [B, 2D]
        feat_proj = self.feat_proj(feat)                            # [B, proj_dim]
        return feat, feat_proj, map_feat

    def extract_feat(self, yhat: torch.Tensor, g_fused: torch.Tensor):
        x = self.in_norm(yhat) if self.use_input_norm else yhat
        if self.clamp_in is not None:
            x = x.clamp(-float(self.clamp_in), float(self.clamp_in))

        feat, feat_proj, stage_map = self._extract_map_summary(x, g_fused)

        assert feat_proj.shape == g_fused.shape, \
            f"feat_proj {feat_proj.shape} vs g_fused {g_fused.shape}"

        feat = torch.cat(
            [feat, g_fused, feat_proj + g_fused, feat_proj * g_fused], dim=1
        )

        return {
            "feat": feat,
            "stage_map": stage_map,
        }

    # -------------------------------------------------------------------------

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
        :param classifier_d_model: hidden width for PairClassifierTokens [default: 64]
        :type classifier_d_model: int
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
        self.classifier_d_model = classifier_d_model
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
        h = 16
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

        self.ff0 = ResidualFFN(d, dropout)
        self.ff1 = ResidualFFN(d, dropout)

        # attention pooling heads
        self.pool0 = nn.Linear(d, 1)
        self.pool1 = nn.Linear(d, 1)

        self.clf = PairClassifierTokens(
            d_model=self.classifier_d_model,
            grid=grid_size,
            p_drop=dropout,
            use_input_norm=True,
            clamp_in=5.0,
            g_fused_dim=in_dim,
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

        g = torch.cat([g_add, g_mul, int_abs, int_sub], dim=1)  # [B,4D]
        logit, feat, stage_map = self.clf(yhat, g)

        return stage_map, logit, feat

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
        classifier_d_model=64,
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
            classifier_d_model=classifier_d_model,
            do_w=do_w,
            do_sigmoid=do_sigmoid,
            do_pool=do_pool,
            pool_size=pool_size,
            theta_init=theta_init,
            lambda_init=lambda_init,
            gamma_init=gamma_init,
        )
