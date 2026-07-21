# Input: C = NxMxH embedding contact matrix
# Output: S = MxN contact prediction matrix

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnected(nn.Module):
    """
    Performs part 1 of Contact Prediction Module. Takes embeddings from
    Projection module and produces broadcast tensor.

    Input embeddings of dimension :math:`d` are combined into a :math:`2d`
    length feature input :math:`z_{cat}`, where
    :math:`z_{cat} = [z_0 \\ominus z_1 | z_0 \\odot z_1]`
    """

    def __init__(self, embed_dim, hidden_dim, activation=nn.ReLU()):
        super().__init__()

        self.D = embed_dim
        self.H = hidden_dim
        self.conv = nn.Conv2d(2 * self.D, self.H, 1)
        self.batchnorm = nn.BatchNorm2d(self.H)
        self.activation = activation

    def forward(self, z0, z1):
        z0 = z0.transpose(1, 2)
        z1 = z1.transpose(1, 2)

        z0 = z0.unsqueeze(3)
        z1 = z1.unsqueeze(2)

        w_dif, w_mul = self.conv.weight.split(self.D, dim=1)

        z_dif = torch.abs(z0 - z1)
        c = F.conv2d(z_dif, w_dif, self.conv.bias)
        del z_dif

        z_mul = z0 * z1
        c = c + F.conv2d(z_mul, w_mul, None)
        del z_mul

        c = self.activation(c)
        c = self.batchnorm(c)

        return c


class ContactMapAttention(nn.Module):
    """Torch self-attention over pooled pairwise positions in contact tensor C.

    This mirrors ``contact_new.ContactMapAttention`` but uses standard
    ``nn.MultiheadAttention`` instead of SpiralMultiheadAttention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        pool_size: int = 16,
        dropout: float = 0.1,
        noise_std: float = 0.05,
        spiral_turns: float | None = None,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.spiral_turns = spiral_turns

        while hidden_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.num_heads = num_heads

        self.sa = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        batch, h, N, M = C.shape
        s = self.pool_size

        C_small = F.adaptive_avg_pool2d(C, (s, s))
        tokens = C_small.flatten(2).transpose(1, 2)

        attn_out, _ = self.sa(tokens, tokens, tokens, need_weights=False)
        tokens = self.ln(tokens + attn_out)
        tokens = self.out_proj(tokens)

        C_delta = tokens.transpose(1, 2).view(batch, h, s, s)
        C_delta = F.interpolate(
            C_delta, size=(N, M), mode="bilinear", align_corners=False
        )
        return C + C_delta


class ContactCNN(nn.Module):
    """Residue Contact Prediction Module with pooled torch attention."""

    def __init__(
        self,
        embed_dim,
        hidden_dim=50,
        width=7,
        out_channels: int = 8,
        activation=nn.Sigmoid(),
        attn_pool_size: int = 16,
        attn_dropout: float = 0.0,
        attn_noise_std: float = 0.05,
        attn_spiral_turns: float | None = None,
    ):
        super().__init__()
        self.out_channels = int(out_channels)

        hidden_dim = embed_dim // 2
        self.hidden = FullyConnected(embed_dim, hidden_dim)

        self.cmap_attn = ContactMapAttention(
            hidden_dim=hidden_dim,
            num_heads=1,
            pool_size=attn_pool_size,
            dropout=attn_dropout,
            noise_std=attn_noise_std,
            spiral_turns=attn_spiral_turns,
        )

        self.conv = nn.Conv2d(hidden_dim, self.out_channels, width, padding=width // 2)
        self.batchnorm = nn.BatchNorm2d(self.out_channels)
        self.activation = activation
        self.clip()

    def clip(self):
        """Force the convolutional layer to be transpose invariant."""
        w = self.conv.weight
        self.conv.weight.data[:] = 0.5 * (w + w.transpose(2, 3))

    def forward(self, z0, z1):
        C = self.cmap(z0, z1)
        return self.predict(C)

    def predict_from_embeddings(self, z0, z1, chunk_size=None):
        if chunk_size is None or chunk_size <= 0 or z0.shape[1] <= chunk_size:
            return self.predict(self.cmap(z0, z1))

        pad = self.conv.padding[0]
        n_rows = z0.shape[1]
        rows = []

        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            halo_start = max(0, start - pad)
            halo_end = min(n_rows, end + pad)

            C = self.cmap(z0[:, halo_start:halo_end], z1)
            S = self.predict(C)

            crop_start = start - halo_start
            crop_end = crop_start + (end - start)
            rows.append(S[:, :, crop_start:crop_end])

        return torch.cat(rows, dim=2)

    def cmap(self, z0, z1):
        C = self.hidden(z0, z1)
        C = self.cmap_attn(C)
        return C

    def predict(self, C):
        s = self.conv(C)
        s = self.batchnorm(s)
        s = self.activation(s)
        return s
