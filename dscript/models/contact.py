# Input: C = NxMxH embedding contact matrix
# Output: S = MxN contact prediction matrix

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnected(nn.Module):
    """
    Performs part 1 of Contact Prediction Module. Takes embeddings from Projection module and produces broadcast tensor.

    Input embeddings of dimension :math:`d` are combined into a :math:`2d` length MLP input :math:`z_{cat}`, where :math:`z_{cat} = [z_0 \\ominus z_1 | z_0 \\odot z_1]`

    :param embed_dim: Output dimension of `dscript.models.embedding <#module-dscript.models.embedding>`_ model :math:`d` [default: 100]
    :type embed_dim: int
    :param hidden_dim: Hidden dimension :math:`h` [default: 50]
    :type hidden_dim: int
    :param activation: Activation function for broadcast tensor [default: torch.nn.ReLU()]
    :type activation: torch.nn.Module
    """

    def __init__(self, embed_dim, hidden_dim, activation=nn.ReLU()):
        super().__init__()

        self.D = embed_dim
        self.H = hidden_dim
        self.conv = nn.Conv2d(2 * self.D, self.H, 1)
        self.batchnorm = nn.BatchNorm2d(self.H)
        """
        self.proj = nn.Linear(121, 100)
        """
        self.activation = activation

    def forward(self, z0, z1):
        """
        :param z0: Projection module embedding :math:`(b \\times N \\times d)`
        :type z0: torch.Tensor
        :param z1: Projection module embedding :math:`(b \\times M \\times d)`
        :type z1: torch.Tensor
        :return: Predicted broadcast tensor :math:`(b \\times N \\times M \\times h)`
        :rtype: torch.Tensor
        """

        # z0 is (b,N,d), z1 is (b,M,d)
        """
        z0 = self.proj(z0)
        z1 = self.proj(z1)
        """

        z0 = z0.transpose(1, 2)
        z1 = z1.transpose(1, 2)

        # z0 is (b,d,N), z1 is (b,d,M)

        z0 = z0.unsqueeze(3)
        z1 = z1.unsqueeze(2)

        w_dif, w_mul = self.conv.weight.split(self.D, dim=1)

        z_dif = torch.abs(z0 - z1)
        c = F.conv2d(z_dif, w_dif, self.conv.bias)
        del z_dif

        z_mul = z0 * z1
        c = c + F.conv2d(z_mul, w_mul, None)
        c = self.activation(c)
        c = self.batchnorm(c)

        return c


class ContactCNN(nn.Module):
    """
    Residue Contact Prediction Module. Takes embeddings from Projection module and produces contact map, output of Contact module.

    :param embed_dim: Output dimension of `dscript.models.embedding <#module-dscript.models.embedding>`_ model :math:`d` [default: 100]
    :type embed_dim: int
    :param hidden_dim: Hidden dimension :math:`h` [default: 50]
    :type hidden_dim: int
    :param width: Width of convolutional filter :math:`2w+1` [default: 7]
    :type width: int
    :param activation: Activation function for final contact map [default: torch.nn.Sigmoid()]
    :type activation: torch.nn.Module
    """

    def __init__(self, embed_dim, hidden_dim=50, width=7, activation=nn.Sigmoid()):
        super().__init__()

        self.hidden = FullyConnected(embed_dim, hidden_dim)

        self.conv = nn.Conv2d(hidden_dim, 1, width, padding=width // 2)
        self.batchnorm = nn.BatchNorm2d(1)
        self.activation = activation
        self.clip()

    def clip(self):
        """
        Force the convolutional layer to be transpose invariant.

        :meta private:
        """
        w = self.conv.weight
        self.conv.weight.data[:] = 0.5 * (w + w.transpose(2, 3))

    def forward(self, z0, z1):
        """
        :param z0: Projection module embedding :math:`(b \\times N \\times d)`
        :type z0: torch.Tensor
        :param z1: Projection module embedding :math:`(b \\times M \\times d)`
        :type z1: torch.Tensor
        :return: Predicted contact map :math:`(b \\times N \\times M)`
        :rtype: torch.Tensor
        """
        C = self.cmap(z0, z1)
        return self.predict(C)

    def predict_from_embeddings(self, z0, z1, chunk_size=None):
        """
        Predict a contact map directly from embeddings.

        When ``chunk_size`` is set, rows from ``z0`` are processed in chunks with
        the convolution halo needed to match full-map prediction at chunk edges.
        This avoids materializing the full hidden broadcast tensor.
        """
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
        """
        Calls `dscript.models.contact.FullyConnected <#module-dscript.models.contact.FullyConnected>`_.

        :param z0: Projection module embedding :math:`(b \\times N \\times d)`
        :type z0: torch.Tensor
        :param z1: Projection module embedding :math:`(b \\times M \\times d)`
        :type z1: torch.Tensor
        :return: Predicted contact broadcast tensor :math:`(b \\times N \\times M \\times h)`
        :rtype: torch.Tensor
        """
        C = self.hidden(z0, z1)
        return C

    def predict(self, C):
        """
        Predict contact map from broadcast tensor.

        :param B: Predicted contact broadcast :math:`(b \\times N \\times M \\times h)`
        :type B: torch.Tensor
        :return: Predicted contact map :math:`(b \\times N \\times M)`
        :rtype: torch.Tensor
        """

        # S is (b,N,M)
        s = self.conv(C)
        s = self.batchnorm(s)
        s = self.activation(s)
        return s
