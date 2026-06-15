import torch

from src.models.semlaflow.layers import EquiNorm
from src.models.semlaflow.layers import InvNorm

"""
Masks are always boolean tensors, where a value of 1 indicates masking, and a value of 0 indicates 
no masking.
"""


class InvariantMLP(torch.nn.Module):
    r"""Implements $h^{\text{ff}}_i = h_i + \Phi_{\theta}(\tilde{h}_i, \| \phi_{equi}(x_i) \|)$ Eq.3"""

    def __init__(self, d_model, n_coord_sets, d_ff=None):
        super().__init__()

        d_ff = d_model * 4 if d_ff is None else d_ff

        self.Phi = torch.nn.Sequential(
            torch.nn.Linear(d_model + n_coord_sets, d_ff),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_ff, d_model),
        )

    def forward(self, x, h, x_norm, h_norm):
        """
        Note: x is not used. Kept for consistency.
        Args:
            x (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            h (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            x_norm (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            h_norm (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]

        Returns:
            torch.Tensor: Updated node features, shape [batch_size, n_nodes, d_model]
        """

        lengths = torch.linalg.vector_norm(x_norm, dim=-1).transpose(1, 2)
        phi_input = torch.cat((h_norm, lengths), dim=2)
        h_ff = h + self.Phi(phi_input)
        return h_ff


class EquivariantMLP(torch.nn.Module):
    r"""Implements $x^{\text{ff}}_i = x_i + W^2_{\theta} (\sum_{j=1}^{d_{equi}} \tilde{x}^j_i \otimes
    \Psi_{\theta}(\tilde{h}_i))$ Eq.4
    """

    def __init__(self, d_model, n_coord_sets, proj_sets=None):
        """
        proj_sets is an integer for increasing/decreasing the number of coordinate sets.
        """
        super().__init__()
        proj_sets = n_coord_sets if proj_sets is None else proj_sets

        self.Psi = torch.nn.Sequential(
            torch.nn.Linear(d_model, proj_sets),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(proj_sets, proj_sets),
        )
        self.W1 = torch.nn.Linear(n_coord_sets, proj_sets, bias=False)
        self.W2 = torch.nn.Linear(proj_sets, n_coord_sets, bias=False)

    def forward(self, x, h, x_norm, h_norm):
        """
        Note: h is not used. Kept for consistency.
        Args:
            x (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            h (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            x_norm (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            h_norm (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]

        Returns:
            torch.Tensor: Updated coord_sets, shape [batch_size, n_sets, n_nodes, 3]
        """
        # B: batch_size
        # S: n_sets
        # P: proj_sets (can be =S, >S or <S)
        # N: n_nodes

        psi = self.Psi(h_norm).unsqueeze(1)
        x_tilde = self.W1(x_norm.transpose(1, -1))  # -1 -> 3
        # Output dims:
        # inv_feats shape [B, 1, N, P]
        # proj_sets shape [B, 3, N, P]

        # Outer product with invariant features is equivariant, then sum over original coord sets
        attentions = x_tilde.unsqueeze(-2) * psi.unsqueeze(-1)
        # Output dims:
        # attentions shape [B, 1, N, P, 1] * [B, 3, N, 1, P] = [B, 3, N, P, P]

        attentions = attentions.sum(-1)
        # attentions shape [B, 3, N, P]

        x_ff = x + self.W2(attentions).transpose(1, -1)  # -1 -> 3
        # attentions shape [B, N, P, 3]
        return x_ff


class FeedForward(torch.nn.Module):
    """Implements Eq. 2, 3 (InvariantMLP) and 4 (EquivariantMLP)"""

    def __init__(
        self,
        d_model,
        n_coord_sets,
        remove_com,
        d_ff=None,
        proj_sets=None,
        coord_norm="length",
    ):
        super().__init__()
        self.phi_equi = EquiNorm(n_coord_sets, norm=coord_norm, remove_com=remove_com)
        self.phi_inv = InvNorm(d_model)
        self.invariant_mlp = InvariantMLP(d_model, n_coord_sets, d_ff=d_ff)
        self.equivariant_mlp = EquivariantMLP(
            d_model, n_coord_sets, proj_sets=proj_sets
        )

    def forward(self, x, h, mask):
        """
         Args:
            x (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            h (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 mask, 0 keep

        Returns:
            torch.Tensor, torch.Tensor: Updates to coords and node features
        """
        x_norm = self.phi_equi(x, mask)
        h_norm = self.phi_inv(h)
        x_ff = self.equivariant_mlp(x, h, x_norm, h_norm)
        h_ff = self.invariant_mlp(x, h, x_norm, h_norm)
        return x_ff, h_ff
