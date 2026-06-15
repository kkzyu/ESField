import torch

from src.models.semlaflow.layers import EquiNorm
from src.models.semlaflow.layers import InvNorm


def adj_to_attn_mask(adj_matrix, pos_inf=False):
    """
    Args:
           adj_matrix (torch.Tensor): adjacency matrix with 0s and 1s shape [batch_size, n_nodes, n_nodes]
       Returns:
           torch.Tensor:  attention mask where paddings are 0 and disconnections are inf
           shape [batch_size, n_nodes, n_nodes]
    """

    inf = float("inf") if pos_inf else float("-inf")
    attn_mask = torch.zeros_like(adj_matrix.float())
    attn_mask[adj_matrix == 0] = inf

    # Ensure nodes with no connections (fake nodes) don't have all -inf in the attn mask
    # Otherwise we would have problems when softmaxing
    n_nodes = adj_matrix.sum(dim=-1)
    attn_mask[n_nodes == 0] = 0.0

    return attn_mask


class Omega(torch.nn.Module):
    def __init__(
        self, d_message, d_out, n_coord_sets, d_message_hidden=None, d_edge=None
    ):
        super().__init__()
        self.n_coord_sets = n_coord_sets
        edge_feats = 0 if d_edge is None else d_edge
        extra_feats = n_coord_sets + edge_feats
        d_in_feats = (d_message * 2) + extra_feats
        d_message_hidden = d_out if d_message_hidden is None else d_message_hidden
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(d_in_feats, d_message_hidden),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_message_hidden, d_out),
        )

    def _equi_pairwise(self, x_norm):
        """'
        Implement dot product message passing for equivariant features (coordinates)
        Args:
            x_norm (torch.Tensor): Normalised Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
        Returns:
            torch.Tensor: Updated pairwise coordinate message, shape [batch_size, n_nodes, n_nodes, n_sets]
        """
        # [batch_size, n_sets, n_nodes, 3] -> [batch_size x n_sets, n_nodes, 3]
        x_norm = x_norm.flatten(0, 1)
        # [batch_size x n_sets, n_nodes, n_nodes]
        x_dotprods = torch.bmm(x_norm, x_norm.transpose(1, 2))
        # [batch_size, n_nodes, n_nodes, n_sets]
        x_pairs = x_dotprods.unflatten(0, (-1, self.n_coord_sets)).movedim(1, -1)
        return x_pairs

    def _invar_pariwise(self, h_norm):
        """'
        Implement latent projection message passing for invariant features (atom types, charges etc)
        Args:
            h_norm (torch.Tensor): Normalised Node feature tensor, shape [batch_size, n_nodes, d_model]
        Returns:
            torch.Tensor: Updated pairwise node message, shape [batch_size, n_nodes, n_nodes, d_message x 2]
        """
        batch_size, n_nodes, _ = tuple(h_norm.shape)
        # [batch_size, n_nodes, n_nodes, d_message]
        h_source = h_norm.unsqueeze(2).expand(batch_size, n_nodes, n_nodes, -1)
        # [batch_size, n_nodes, n_nodes, d_message]
        h_target = h_norm.unsqueeze(1).expand(batch_size, n_nodes, n_nodes, -1)
        # [batch_size, n_nodes, n_nodes, d_message x 2]
        h_pairs = torch.cat((h_source, h_target), dim=-1)
        return h_pairs

    def forward(self, x_norm, h_norm, *, e_norm=None):
        """'
        Implement the omega function that compute message passing for both equivariant and invariant features
        Args:
            x_norm (torch.Tensor): Normalised Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            h_norm (torch.Tensor): Normalised Node feature tensor, shape [batch_size, n_nodes, d_model]
            h_norm (torch.Tensor): Normalised Edge feature tensor, shape [batch_size, n_nodes, n_nodes, d_edge]
        Returns:
            torch.Tensor: Updated pairwise message, shape [batch_size, n_nodes, n_nodes, d_out]
        """
        x_pairs = self._equi_pairwise(x_norm)
        h_pairs = self._invar_pariwise(h_norm)
        message_pairs = torch.cat((h_pairs, x_pairs), dim=3)
        if e_norm is not None:
            message_pairs = torch.cat((message_pairs, e_norm), dim=-1)
        return self.message_mlp(message_pairs)


class SimpleLigandProteinMessage(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_out,
        n_coord_sets,
        remove_com,
        coord_norm="none",
        eps=1e-6,
    ):
        super().__init__()
        self.phi_equi = EquiNorm(n_coord_sets, norm=coord_norm, remove_com=remove_com)
        self.phi_inv = InvNorm(d_model)
        self.lpW = None
        self.eps = eps
        d_in = d_model * 2 + n_coord_sets
        self.project = torch.nn.Sequential(
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_in, d_out),
        )

    def _equi_lp_message(self, x_norm, px_norm):
        # Compute squared Euclidean distance
        diff = x_norm.unsqueeze(-2) - px_norm.unsqueeze(-3)
        # [batch_size, n_nodes, n_pnodes, n_sets]
        lp_dist = torch.sqrt((diff**2).sum(dim=-1) + self.eps).movedim(1, -1)
        return lp_dist

    def _inv_lp_message(self, h_norm, ph_norm):
        n_nodes, n_pnodes = h_norm.shape[1], ph_norm.shape[1]
        # [batch_size, n_nodes, n_pnodes, d_model]
        h_expanded = h_norm.unsqueeze(2).expand(-1, -1, n_pnodes, -1)
        # [batch_size, n_nodes, n_pnodes, d_model]
        hp_expanded = ph_norm.unsqueeze(1).expand(-1, n_nodes, -1, -1)
        # [batch_size, n_nodes, n_pnodes, d_modelx2]
        lp_h = torch.cat([h_expanded, hp_expanded], dim=-1)
        return lp_h

    def forward(self, x_norm, h_norm, px, ph, pmask):
        px_norm = self.phi_equi(px, pmask)
        ph_norm = self.phi_inv(ph)
        lph_m = self._inv_lp_message(h_norm, ph_norm)
        lpx_m = self._equi_lp_message(x_norm, px_norm)
        lp_m = torch.cat((lph_m, lpx_m), dim=3)
        lp_m = self.project(lp_m)
        return lp_m


class LatentMessagePassing(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_message,
        d_out,
        n_coord_sets,
        remove_com,
        d_message_hidden=None,
        d_edge=None,
        coord_norm="none",
    ):
        super().__init__()
        self.phi_equi = EquiNorm(n_coord_sets, norm=coord_norm, remove_com=remove_com)
        self.phi_inv = InvNorm(d_model)
        self.phi_inv_e = InvNorm(d_edge) if d_edge is not None else None
        # TO CHANGE : use_bias = False
        self.W3 = torch.nn.Linear(d_model, d_message)
        self.omega = Omega(
            d_message,
            d_out,
            n_coord_sets,
            d_message_hidden=d_message_hidden,
            d_edge=d_edge,
        )

    def forward(self, x, h, e, mask):
        """'
        Implement Section 3.2(a) : Latent Message Passing
        Args:
            x (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            h (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            e (torch.Tensor): Edge feature tensor, shape [batch_size, n_nodes, n_nodes, d_edge]
            mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 mask, 0 keep
        Returns:
            message_pairs (torch.Tensor): Updated pairwise message, shape [batch_size, n_nodes, n_nodes, d_out]
            x_norm (torch.Tensor): Normalised Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            h_norm (torch.Tensor): Normalised Node feature tensor, shape [batch_size, n_nodes, d_model]
        """

        x_norm = self.phi_equi(x, mask)
        h_norm = self.phi_inv(h)
        e_norm = self.phi_inv_e(e) if e is not None else None
        self.e_norm = e_norm
        # [batch_size, n_nodes, d_model] -> [batch_size, n_nodes, d_message]
        h_norm_out = self.W3(h_norm)
        message_pairs = self.omega(x_norm, h_norm_out, e_norm=e_norm)
        return message_pairs, x_norm, h_norm


class InvariantAttention(torch.nn.Module):
    def __init__(self, d_model, n_attn_heads, d_attn=None):
        super().__init__()
        d_attn = d_model if d_attn is None else d_attn
        if d_attn % n_attn_heads != 0:
            raise ValueError(
                "n_attn_heads must divide d_model (or d_attn if provided) exactly."
            )
        d_head = d_attn // n_attn_heads

        self.d_model = d_model
        self.d_attn = d_attn
        self.n_attn_heads = n_attn_heads
        self.d_head = d_head
        self.phi_inv = InvNorm(d_model)
        # TO FIX: use_bias=False
        self.W4 = torch.nn.Linear(d_model, d_attn)
        self.W5 = torch.nn.Linear(d_attn, d_model)

    def forward(self, h, m_inv, adj_matrix):
        """Compute invariant attention updates

        Args:
            h (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            m_inv (torch.Tensor): Messages tensor, shape [batch_size, n_nodes, n_nodes, n_heads]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_nodes, n_nodes]

        Returns:
            torch.Tensor: Aggregated node features, shape [batch_size, n_nodes, d_model]
        """
        attn_mask = adj_to_attn_mask(adj_matrix)
        m_inv = m_inv + attn_mask.unsqueeze(3)
        attentions = torch.softmax(m_inv, dim=2)
        h_norm = self.phi_inv(h)
        # [batch_size, n_nodes, d_attn]
        h_tilde = self.W4(h_norm)
        # spliting d_attn node features into n_attn_heads segments: [batch_size, n_nodes, n_attn_heads, d_head]
        h_tilde = h_tilde.unflatten(-1, (self.n_attn_heads, self.d_head))
        # [batch_size, n_nodes, n_attn_heads, d_head]  -> [batch_size * n_attn_heads, n_nodes, d_head]
        h_tilde = h_tilde.movedim(-2, 1).flatten(0, 1)
        # [batch_size, n_nodes, n_nodes, n_heads]  -> [batch_size * n_attn_heads, n_nodes, n_nodes]
        attentions = attentions.movedim(-1, 1).flatten(0, 1)
        # [batch_size * n_attn_heads, n_nodes, d_head]
        aggregated_h = torch.bmm(attentions, h_tilde)
        # Apply variance preserving updates as proposed in GNN-VPA (https://arxiv.org/abs/2403.04747)
        # $\sqrt{\sum_{j=1}^N{(\alpha_{i,j}^2)^k}}$
        # TO FIX IN PAPER: add squared
        weights = torch.sqrt((attentions**2).sum(dim=-1))
        # $\sum_{j=1}^{N}{\alpha_{i,j}^{k}\tilde{h}_j^k}$
        aggregated_h = aggregated_h * weights.unsqueeze(-1)
        # [batch_size, n_attn_heads, n_nodes, d_head]
        aggregated_h = aggregated_h.unflatten(0, (-1, self.n_attn_heads))
        # [batch_size, n_nodes, n_attn_heads * d_head]
        # $\|_{k=1}^{K}w_i^k a_i^k$
        aggregated_h = aggregated_h.movedim(1, -2).flatten(2, 3)
        return self.W5(aggregated_h)


class EquivariantAttention(torch.nn.Module):
    def __init__(
        self, n_coord_sets, proj_sets, remove_com, coord_norm="length", eps=1e-6
    ):
        super().__init__()
        self.eps = eps
        self.phi_equi = EquiNorm(n_coord_sets, norm=coord_norm, remove_com=remove_com)
        # does not present in the paper, included to avoid shape error when proj_sets!= n_coord_sets
        # when proj_sets == n_coord_sets behaves the same as described in paper
        self.W_fix_0 = (
            torch.nn.Linear(n_coord_sets, proj_sets, bias=False)
            if n_coord_sets != proj_sets
            else lambda x: x
        )
        self.W6 = torch.nn.Linear(n_coord_sets, proj_sets, bias=False)
        self.W7 = torch.nn.Linear(proj_sets, n_coord_sets, bias=False)

    def forward(self, x, m_equi, adj_matrix, mask):
        """Compute Equivariant Attention Updates

        Args:
            x (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            m_equi (torch.Tensor): Equivariant messages, shape [batch_size, n_nodes, n_nodes, n_sets]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_nodes, n_nodes]
            mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 for real, 0 otherwise

        Returns:
            torch.Tensor: Updated coordinate sets, shape [batch_size, n_sets, n_nodes, 3]
        """

        attn_mask = adj_to_attn_mask(adj_matrix)
        m_equi = m_equi + attn_mask.unsqueeze(3)
        attentions = torch.softmax(m_equi, dim=2)
        # attentions shape: [batch_size, n_nodes, n_nodes, n_sets]

        attentions_tilde = self.W_fix_0(attentions)
        # attentions_tilde shape: [batch_size, n_nodes, n_nodes, proj_sets]

        x_norm = self.phi_equi(x, mask)
        # x_norm shape: [batch_size, n_sets, n_nodes, 3]

        x_tilde = self.W6(x_norm.transpose(1, -1))
        # x_norm.transpose(1, -1) shape: [batch_size, 3, n_nodes, n_sets]
        # x_tilde shape: [batch_size, 3, n_nodes, proj_sets]

        x_dists = x_tilde.unsqueeze(3) - x_tilde.unsqueeze(2)
        # x_dists shape: [batch_size, 3, n_nodes, n_nodes, proj_sets]

        lengths = torch.linalg.norm(x_dists, dim=1, keepdim=True)
        # lengths shape: [batch_size, 1, n_nodes, n_nodes, proj_sets]
        x_dists = x_dists / (lengths + self.eps)
        # generalised to protein conditioned case where x_dists.shape[2] = n_nodes+n_pnodes
        if x_dists.shape[2] != attentions_tilde.shape[1]:
            x_dists = x_dists[:, :, : attentions_tilde.shape[1], :]
        aggregated_x_dists = torch.einsum(
            "bkniaj,bknaij->bknij",
            x_dists.unsqueeze(3),
            attentions_tilde.unsqueeze(-2).unsqueeze(1),
        ).squeeze(-2)
        # x_dists.unsqueeze(3) shape: [B, 3, N, 1, N, P]
        # attentions_tilde.unsqueeze(-2).unsqueeze(1) shape: [B, 1, N, N, 1, P]
        # aggregated_x_dists shape: [B, 3, N, P]

        """
        Equivalent to:
        B, N, P = 32, 50, 256
        device = "cuda"
        x_tilde = torch.randn(B, 3, N, P, device=device)
        attentions_tilde = torch.randn(B, 1, N, N, 1, P, device=device)
            
        x_dists = x_tilde.unsqueeze(3) - x_tilde.unsqueeze(2)
        lengths = torch.linalg.vector_norm(x_dists, dim=1, keepdim=True)
        x_dists /= (lengths + 0.01)
        x_dists = x_dists.unsqueeze(3)
        
        A1, B1 = x_dists, attentions_tilde
    
        r1 = torch.einsum("bkniaj,bknaij->bknij", A1, B1).squeeze(-2)
  
        A2 = A1.permute(0, 1, 2, 5, 3, 4).reshape((-1, 1, N))
        B2 = B1.permute(0, 1, 2, 5, 3, 4).repeat(1,3,1,1,1,1).reshape((-1, N, 1))
        r2 = torch.bmm(A2, B2).reshape((B, 3, N, P))
        assert torch.allclose(r1, r2, atol=1e-4)
        """

        # $\sqrt{\sum_{j=1}^N{(\alpha_{i,j}^2)^k}}$
        weights = torch.sqrt((attentions_tilde**2).sum(dim=2))
        aggregated_x_dists = aggregated_x_dists * weights.unsqueeze(1)
        # output shape: [batch_size, n_nodes, n_nodes, n_sets, 3]
        return self.W7(aggregated_x_dists).transpose(1, -1)


class GraphAttention(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_message,
        n_coord_sets,
        proj_sets,
        remove_com,
        n_attn_heads=None,
        d_message_hidden=None,
        d_edge=None,
        edge_in=False,
        edge_out=False,
        coord_norm="length",
        protein_conditioning=False,
        eps=1e-6,
    ):
        super().__init__()
        if edge_in and edge_out:
            raise ValueError("`edge_in` and `edge_out` cannot be both positive")
        if (edge_in or edge_out) and d_edge is None:
            raise ValueError(
                "`edge_in` or `edge_out` are positive but `d_edge` is None"
            )

        d_attn = d_model
        n_attn_heads = d_message if n_attn_heads is None else n_attn_heads
        self.n_attn_heads = n_attn_heads
        self.n_coord_sets = n_coord_sets
        d_message_out = n_attn_heads + self.n_coord_sets
        if protein_conditioning:
            self.simple_lp_message = SimpleLigandProteinMessage(
                d_model, d_message_out, n_coord_sets, remove_com, coord_norm
            )
        if (d_edge is not None) and edge_out:
            d_message_out += d_edge
        if proj_sets is None:
            proj_sets = n_coord_sets
        self.pairwise_messages = LatentMessagePassing(
            d_model,
            d_message,
            d_message_out,
            n_coord_sets,
            remove_com=remove_com,
            d_message_hidden=d_message_hidden,
            d_edge=d_edge if edge_in else None,
        )
        self.invariant_attention = InvariantAttention(
            d_model, n_attn_heads, d_attn=d_attn
        )
        self.coordinate_attention = EquivariantAttention(
            n_coord_sets,
            remove_com=remove_com,
            proj_sets=proj_sets,
            coord_norm=coord_norm,
            eps=eps,
        )
        self.edge_in = edge_in
        self.edge_out = edge_out
        self.protein_conditioning = protein_conditioning

    def forward(
        self, x_ff, h_ff, adj_matrix, e, mask, px_ff=None, ph_ff=None, pmask=None
    ):
        """
         Args:
            x_ff (torch.Tensor): Updated Coordinate tensor from feedforward layer, shape [batch_size, n_sets, n_nodes, 3]
            h_ff (torch.Tensor): Updated Node feature tensor from feedforward layer, shape [batch_size, n_nodes, d_model]
            adj_matrix (torch.Tensor): Adjacency matrix, shape [batch_size, n_nodes, n_nodes | n_nodes+n_pnodes]
            e (torch.Tensor): Edge feature tensor, shape [batch_size, n_nodes, n_nodes, d_edge]
            mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 mask, 0 keep
            px_ff (torch.Tensor): Updated Protein Coordinate tensor from feedforward layer, shape [batch_size, n_sets, n_pnodes, 3]
            ph_ff (torch.Tensor): Updated Protein Node feature tensor from feedforward layer, shape [batch_size, n_pnodes, d_model]
            pmask (torch.Tensor): Mask for protein nodes, shape [batch_size, n_sets, n_nodes], 1 mask, 0 keep

        Returns:
            Updated x, h and (e)  (torch.Tensor)
            x_out (torch.Tensor): Coordinate tensor updated by message passing and attention mechanism [batch_size, n_nodes, 3],
            h_out (torch.Tensor): Node feature tensor updated by message passing and attention mechanism [batch_size, n_nodes, d_model],
            optional : e_out (torch.Tensor): Edge feature tensor updated by message passing mechanism [batch_size, n_nodes, n_nodes, d_edge],

        """
        m, x_norm, h_norm = self.pairwise_messages(x_ff, h_ff, e, mask)
        m_inv = m[:, :, :, : self.n_attn_heads]
        m_equi = m[:, :, :, self.n_attn_heads : (self.n_attn_heads + self.n_coord_sets)]
        h_ff_original = h_ff.clone().to(h_ff.device)
        x_ff_original = x_ff.clone().to(x_ff.device)
        if (self.protein_conditioning) and (px_ff is not None):
            lp_m = self.simple_lp_message(x_norm, h_norm, px_ff, ph_ff, pmask)
            lp_m_inv = lp_m[:, :, :, : self.n_attn_heads]
            lp_m_equi = lp_m[
                :, :, :, self.n_attn_heads : (self.n_attn_heads + self.n_coord_sets)
            ]
            # [batch_size, n_nodes, n_nodes+n_pnodes, n_attn_heads]
            m_inv = torch.cat([m_inv, lp_m_inv], dim=-2)
            # [batch_size, n_nodes, n_nodes+n_pnodes, n_coord_sets]
            m_equi = torch.cat([m_equi, lp_m_equi], dim=-2)
            # [batch_size, n_nodes+n_pnodes, d_model]
            h_ff = torch.cat([h_ff, ph_ff], dim=-2)
            # [batch_size, n_sets, n_nodes+n_pnodes, 3]
            x_ff = torch.cat([x_ff, px_ff], dim=-2)
            mask = torch.cat([mask, pmask], dim=-1)
        h_out = h_ff_original + self.invariant_attention(h_ff, m_inv, adj_matrix)
        x_out = x_ff_original + self.coordinate_attention(
            x_ff, m_equi, adj_matrix, mask
        )
        if self.edge_out:
            m_e = m[:, :, :, (self.n_attn_heads + self.n_coord_sets) :]
            e_out = e + m_e if e is not None else m_e
            return x_out, h_out, e_out
        return x_out, h_out, None
