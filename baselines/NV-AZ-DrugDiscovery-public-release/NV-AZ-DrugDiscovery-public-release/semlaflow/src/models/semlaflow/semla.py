import torch
import copy

from src.models.semlaflow.layers import EquiNorm
from src.models.semlaflow.layers import InvNorm
from src.models.semlaflow.layers import BondRefine
from src.models.semlaflow.feed_forward import FeedForward
from src.models.semlaflow.graph_attention import GraphAttention
from src.models.semlaflow.utils import edges_from_nodes


class SemlaLayer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_message,
        n_coord_sets,
        d_ff,
        d_message_hidden,
        d_edge,
        edge_in,
        edge_out,
        n_attn_heads,
        proj_sets,
        coord_norm,
        eps,
        remove_com,
        protein_conditioning,
    ):
        super().__init__()

        # are we sure this is correct
        # proj_sets=d_message?
        self.feed_forward = FeedForward(
            d_model,
            n_coord_sets,
            d_ff=d_ff,
            proj_sets=d_message,
            coord_norm=coord_norm,
            remove_com=remove_com,
        )
        self.pfeed_forward = FeedForward(
            d_model,
            n_coord_sets,
            d_ff=d_ff,
            proj_sets=d_message,
            coord_norm=coord_norm,
            remove_com=remove_com,
        )
        self.graph_attention = GraphAttention(
            d_model,
            d_message,
            n_coord_sets,
            proj_sets,
            remove_com=remove_com,
            n_attn_heads=n_attn_heads,
            d_message_hidden=d_message_hidden,
            d_edge=d_edge,
            edge_in=edge_in,
            edge_out=edge_out,
            coord_norm=coord_norm,
            eps=eps,
            protein_conditioning=protein_conditioning,
        )
        self.protein_conditioning = protein_conditioning

    def forward(self, x, h, adj_matrix, e, mask, px=None, ph=None, pmask=None):
        """
         Args:
            x (torch.Tensor): Coordinate tensor, shape [batch_size, n_sets, n_nodes, 3]
            h (torch.Tensor): Node feature tensor, shape [batch_size, n_nodes, d_model]
            adj_matrix (torch.Tensor): Adjacency matrix, typically computed
                upstream via ``smolF.edges_from_nodes(coords, node_mask=atom_mask)``.
            e (torch.Tensor): Edge feature tensor, shape [batch_size, n_nodes, n_nodes, d_edge]
            mask (torch.Tensor): Mask for nodes, shape [batch_size, n_sets, n_nodes], 1 mask, 0 keep

        Returns:
            torch.Tensor, torch.Tensor: Updates to coords and node features
        """
        x_ff, h_ff = self.feed_forward(x, h, mask)
        px_ff, hp_ff = [None, None]
        if (self.protein_conditioning) and (px is not None):
            px_ff, hp_ff = self.pfeed_forward(px, ph, pmask)
        x_ff, h_ff, e_ff = self.graph_attention(
            x_ff, h_ff, adj_matrix, e, mask, px_ff, hp_ff, pmask
        )
        return x_ff, h_ff, e_ff


class SemlaFlow(torch.nn.Module):
    def __init__(
        self,
        n_layers,
        d_model,
        d_message,
        n_coord_sets,
        atom_vocab_size,
        charge_vocab_size,
        edge_vocab_size=None,
        size_emb=64,
        max_atoms=256,
        d_ff=None,
        d_message_hidden=None,
        d_edge=None,
        n_attn_heads=None,
        proj_sets=None,
        coord_norm="length",
        eps=1e-6,
        bond_refine=True,
        self_conditioning=False,
        strict=True,
        remove_com=True,
        protein_conditioning=False,
        patom_vocab_size=None,
        max_patoms=256,
    ):
        self.params = {}
        for key, value in locals().items():
            if key not in ["self", "__class__"]:
                self.params[key] = value
        super().__init__()

        extra_layers = 2 if d_edge is not None else 0
        if extra_layers > n_layers:
            raise ValueError("`n_layers` is too small.")

        n_attn_heads = d_message if n_attn_heads is None else n_attn_heads
        self.params["n_attn_heads"] = n_attn_heads

        if d_model != ((d_model // n_attn_heads) * n_attn_heads):
            if strict:
                raise ValueError(
                    f"`n_attn_heads` must exactly divide `d_model`, got {n_attn_heads} and {d_model}"
                )
            else:
                d_model = (d_model // n_attn_heads) * n_attn_heads
                self.params["d_model"] = d_model

        if d_edge is not None or edge_vocab_size is not None:
            if None in [d_edge, edge_vocab_size]:
                raise ValueError(
                    "If either d_edge or edge_vocab_size are given both must be provided."
                )

            edge_in_feats = (
                edge_vocab_size * 2 if self_conditioning else edge_vocab_size
            )

            self.edge_in_proj = torch.nn.Sequential(
                torch.nn.Linear(edge_in_feats, d_edge),
                torch.nn.SiLU(inplace=False),
                torch.nn.Linear(d_edge, d_edge),
            )
            self.edge_out_proj = torch.nn.Sequential(
                torch.nn.Linear(d_edge, d_edge),
                torch.nn.SiLU(inplace=False),
                torch.nn.Linear(d_edge, edge_vocab_size),
            )

        # Input features: atom-type logits (doubled when self-conditioning
        # since the model also sees its last prediction) plus one dimension
        # for the timestep.
        in_feats = atom_vocab_size * 2 + 1 if self_conditioning else atom_vocab_size + 1
        in_feats = in_feats + size_emb

        self.size_emb = torch.nn.Embedding(max_atoms, size_emb)
        self.feat_proj = torch.nn.Sequential(
            torch.nn.Linear(in_feats, d_model),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_model, d_model),
        )
        semla_layers = []
        for i in range(n_layers):
            if ((i == 0) or (i == n_layers - 1)) and d_edge is not None:
                d_m_h = None
                d_e = d_edge
                edge_in = i == 0
                edge_out = i == n_layers - 1

            else:
                d_m_h = d_message_hidden
                d_e = None
                edge_in = False
                edge_out = False

            semla_layer = SemlaLayer(
                d_model=d_model,
                d_message=d_message,
                n_coord_sets=n_coord_sets,
                d_ff=d_ff,
                d_message_hidden=d_m_h,
                d_edge=d_e,
                edge_in=edge_in,
                edge_out=edge_out,
                n_attn_heads=n_attn_heads,
                proj_sets=proj_sets,
                coord_norm=coord_norm,
                eps=eps,
                remove_com=remove_com,
                protein_conditioning=protein_conditioning,
            )
            semla_layers.append(semla_layer)

        self.semla_layers = torch.nn.ModuleList(semla_layers)

        self.final_ff_block = FeedForward(
            d_model,
            n_coord_sets,
            d_ff=d_ff,
            proj_sets=proj_sets,
            coord_norm=coord_norm,
            remove_com=remove_com,
        )
        self.coord_norm = EquiNorm(
            n_coord_sets, remove_com=remove_com, norm=coord_norm, eps=eps
        )
        self.feat_norm = InvNorm(d_model)

        in_coord_sets = 2 if self_conditioning else 1
        self.coord_proj = torch.nn.Linear(in_coord_sets, n_coord_sets, bias=False)
        self.coord_head = torch.nn.Linear(n_coord_sets, 1, bias=False)

        if d_edge is not None:
            self.bond_norm = InvNorm(d_edge)

        if bond_refine:
            self.refine_layer = BondRefine(
                d_model, d_message, d_edge, d_ff, remove_com=remove_com
            )

        self.atom_classifier_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_model, atom_vocab_size),
        )
        self.charge_classifier_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.SiLU(inplace=False),
            torch.nn.Linear(d_model, charge_vocab_size),
        )
        if protein_conditioning:
            self.psize_emb = torch.nn.Embedding(max_patoms, size_emb)
            if self_conditioning:
                self.pcoord_proj = torch.nn.Linear(1, n_coord_sets, bias=False)
                in_pfeats = patom_vocab_size + size_emb
                self.pfeat_proj = torch.nn.Sequential(
                    torch.nn.Linear(in_pfeats, d_model),
                    torch.nn.SiLU(inplace=False),
                    torch.nn.Linear(d_model, d_model),
                )
            else:
                self.pcoord_proj = copy.deepcopy(self.coord_proj)
                self.pfeat_proj = copy.deepcopy(self.feat_proj)

    def forward(
        self,
        x,
        h,
        e=None,
        mask=None,
        given_x=None,
        given_h=None,
        given_e=None,
        px=None,
        ph=None,
        pmask=None,
    ):
        """Predict molecular coordinates and atom types

        Args:
            x (torch.Tensor): Input coordinates, shape [batch_size, n_atoms, 3]
            h (torch.Tensor): Invariant atom features, shape [batch_size, n_atoms, n_feats]
            e (torch.Tensor): In edge features, shape [batch_size, n_atoms, n_atoms, edge_vocab_size]
            mask (torch.Tensor): Mask for fake atoms, shape [batch_size, n_atoms], 1 mask, 0 keep
            given_x (torch.Tensor): Conditional coords, shape [batch_size, n_atoms, 3]
            given_h (torch.Tensor): Conditional atom type logits, shape [batch_size, n_atoms, n_feats]
            given_e (torch.Tensor): Cond bond type logits, shape [batch_size, n_atoms, n_atoms, edge_vocab_size]
            px (torch.Tensor): Input coordinates, shape [batch_size, n_protein_atoms, 3]
            ph (torch.Tensor): Invariant atom features, shape [batch_size, n_protein_atoms, n_feats]
            pmask (torch.Tensor): Mask for fake atoms, shape [batch_size, n_protein_atoms], 1 mask, 0 keep

        Returns:
            (predicted coordinates, atom type logits, bond logits, atom charges)
            All torch.Tensor, shapes:
                Coordinates: [batch_size, n_atoms, 3]
                Type logits: [batch_size, n_atoms, vocab_size],
                Bond logits: [batch_size, n_atoms, n_atoms, edge_vocab_size]
                Charge logits: [batch_size, n_atoms, 7] #not good!
        """
        self_cond = self.params["self_conditioning"]
        remove_com = self.params["remove_com"]

        if e is not None and self.params["d_edge"] is None:
            raise ValueError(
                "`e` was provided but the model was initialised with `d_edge` as None."
            )

        if e is None and self.params["d_edge"] is not None:
            raise ValueError(
                "The model was initialised with `d_edge` but no edge feats `e` were provided to forward."
            )

        if given_x is not None and not self_cond:
            raise ValueError(
                "`given_x` was provided but the model was initialised with `self_conditioning` as False."
            )

        if given_x is None and self_cond:
            raise ValueError(
                "The model was initialsed with `self_conditioning` but `given_x` was not provided."
            )

        if e is None and given_e is not None:
            raise ValueError("`e` must be provided if using bond conditioning.")

        mask = torch.zeros_like(x[..., 0], dtype=torch.bool) if mask is None else mask
        inv_mask = (~mask.unsqueeze(-1)).float()

        if (self.params["protein_conditioning"]) and (px is not None):
            x_and_px = torch.cat([x, px], dim=-2)
            inv_pmask = (~pmask.unsqueeze(-1)).float()
            inv_mask_x_and_px = torch.cat([inv_mask, inv_pmask], dim=-2)
            n_ligand_atoms = x.shape[1]
            # compute connection based on nearest neighbour
            # batch_size x n_ligand_atoms x (n_ligand_atoms + n_protein_atoms)
            # adj_full = edges_from_nodes(x_and_px, node_mask=inv_mask_x_and_px[...,0], k=5)
            adj_full = edges_from_nodes(x_and_px, node_mask=inv_mask_x_and_px[..., 0])
            adj_matrix = adj_full[:, :n_ligand_atoms, :]
            px = self.pcoord_proj(px.unsqueeze(0).movedim(0, -1)).movedim(-1, 1)
            px = px * inv_pmask[:, None]
            pmask = pmask.unsqueeze(1)
            n_patoms = inv_pmask.sum(dim=-2)
            n_patoms = torch.clamp(n_patoms, min=1.0)
            psize_emb = self.psize_emb(n_patoms.long()).expand(-1, ph.size(1), -1)
            ph = torch.cat((ph, psize_emb), dim=-1)
            ph = self.pfeat_proj(ph)
        else:
            adj_matrix = edges_from_nodes(x, node_mask=inv_mask[..., 0])

        # All projections
        n_atoms = inv_mask.sum(dim=-2)
        n_atoms = torch.clamp(n_atoms, min=1.0)
        size_emb = self.size_emb(n_atoms.long()).expand(-1, h.size(1), -1)
        h = torch.cat((h, size_emb), dim=-1)
        if given_h is not None:
            h = torch.cat((h, given_h), dim=-1)
        h = self.feat_proj(h)

        if e is not None:
            e = e.float()
            e = torch.cat((e, given_e), dim=-1) if given_e is not None else e
            e = self.edge_in_proj(e)

        x = torch.stack((x, given_x)) if self_cond else x.unsqueeze(0)
        x = self.coord_proj(x.movedim(0, -1)).movedim(-1, 1)
        x = x * inv_mask[:, None]

        # Update coords and node feats using the semla layers
        mask = mask.unsqueeze(1)
        for semla_layer in self.semla_layers:
            x, h, e = semla_layer(x, h, adj_matrix, e, mask, px, ph, pmask)
        # Apply a final feedforward block and project coord sets to single coord set
        xc, hc = self.final_ff_block(x, h, mask)
        x = xc - x  # ugly fix
        h = hc - h  # ugly fix
        x = self.coord_norm(x, mask)
        x = self.coord_head(x.transpose(1, -1))
        x = x.transpose(1, -1).squeeze(1)
        if self.params["bond_refine"]:
            e = self.refine_layer(x, h, e, mask[:, 0, :])
        x_hat = x
        h_hat = self.feat_norm(h)
        # otherwise e_hat is None when d_edge is given, i assume this is not intended?
        if self.params["d_edge"] is not None:
            e_hat = self.bond_norm(e)
        else:
            e_hat = None
        # otherwise com has shape batch x batch x 3
        if remove_com:
            com = (x_hat * inv_mask).sum(dim=-2, keepdim=True) / n_atoms.unsqueeze(-1)
            x_hat = (x_hat - com) * inv_mask
        else:
            x_hat = x_hat * inv_mask

        atom_logits = self.atom_classifier_head(h_hat)
        charge_logits = self.charge_classifier_head(h_hat)

        # If we are predicting edges ensure that the matrix is symmetrical
        if e_hat is not None:
            e_hat = e_hat + e_hat.transpose(1, 2)
            e_logits = self.edge_out_proj(e_hat)
        else:
            e_logits = None

        return x_hat, atom_logits, charge_logits, e_logits

    def get_state_dict(self):
        return {
            "class_name": self.__class__.__name__,
            "class_module": self.__class__.__module__,
            "params": self.params,
            "state": self.state_dict(),
        }

    @classmethod
    def load_from_state_dict(cls, save_dict, device, instance=None):
        allowed_missing_keys = set(
            [
                "semla_layers._.pfeed_forward.phi_equi.set_weights",
                "semla_layers._.pfeed_forward.phi_inv.norm.0.weight",
                "semla_layers._.pfeed_forward.phi_inv.norm.0.bias",
                "semla_layers._.pfeed_forward.invariant_mlp.Phi.0.weight",
                "semla_layers._.pfeed_forward.invariant_mlp.Phi.0.bias",
                "semla_layers._.pfeed_forward.invariant_mlp.Phi.2.weight",
                "semla_layers._.pfeed_forward.invariant_mlp.Phi.2.bias",
                "semla_layers._.pfeed_forward.equivariant_mlp.Psi.0.weight",
                "semla_layers._.pfeed_forward.equivariant_mlp.Psi.0.bias",
                "semla_layers._.pfeed_forward.equivariant_mlp.Psi.2.weight",
                "semla_layers._.pfeed_forward.equivariant_mlp.Psi.2.bias",
                "semla_layers._.pfeed_forward.equivariant_mlp.W1.weight",
                "semla_layers._.pfeed_forward.equivariant_mlp.W2.weight",
                "semla_layers._.graph_attention.simple_lp_message.phi_equi.set_weights",
                "semla_layers._.graph_attention.simple_lp_message.phi_inv.norm.0.weight",
                "semla_layers._.graph_attention.simple_lp_message.phi_inv.norm.0.bias",
                "semla_layers._.graph_attention.simple_lp_message.project.1.weight",
                "semla_layers._.graph_attention.simple_lp_message.project.1.bias",
                "psize_emb.weight",
                "pcoord_proj.weight",
                "pfeat_proj.0.weight",
                "pfeat_proj.0.bias",
                "pfeat_proj.2.weight",
                "pfeat_proj.2.bias",
            ]
        )
        if instance is None:
            model = cls(**save_dict["params"])
        else:
            model = instance
        missing_keys, unexpected_keys = model.load_state_dict(
            save_dict["state"], strict=False
        )

        if len(unexpected_keys) > 0:
            raise ValueError("Unexpeced keys in checkpoint")

        for k in missing_keys:
            if k.startswith("semla_layers"):
                k2 = ".".join([k.split(".")[0], "_"] + k.split(".")[2:])
                if k2 not in allowed_missing_keys:
                    raise ValueError(f"Missing key {k2} not allowed")

        return model.to(device)

    @classmethod
    def load_from_file(cls, file, device, instance=None):
        save_dict = torch.load(file, map_location="cpu", weights_only=False)
        model = cls.load_from_state_dict(save_dict, device, instance=instance)
        return model
