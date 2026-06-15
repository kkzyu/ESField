import numpy as np
import torch

from .utils import dense2scatter, scatter2dense
from .utils import pairwise_concat


class Phi_e(torch.nn.Module):
    """
    As implemented in EDM paper
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(feature_dim * 2 + 2, feature_dim)
        self.silu1 = torch.nn.SiLU()
        self.linear2 = torch.nn.Linear(feature_dim, feature_dim)
        self.silu2 = torch.nn.SiLU()

    def forward(self, x):
        # x contains the (concatenated) features h_i, h_j, d(x_i, x_j)^2, a_ij
        h = x
        h = self.linear1(h)
        h = self.silu1(h)
        h = self.linear2(h)
        h = self.silu2(h)
        return h


class Phi_inf(torch.nn.Module):
    """
    As implemented in EDM paper
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(feature_dim, 1)
        self.sigmoid1 = torch.nn.Sigmoid()

    def forward(self, x):
        h = x
        h = self.linear1(h)
        h = self.sigmoid1(h)
        return h


class Phi_h(torch.nn.Module):
    """
    As implemented in EDM paper
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(feature_dim * 2, feature_dim)
        self.silu1 = torch.nn.SiLU()
        self.linear2 = torch.nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        # x contains the (concatenated) features h_i, h_j, d(x_i, x_j)^2, a_ij
        h = x
        h = self.linear1(h)
        h = self.silu1(h)
        h = self.linear2(h)
        return h


class Phi_x(torch.nn.Module):
    """
    As implemented in EDM paper
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(feature_dim * 2 + 2, feature_dim)
        self.silu1 = torch.nn.SiLU()
        self.linear2 = torch.nn.Linear(feature_dim, feature_dim)
        self.silu2 = torch.nn.SiLU()
        self.linear3 = torch.nn.Linear(feature_dim, 1)

    def forward(self, x):
        # x contains the (concatenated) features h_i, h_j, d(x_i, x_j)^2, a_ij
        h = x
        h = self.linear1(h)
        h = self.silu1(h)
        h = self.linear2(h)
        h = self.silu2(h)
        h = self.linear3(h)
        return h


class PosLayer(torch.nn.Module):
    def __init__(self, feature_dim, normalization):
        super().__init__()
        self.phi_x = Phi_x(feature_dim)
        self.normalization = normalization

    def forward(self, x, hx, x_mask, y, hy, y_mask, a=None):
        """
        x: B x Max_cardinality x 3
        h: B x Max_cardinality x feature_dim
        mask: B x Max_cardinality - 0 means keep, 1 means ignore
        """
        rel_distance = x.unsqueeze(2) - torch.cat((x, y), axis=1).unsqueeze(
            1
        )  # B x Max_cardinality_L x (Max_cardinality_L + Max_cardinality_P) x 3
        squared_distance = (rel_distance**2).sum(
            -1
        )  # B x Max_cardinality x Max_cardinality
        squared_distance = torch.clip(squared_distance, 0.0, 30.0)
        if self.normalization == "neighbors":
            x_den = (~x_mask).sum(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
            y_den = (~y_mask).sum(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
            rel_distance = rel_distance / (1 + x_den + y_den)
        elif self.normalization == "distance":
            # alternative way to improve train stability
            mask_distance = (squared_distance.unsqueeze(-1) >= 1e-4).detach()
            rel_distance = torch.where(
                mask_distance,
                rel_distance / (1 + (squared_distance.unsqueeze(-1) + 1e-8) ** 0.5),
                rel_distance,
            )
        else:
            raise ValueError(f"Normalization={self.normalization} not supported")

        # add squared distance to h
        # and a (which is dummy for the moment)
        if a is None:
            a = torch.zeros_like(squared_distance)

        #  B x Max_cardinality_L x (Max_cardinality_L + Max_cardinality_P)
        pair_mask = torch.logical_or(
            x_mask.unsqueeze(-1), torch.cat((x_mask, y_mask), dim=1).unsqueeze(1)
        )
        h_cat = pairwise_concat(hx, x_mask, hy, y_mask, squared_distance, a)

        h_cat_flat = h_cat.view(
            (h_cat.shape[0], -1) + h_cat.shape[3:]
        )  # B x (Max_cardinality^2) x (feature_dim*2 + 2)
        pair_mask_flat = pair_mask.view(
            (h_cat.shape[0], -1)
        )  # B x (Max_cardinality^2) x 1

        h_scatter, batch_idx = dense2scatter(
            h_cat_flat, pair_mask_flat
        )  # B_new x (feature_dim*2 + 2) / B_new x 1 (ex. 0,0,0,0,0,1,1,2,2,2,2,2,2,2,...)
        weights = self.phi_x(h_scatter)

        dense_weights_flat, _ = scatter2dense(weights, batch_idx, pair_mask)
        dense_weights = dense_weights_flat.view(h_cat.shape[:-1])
        x_delta = (rel_distance * dense_weights.unsqueeze(-1)).sum(-2)

        # second magic: Future work
        # torch.scatter_reduce(dummy, 0, batch_idx.unsqueeze(-1).expand(len(batch_idx),3), xs, reduce="sum")
        return x + x_delta


class FeatureLayer(torch.nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.phi_e = Phi_e(feature_dim)
        self.phi_inf = Phi_inf(feature_dim)
        self.phi_h = Phi_h(feature_dim)

    def forward(self, x, hx, x_mask, y, hy, y_mask, a=None):

        rel_distance = x.unsqueeze(2) - torch.cat((x, y), axis=1).unsqueeze(1)

        """
        x: B x Max_cardinality x 3
        h: B x Max_cardinality x feature_dim
        mask: B x Max_cardinality - 0 means keep, 1 means ignore
        """

        rel_distance = x.unsqueeze(2) - torch.cat((x, y), axis=1).unsqueeze(
            1
        )  # B x Max_cardinality x Max_cardinality x 3
        squared_distance = (rel_distance**2).sum(
            -1
        )  # B x Max_cardinality x Max_cardinality
        squared_distance = torch.clip(squared_distance, 0.0, 30.0)

        # add squared distance to h
        # and a (which is dummy for the moment)
        if a is None:
            a = torch.zeros_like(squared_distance)
        pair_mask = torch.logical_or(
            x_mask.unsqueeze(-1), torch.cat((x_mask, y_mask), dim=1).unsqueeze(1)
        )

        h_cat = pairwise_concat(hx, x_mask, hy, y_mask, squared_distance, a)

        h_cat_flat = h_cat.view(
            (h_cat.shape[0], -1) + h_cat.shape[3:]
        )  # B x (Max_cardinality^2) x (feature_dim*2 + 2)
        pair_mask_flat = pair_mask.view(
            (h_cat.shape[0], -1)
        )  # B x (Max_cardinality^2) x 1

        h_scatter, batch_idx = dense2scatter(
            h_cat_flat, pair_mask_flat
        )  # B_new x (feature_dim*2 + 2) / B_new x 1 (ex. 0,0,0,0,0,1,1,2,2,2,2,2,2,2,...)
        m = self.phi_e(h_scatter)  # m_ij as in the paper
        e = self.phi_inf(m)

        h_neighs, _ = scatter2dense(e * m, batch_idx, pair_mask)
        h_neighs = h_neighs.view(h_cat.shape[:-1] + (-1,))
        # mask self
        diag_mask = (
            (1 - torch.eye(h_neighs.shape[1], h_neighs.shape[2]).to(h_neighs))
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        h_neighs = (h_neighs * diag_mask).sum(-2)
        h_all = torch.cat((hx, h_neighs), -1)
        h_all_scatter, batch_idx = dense2scatter(h_all, x_mask)
        h_all_scatter = self.phi_h(h_all_scatter)
        h_all_scatter, _ = scatter2dense(h_all_scatter, batch_idx, x_mask)
        return h_all_scatter + hx


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = self.dim // 2
        emb = np.log(10000.0) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim) * -emb)
        self.emb = torch.nn.Parameter(
            torch.from_numpy(emb[None].astype(np.float32)), requires_grad=False
        )

    def forward(self, x):
        emb = x[:, None] * 4.0 * self.emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class EDMPocket(torch.nn.Module):
    def __init__(
        self, n_real, n_classes, feature_dim=256, L=1, normalization="neighbors"
    ):
        super().__init__()
        assert normalization in ["neighbors", "distance"]
        self.n_real = n_real
        if isinstance(n_classes, list):
            self.n_classes = n_classes[0]
        elif isinstance(n_classes, int):
            self.n_classes = n_classes
        else:
            raise ValueError("`n_classes` must be either list or int")
        self.feature_dim = feature_dim
        self.L = L
        self.normalization = normalization
        self.atom_embedder = torch.nn.Embedding(self.n_classes, feature_dim)
        self.atom_project = torch.nn.Linear(feature_dim, self.n_classes)
        self.time_embedding = SinusoidalPosEmb(feature_dim)

        self.pos_layers = torch.nn.ModuleList(
            [PosLayer(feature_dim, normalization) for _ in range(L)]
        )
        self.feat_layers = torch.nn.ModuleList(
            [FeatureLayer(feature_dim) for _ in range(L)]
        )

    def forward(self, x, y, t, x_mask, y_mask):
        # x: ligand
        # y: protein
        x1 = x[..., :3]
        h1 = self.atom_embedder(x[..., 3].long())

        y1 = y[..., :3]
        h2 = self.atom_embedder(y[..., 3].long())

        inv_x_mask = ~x_mask.unsqueeze(-1)
        x_temb = self.time_embedding(t).unsqueeze(1) * inv_x_mask
        h1 = h1 + x_temb

        inv_y_mask = ~y_mask.unsqueeze(-1)
        y_temb = (
            self.time_embedding(torch.zeros_like(t).to(t)).unsqueeze(1) * inv_y_mask
        )
        h2 = h2 + y_temb

        rel_distance = x1.unsqueeze(2) - torch.cat((x1, y1), axis=1).unsqueeze(
            1
        )  # B x Max_cardinality x Max_cardinality x 3
        a = (rel_distance**2).sum(-1)
        a = torch.clip(a, 0.0, 30.0)
        for fl, pl in zip(self.feat_layers, self.pos_layers):
            hn = fl(x1, h1, x_mask, y1, h2, y_mask, a=a)
            xn = pl(x1, h1, x_mask, y1, h2, y_mask, a=a)
            x1, h1 = xn, hn

        return [x1, torch.nn.functional.log_softmax(self.atom_project(h1), dim=-1)]

    def get_output_and_jacobian_old(self, x, yp, t, x_mask, yp_mask):
        x_pos = torch.clone(x[..., :3]).requires_grad_(True)
        x_cat = x[..., 3]
        x_shape = x_pos.shape
        size = np.prod(x_shape[:-1])
        dim = x_shape[-1]

        # Disable the gradient for the model's weights
        require_grad_params = {}
        for i, p in enumerate(self.parameters()):
            require_grad_params[i] = p.requires_grad
            p.requires_grad_(False)

        x_jacob = torch.zeros((size, dim, dim), dtype=x.dtype, device=x.device)
        y, y_cat = self(
            torch.cat((x_pos, x_cat[..., None]), -1), yp, t, x_mask, yp_mask
        )

        for i in range(dim):
            x_pos.grad = torch.zeros_like(x_pos)
            y.view(-1, dim)[:, i].sum().backward(retain_graph=True)
            x_jacob[:, i] = x_pos.grad.view(-1, dim)

        # Enable the gradient for the model's weights
        for i, p in enumerate(self.parameters()):
            p.requires_grad_(require_grad_params[i])
        return y.detach(), y_cat.detach(), x_jacob.view(x_shape + (dim,))

    def get_output_and_jacobian_slow(self, x, yp, t, x_mask, yp_mask):
        x_pos = torch.clone(x[..., :3]).requires_grad_(True)
        x_cat = x[..., 3]
        x_shape = x_pos.shape
        size = x_shape[0]
        dim = np.prod(x_shape[1:])

        # Disable the gradient for the model's weights
        require_grad_params = {}
        for i, p in enumerate(self.parameters()):
            require_grad_params[i] = p.requires_grad
            p.requires_grad_(False)

        x_jacob = torch.zeros((size, dim, dim), dtype=x.dtype, device=x.device)
        y, y_cat = self(
            torch.cat((x_pos, x_cat[..., None]), -1), yp, t, x_mask, yp_mask
        )
        yv = y.view(-1, dim).sum(0)
        for i in range(dim):
            x_pos.grad = torch.zeros_like(x_pos)
            yv[i].backward(retain_graph=True)
            x_jacob[:, i] = x_pos.grad.view(-1, dim)

        # Enable the gradient for the model's weights
        for i, p in enumerate(self.parameters()):
            p.requires_grad_(require_grad_params[i])
        return y.detach(), y_cat.detach(), x_jacob

    def get_output_and_jacobian(self, x, yp, t, x_mask, yp_mask):
        x_pos = x[..., :3]
        x_cat = x[..., 3]
        x_shape = x_pos.shape
        size = x_shape[0]
        dim = np.prod(x_shape[1:])

        # Disable the gradient for the model's weights

        x_jacob = torch.zeros((size, dim, dim), dtype=x.dtype, device=x.device)
        with torch.no_grad():
            y, y_cat = self(
                torch.cat((x_pos, x_cat[..., None]), -1), yp, t, x_mask, yp_mask
            )
            eps = 1e-3
            e_i = torch.zeros(dim, dtype=x_pos.dtype, device=x.device)
            for i in range(dim):
                e_i = e_i * 0.0
                e_i[i] = eps
                y_ei_p, _ = self(
                    torch.cat((x_pos + e_i.view((1, -1, 3)), x_cat[..., None]), -1),
                    yp,
                    t,
                    x_mask,
                    yp_mask,
                )
                # y_ei_m, _ = self(torch.cat((x_pos - e_i.view((1, -1, 3)) , x_cat[...,None]), -1), yp, t, x_mask, yp_mask)
                # x_jacob[:, i] = ( y_ei_p - y_ei_m).view(size, dim) / ( 2 * eps)
                x_jacob[:, i] = (y_ei_p - y).view(size, dim) / eps
        return y.detach(), y_cat.detach(), x_jacob

    @classmethod
    def load_from_file(cls, file_path, device: torch.device):
        """
        Loads a model from a single file
        :param file_path: Path to the saved model
        :return: An instance of the network
        """

        save_dict = torch.load(file_path, map_location="cpu")
        model = EDMPocket(**save_dict["params"])
        model.load_state_dict(save_dict["state"])
        return model.to(device)

    @classmethod
    def load_from_checkpoint(cls, save_dict, device: torch.device):
        """
        Loads a model from a checkpoint file
        :param save_dict: torch.load returned object
        :return: An instance of the network
        """
        model = EDMPocket(**save_dict["params"])
        model.load_state_dict(save_dict["state"])
        return model.to(device)

    def get_save_dict(self):
        """Returns the checkpoint object"""

        save_dict = dict(
            model_type=self.__class__.__name__,
            params={
                "n_real": self.n_real,
                "n_classes": self.n_classes,
                "feature_dim": self.feature_dim,
                "L": self.L,
                "normalization": self.normalization,
            },
            state=self.state_dict(),
        )

        return save_dict

    def save(self, path_to_file):
        """
        Saves the model to a file.
        :param path_to_file: Path to the file which the model will be saved to.
        """
        save_dict = self.get_save_dict()
        torch.save(save_dict, path_to_file)
