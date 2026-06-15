import numpy as np
import torch


def generate_random_point_cloud_batch(batch_size, max_cardinality):
    sizes = np.random.choice(max_cardinality - 1, batch_size, replace=True) + 1
    x = torch.zeros((batch_size, max(sizes), 3))
    mask = torch.zeros((batch_size, max(sizes)), dtype=torch.bool)

    for i, s in enumerate(sizes):
        x[i, :s] = torch.rand(s, 3)
        mask[i, s:] = True
    return x, mask


def dense2scatter(x, mask):
    """
    Return flatten tensor containing only non-padding elements.

    Parameters
    ----------
    x : torch.tensor
        The input tensor of size B x C x dim
    mask: torch.tensor
        The mask boolean tensor of size B x C
        0 means keep, 1 means ignore.


    Returns
    -------
    y: torch.tensor
        The flattened tensor of size BC x dim
    batch_idx: torch.tensor
        A 1D tensor containing the cloud indices for the dense representation.
        Example: if x has size 2 x 3 x 256. If the first cloud contains 3 elements
        and the second cloud contains 2 elements batch_idx will contain [0, 0, 0, 1, 1]
    """
    batch_idx = (
        torch.arange(len(x), device=x.device).view(-1, 1).expand(x.shape[0], x.shape[1])
    )
    return x[~mask], batch_idx[~mask]


def scatter2dense(x, idx, mask=None, pad_val=0.0, sanity_check=False):
    """
    Return flatten tensor containing only non-padding elements.

    Parameters
    ----------
    x : torch.tensor
        The input tensor of size BC x dim
    idx: torch.tensor
        A 1D tensor containing the cloud indices for the dense representation.
        Example: if x has size 2 x 3 x 256. If the first cloud contains 3 elements
        and the second cloud contains 2 elements batch_idx will contain [0, 0, 0, 1, 1]
    mask: torch.tensor (default: None)
        The mask boolean tensor of size B x C
        0 means keep, 1 means ignore. If None an attempt to infer the mask will be made
        (never used in our code)
    pad_val: float (default: 0.)
        The padding value to populate the masked positions with
    sanity_check: bool (default: False)

    Returns
    -------
    y: torch.tensor
        The densified tensor of size B x C x dim
    mask: torch.tensor
        The mask boolean tensor of size B x C
        0 means keep, 1 means ignore.
    """
    if sanity_check:
        assert torch.all(torch.sort(idx).values == idx)

    infer_mask = mask is None
    if infer_mask:
        ivals, icounts = torch.unique(idx, return_counts=True)
        max_cardinality = torch.max(icounts)
        mask = torch.zeros(
            (len(ivals), max_cardinality), dtype=torch.bool, device=x.device
        )
        offset = 0
        all_idx = []
        for i, c in enumerate(icounts):
            tmp_idx = torch.arange(offset, offset + c)
            extra_idx = torch.ones(max_cardinality - c) * tmp_idx[-1]
            idx = torch.cat((tmp_idx, extra_idx)).long()
            all_idx.append(idx)
            mask[i, c:] = True
            offset = offset + c
        all_idx = torch.cat(all_idx)
        xd = x[all_idx].view((len(ivals), -1) + x.shape[1:])
        xd[mask] = pad_val
    else:
        xd = torch.empty(
            mask.shape[:3] + x.shape[-1:], dtype=x.dtype, device=x.device
        ).fill_(pad_val)
        xd[~mask] = x
    return xd, mask


def pairwise_concat(x, x_mask, *pairwise_features):
    """
    Return flatten tensor containing only non-padding elements.

    Parameters
    ----------
    x : torch.tensor
        The input tensor of size B x C x dim (ligand)
    x_mask: torch.tensor
        The mask boolean tensor of size B x C.
        0 means keep, 1 means ignore. If None an attempt to infer the mask will be made
        (never used in our code)
    pairwise_features: list of torch.tensors of size B x C x C

    Returns
    -------
    x_cat : torch.tensor
        The pairwise concatenated tensor of size B x C x C x (dim + dim + len(pairwise_features))
    """

    x_pad = torch.zeros_like(x)

    x1 = torch.cat(
        (x.unsqueeze(2), x_pad.unsqueeze(2)), -1
    )  # B x C x 1 x feature_dim*2
    x2 = torch.cat(
        (x_pad.unsqueeze(1), x.unsqueeze(1)), -1
    )  # B x 1 x C x feature_dim*2
    x12 = x1 + x2  # B x C x C x feature_dim*2
    pwf = []
    for p in pairwise_features:
        pwf.append(p.unsqueeze(-1))
    x_cat_tmp = torch.cat(
        (x12,) + tuple(pwf), -1
    )  # B x C x C x ( feature_dim*2 + len(pairwise_features))

    # Add padding
    pair_mask = ~torch.logical_or(x_mask.unsqueeze(-1), x_mask.unsqueeze(1)).unsqueeze(
        -1
    )
    return x_cat_tmp * pair_mask.float()
