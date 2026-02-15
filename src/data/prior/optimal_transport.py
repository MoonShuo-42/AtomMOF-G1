import torch
from scipy.optimize import linear_sum_assignment


def align_coords(coords_0, coords_1, atom_types, invalid_cost=1e6):
    """
    Args:
        coords_0: [N, 3] torch.Tensor, the prior coordinates
        coords_1: [N, 3] torch.Tensor, the target coordinates
        atom_types: [N] torch.Tensor, atom types for both coords_0 and coords_1
    Returns:
        coords_1: [N, 3] torch.Tensor where rows of coords_1 are permuted to best match coords_0
    """
    assert coords_0.shape == coords_1.shape, "coords_0 and coords_1 must have the same shape"

    # Compute pairwise squared distances
    cost_matrix = torch.cdist(coords_0, coords_1)**2  # shape [N, N]

    # Mask out incompatible atom types
    type_mask = atom_types[:, None] == atom_types[None, :]  # shape [N, N], bool
    cost_matrix[~type_mask] = invalid_cost  # large penalty for invalid matches

    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

    # Apply permutation to coords_1 (to match coords_0)
    permutation = torch.zeros_like(torch.from_numpy(col_ind))
    permutation[row_ind] = torch.from_numpy(col_ind)
    coords_1_permuted = coords_1[permutation.to(coords_1.device)]

    return coords_1_permuted