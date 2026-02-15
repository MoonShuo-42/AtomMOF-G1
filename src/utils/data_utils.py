import numpy as np
import itertools
import os
import torch
import copy

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis import local_env

from omegaconf import OmegaConf
import hydra

import pytorch_lightning as pl

SUPERCELLS = torch.FloatTensor(list(itertools.product((-1, 0, 1), repeat=3)))
EPSILON = 1e-5


def repeat_interleave(repeats):
    """
    Returns a tensor of indices where each index i is repeated repeats[i] times.
    
    Example:
        repeats = torch.tensor([4, 2])
        # Output: tensor([0, 0, 0, 0, 1, 1])
    """
    outs = torch.repeat_interleave(
        input=torch.arange(len(repeats), device=repeats.device),
        repeats=repeats
    )
    return outs


def logmod(x):
    return torch.sign(x) * torch.log(x.abs() + 1)


def reverse_logmod(x):
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1.0, 1.0)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack(
        [
            lengths[:, 0] * sins[:, 1],
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 0] * coses[:, 1],
        ],
        dim=1,
    )
    vector_b = torch.stack(
        [
            -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
            lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
            lengths[:, 1] * coses[:, 0],
        ],
        dim=1,
    )
    vector_c = torch.stack(
        [
            torch.zeros(lengths.size(0), device=lengths.device),
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 2],
        ],
        dim=1,
    )

    return torch.stack([vector_a, vector_b, vector_c], dim=1)


def compute_volume(batch_lattice):
    """Compute volume from batched lattice matrix

    batch_lattice: (N, 3, 3)
    """
    vector_a, vector_b, vector_c = torch.unbind(batch_lattice, dim=1)
    return torch.abs(
        torch.einsum("bi,bi->b", vector_a, torch.cross(vector_b, vector_c, dim=1))
    )


def lengths_angles_to_volume(lengths, angles):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    return compute_volume(lattice)


def lattice_matrix_to_params(matrix):
    lengths = np.sqrt(np.sum(matrix**2, axis=1)).tolist()

    angles = np.zeros(3)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[i] = abs_cap(np.dot(matrix[j], matrix[k]) / (lengths[j] * lengths[k]))
    angles = np.arccos(angles) * 180.0 / np.pi
    a, b, c = lengths
    alpha, beta, gamma = angles
    return a, b, c, alpha, beta, gamma


def frac2cart(fcoord, cell):
    return fcoord @ cell


def cart2frac(coord, cell):
    # pylint: disable=E1102
    invcell = torch.linalg.inv(cell)
    return coord @ invcell


def compute_distance_matrix(cell, cart_coords, num_cells=1):
    pos = torch.arange(-num_cells, num_cells + 1, 1).to(cell.device)
    combos = (
        torch.stack(torch.meshgrid(pos, pos, pos, indexing="xy"))
        .permute(3, 2, 1, 0)
        .reshape(-1, 3)
        .to(cell.device)
    )
    shifts = torch.sum(cell.unsqueeze(0) * combos.unsqueeze(-1), dim=1)
    shifted = cart_coords.unsqueeze(1) + shifts.unsqueeze(0)
    dist = cart_coords.unsqueeze(1).unsqueeze(1) - shifted.unsqueeze(0)
    # +eps to avoid nan in differentiation
    dist = (dist.pow(2).sum(dim=-1) + 1e-32).sqrt()
    distance_matrix = dist.min(dim=-1)[0]
    return distance_matrix


def frac_to_cart_coords(
    frac_coords,
    lengths,
    angles,
    num_atoms,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
    pos = torch.einsum("bi,bij->bj", frac_coords, lattice_nodes)  # cart coords
    return pos


def cart_to_frac_coords(
    cart_coords,
    lengths,
    angles,
    num_atoms,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    # use pinv in case the predicted lattice is not rank 3
    inv_lattice = torch.linalg.pinv(lattice)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)
    frac_coords = torch.einsum("bi,bij->bj", cart_coords, inv_lattice_nodes)
    return frac_coords % 1.0


def get_wrapped_cart_coords(cart_coords, lengths, angles, num_atoms):
    frac_coords = cart_to_frac_coords(cart_coords, lengths, angles, num_atoms)
    return frac_to_cart_coords(frac_coords, lengths, angles, num_atoms)


class StandardScalerTorch(object):
    """Normalizes the targets of a dataset."""

    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float)
        self.means = torch.mean(X, dim=0)
        # https://github.com/pytorch/pytorch/issues/29372
        self.stds = torch.std(X, dim=0, unbiased=False) + EPSILON

    def transform(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        elif not isinstance(X, torch.FloatTensor):
            X = X.float()
        return (X - self.means) / self.stds

    def inverse_transform(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float)
        elif not isinstance(X, torch.FloatTensor):
            X = X.float()
        return X * self.stds + self.means

    def match_device(self, tensor):
        if self.means.device != tensor.device:
            self.means = self.means.to(tensor.device)
            self.stds = self.stds.to(tensor.device)

    def copy(self):
        return StandardScalerTorch(
            means=self.means.clone().detach(), stds=self.stds.clone().detach()
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"means: {self.means.tolist()}, "
            f"stds: {self.stds.tolist()})"
        )


def get_scaler_from_data_list(data_list, key):
    if isinstance(data_list[0][key], torch.Tensor):
        if len(data_list[0][key].shape) == 2:
            targets = torch.cat([d[key] for d in data_list], dim=0)
        else:
            targets = torch.stack([d[key] for d in data_list], dim=0)
    else:
        targets = torch.tensor([d[key] for d in data_list])
    scaler = StandardScalerTorch()
    scaler.fit(targets)
    return scaler


def mard(targets, preds):
    """Mean absolute relative difference."""
    assert torch.all(targets > 0.0)
    return torch.mean(torch.abs(targets - preds) / targets)


class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.
    When it is fit on a dataset, the :class:`StandardScaler` learns the
        mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the
        means and divides by the standard deviations.
    """

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.
        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(
            np.isnan(self.means), np.zeros(self.means.shape), self.means
        )
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

        return self

    def transform(self, X):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.
        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan
        )

        return transformed_with_none

    def inverse_transform(self, X):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.
        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan
        )

        return transformed_with_none