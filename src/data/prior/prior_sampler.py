import abc
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
from torch.distributions import Distribution, LogNormal, Uniform

from src.models.modules.utils import center_random_augmentation


# Abstract Base Class for Individual Samplers

class Sampler(abc.ABC):
    """Abstract base class for a single prior sampler."""
    @abc.abstractmethod
    def sample(self, data: dict) -> dict:
        """Samples a prior for a specific data type and returns it in a dictionary."""
        pass


# Concrete Sampler Implementations 

@dataclass
class CartCoordsSampler(Sampler):
    """Samples a prior for Cartesian coordinates."""
    variance: float = 3.0
    center: bool = True

    def sample(self, data: dict) -> dict:
        """Samples a prior for cartesian coordinates from a Gaussian distribution."""
        device = data['cart_coords'].device

        cart_coords_0 = self.variance * torch.randn_like(data['cart_coords'], device=device)

        # Center prior
        if self.center:
            if data['cart_coords'].ndim == 3:
                cart_coords_0 = center_random_augmentation(
                    cart_coords_0, data['atom_mask'], augmentation=False, centering=True
                )
            else:
                cart_coords_0 = cart_coords_0 - cart_coords_0.mean(dim=0, keepdim=True)
        
        return {'cart_coords_0': cart_coords_0}


@dataclass
class LatticeSampler(Sampler):
    """
    Samples a prior for lattice parameters.
    """
    length_loc: Tuple[float, float, float] = (2.50, 2.50, 2.50)
    length_scale: Tuple[float, float, float] = (0.30, 0.30, 0.30)
    angle_range: Tuple[float, float] = (60.0, 120.0)

    _log_normal_dist: Distribution = field(init=False)
    _uniform_dist: Distribution = field(init=False)

    def __post_init__(self):
        self._log_normal_dist = LogNormal(
            torch.tensor(self.length_loc),
            torch.tensor(self.length_scale)
        )
        self._uniform_dist = Uniform(
            low=self.angle_range[0],
            high=self.angle_range[1],
        )

    def sample(self, data: dict) -> dict:
        """
        Samples priors for lattice parameters, handling both single and batched data.
        """
        # Get a reference tensor to determine batch size and device
        ref_tensor = data['cart_coords']
        device = ref_tensor.device

        # Check if the input is batched (e.g., shape [B, N, 3])
        is_batched = ref_tensor.ndim == 3

        if is_batched:
            batch_size = ref_tensor.shape[0]
            # Provide the batch size to the sample() method
            lengths = self._log_normal_dist.sample((batch_size,))
            angles = self._uniform_dist.sample((batch_size, 3))
        else:
            # Sample a single instance
            lengths = self._log_normal_dist.sample()
            angles = self._uniform_dist.sample((3,))

        # Concatenate and ensure the tensor is on the correct device
        lattice_0 = torch.cat([lengths, angles], dim=-1).to(device)

        return {'lattice_0': lattice_0}


# Main PriorSampler Class

class PriorSampler:
    """
    Manages a collection of individual sampler modules.
    This class iterates through its list of samplers and aggregates their results.
    """
    # The __init__ is now much simpler and more flexible!
    def __init__(self, samplers: List[Sampler]):
        self.samplers = samplers

    def sample(self, data: dict) -> dict:
        """
        Runs all configured samplers on the data and returns a combined dictionary
        of all sampled priors.
        """
        prior_data = {}
        for sampler in self.samplers:
            prior_data.update(sampler.sample(data))
        return prior_data