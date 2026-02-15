import numpy as np
import torch
from typing import Dict

from src.data import const
from src.data.types import Structure
from src.models.modules.utils import center_random_augmentation


class Featurizer:
    """
    Converts a Structure dataclass into dictionary of PyTorch tensors for model input.
    """
    def __init__(self):
        pass

    def _apply_augmentations(self, ref_pos: torch.Tensor, block_index: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Applies random augmentation to each block's local coordinates."""
        for i in range(torch.max(block_index) + 1):
            included = (block_index == i)
            if torch.sum(included) > 0 and torch.any(mask[included]):
                # Apply augmentation to the atoms of the current molecule
                ref_pos[included] = center_random_augmentation(
                    ref_pos[included][None], mask[included][None], centering=True
                )[0]
        return ref_pos

    def process(self, data: Structure) -> Dict[str, torch.Tensor]:
        ref_pos_list, ref_element_list, ref_charge_list, ref_chirality_list = [], [], [], []
        block_index_list, block_type_list = [], []
        bond_index_list, bond_type_list = [], []

        # Iterate over blocks to extract features
        current_atom_idx = 0
        for i, block in enumerate(data.blocks):
            num_atoms = block.atomic_numbers.shape[0]
            if num_atoms == 0:
                continue

            # Atom-level features
            ref_pos_list.append(block.cart_coords)
            ref_element_list.append(block.atomic_numbers)
            ref_charge_list.append(block.formal_charges)
            ref_chirality_list.append(block.chirality_tags)
            
            # Block-level features
            block_index_list.extend([i] * num_atoms)
            block_type_id = const.block_type_ids[block.block_type]
            block_type_list.extend([block_type_id] * num_atoms)

            # Bond features
            shifted_bond_indices = block.bond_indices + current_atom_idx
            bond_index_list.append(shifted_bond_indices)
            bond_type_list.append(block.bond_types)
            
            # Update current atom index
            current_atom_idx += num_atoms

        # Concatenate features
        ref_pos = np.concatenate(ref_pos_list)
        ref_element = np.concatenate(ref_element_list)
        ref_charge = np.concatenate(ref_charge_list)
        ref_chirality = np.concatenate(ref_chirality_list)
        block_index = np.array(block_index_list, dtype=int)
        block_type = np.array(block_type_list, dtype=int)

        # Process bond info
        if len(bond_index_list) > 0:
            bond_indices = np.concatenate(bond_index_list, axis=1)  # (2, total_num_bonds)
            bond_types = np.concatenate(bond_type_list)             # (total_num_bonds,)
        else:
            bond_indices = np.zeros((2, 0), dtype=int)
            bond_types = np.zeros((0,), dtype=int)

        # Atom-token mapping (TODO: Implement proper tokenization)
        atom_to_token = np.eye(ref_pos.shape[0])
        token_to_rep_atom = np.eye(ref_pos.shape[0])

        # Convert to PyTorch tensors
        ref_pos = torch.from_numpy(ref_pos).float()
        ref_element = torch.from_numpy(ref_element).long()
        ref_charge = torch.from_numpy(ref_charge).float()
        ref_chirality = torch.from_numpy(ref_chirality).long()
        block_index = torch.from_numpy(block_index).long()
        block_type = torch.from_numpy(block_type).long()
        bond_indices = torch.from_numpy(bond_indices).long()
        bond_types = torch.from_numpy(bond_types).long()
        atom_to_token = torch.from_numpy(atom_to_token).float()
        token_to_rep_atom = torch.from_numpy(token_to_rep_atom).float()

        # Apply augmentations to reference positions
        mask = torch.ones_like(ref_element).bool()
        ref_pos = self._apply_augmentations(ref_pos, block_index, mask)

        return {
            "ref_pos": ref_pos,                     # (num_atoms, 3)
            "ref_element": ref_element,             # (num_atoms,)
            "ref_charge": ref_charge,               # (num_atoms,)
            "ref_chirality": ref_chirality,         # (num_atoms,)
            "block_index": block_index,             # (num_atoms,)
            "block_type": block_type,               # (num_atoms,)
            "bond_indices": bond_indices,           # (2, num_bonds)
            "bond_types": bond_types,               # (num_bonds,)
            "atom_to_token": atom_to_token,         # (num_atoms, num_tokens)
            "token_to_rep_atom": token_to_rep_atom, # (num_tokens, num_atoms)

            # Ground-truth data
            'cart_coords': torch.from_numpy(data.cart_coords).float(),
            'frac_coords': torch.from_numpy(data.frac_coords).float(),
            'lattice': torch.from_numpy(data.lattice).float(),
            'cell': torch.from_numpy(data.cell).float(),
        }