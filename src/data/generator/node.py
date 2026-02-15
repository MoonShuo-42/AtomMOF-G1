import pickle
import random
from pathlib import Path
from typing import Optional, Tuple

from src.utils.mol_utils import convert_xyz_to_rdkit_mol, extract_features_from_rdkit_mol
from src.data.generator.base import BlockGenerator
from src.data.types import Fragment, Block, MetalLibrary, MetalNodeVariant


class NodeGenerator(BlockGenerator):
    def __init__(self, metal_path: str):
        self.metal_library = self._load_metal_library(metal_path)

    def _load_metal_library(self, metal_path: str) -> MetalLibrary:
        metal_path = Path(metal_path)
        with open(metal_path, "rb") as f:
            return pickle.load(f)
    
    def generate_block_from_fragment(self, fragment: Fragment) -> Optional[Block]:
        # TODO: Sample variant from library

        # Extract chemical features
        mol = convert_xyz_to_rdkit_mol(
            atom_types=fragment.atomic_numbers,
            positions=fragment.cart_coords_contiguous,
            add_bonds=False # TODO: Consider changing this to True?
        )
        chem_feats = extract_features_from_rdkit_mol(mol)
        
        return Block(
            block_type=fragment.block_type,
            cart_coords=fragment.cart_coords_contiguous,
            atomic_numbers=fragment.atomic_numbers,
            **chem_feats
        )