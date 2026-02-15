from typing import Optional

from nbformat import convert

from src.data.generator.base import BlockGenerator
from src.data.types import Fragment, Block
from src.utils.mol_utils import (
    compute_3d_conformer, 
    convert_xyz_to_rdkit_mol, 
    extract_features_from_rdkit_mol,
)


class RDKitBlockGenerator(BlockGenerator):
    def generate_block_from_fragment(self, fragment: Fragment) -> Optional[Block]:
        try:
            mol = convert_xyz_to_rdkit_mol(
                fragment.atomic_numbers, fragment.cart_coords_contiguous, add_bonds=True
            )

            # Generate conformer
            mol.RemoveAllConformers()
            if not compute_3d_conformer(mol):
                return None
            
            # Extract features
            local_coords = mol.GetConformer().GetPositions()
            chem_feats = extract_features_from_rdkit_mol(mol)

            return Block(
                block_type=fragment.block_type,
                cart_coords=local_coords,
                atomic_numbers=fragment.atomic_numbers,
                **chem_feats,
            )
        
        except Exception:
            return None
        
# TODO: OpenBabelBlockGenerator