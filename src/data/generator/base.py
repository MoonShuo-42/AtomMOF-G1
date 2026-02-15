from abc import ABC, abstractmethod
from typing import Optional
from src.data.types import Fragment, Block


class BlockGenerator(ABC):
    @abstractmethod
    def generate_block_from_fragment(self, fragment: Fragment) -> Optional[Block]:
        pass

    def extract_chem_feats_from_rdkit_mol(self, mol):
        pass

    # TODO: Add generate_block_from_smiles