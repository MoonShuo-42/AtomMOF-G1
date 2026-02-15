from typing import Union, Dict, Any, Tuple

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from openbabel import openbabel as ob
from openbabel import pybel

from src.data import const
from src.data.const import METALS

METALS = torch.tensor(METALS).long()


def convert_xyz_to_pybel_mol(
    atom_types: Union[torch.Tensor, np.ndarray], 
    positions: Union[torch.Tensor, np.ndarray], 
    add_bonds: bool = True,
) -> pybel.Molecule:
    """
    Create an OpenBabel molecule object from atom types and positions.
    """
    if isinstance(atom_types, torch.Tensor):
        atom_types = atom_types.numpy()
        positions = positions.numpy()
    
    mol = pybel.ob.OBMol()
    for atom_type, position in zip(atom_types, positions):
        atom = pybel.ob.OBAtom()
        atom.SetAtomicNum(int(atom_type))
        atom.SetVector(*position.tolist())
        mol.AddAtom(atom)

    # Add bonds for organic molecules
    if add_bonds:
        # Determine connectivity based on distance
        mol.ConnectTheDots()
        
        # Identify aromatic nitrogens
        for atom in ob.OBMolAtomIter(mol):
            if "N" in atom.GetType() and atom.IsInRing():
                atom.SetAromatic()
        
        # Assign bond orders
        mol.PerceiveBondOrders()
    return pybel.Molecule(mol)


def convert_xyz_to_rdkit_mol(
    atom_types: Union[torch.Tensor, np.ndarray],
    positions: Union[torch.Tensor, np.ndarray],
    add_bonds: bool = True,
) -> Mol:
    """
    Create an RDKit molecule object from atom types and positions.
    """
    ob_mol = convert_xyz_to_pybel_mol(atom_types, positions, add_bonds=add_bonds)
    rd_mol = convert_pybel_to_rdkit_mol(ob_mol)
    return rd_mol


def convert_xyz_to_canonical_smiles(
    atom_types: Union[torch.Tensor, np.ndarray], 
    positions: Union[torch.Tensor, np.ndarray], 
    add_bonds: bool = True,
) -> str:
    mol = convert_xyz_to_pybel_mol(atom_types, positions, add_bonds=add_bonds)
    return mol.write("can").strip()


def convert_pybel_to_rdkit_mol(ob_mol: pybel.Molecule) -> Mol:
    """
    Convert an OpenBabel molecule to an RDKit molecule.
    """
    # NOTE: Convert using mol/mol2 format to preserve atom orders (SMILES may change order)
    rd_mol = Chem.MolFromMol2Block(ob_mol.write("mol2"), removeHs=False)
    if rd_mol is None:
        rd_mol = Chem.MolFromMolBlock(ob_mol.write("mol"), removeHs=False)
    return rd_mol


def convert_rdkit_to_pybel_mol(rd_mol: Mol) -> pybel.Molecule:
    """
    Convert an RDKit molecule to an OpenBabel molecule.
    """
    return pybel.readstring("mol", Chem.MolToMolBlock(rd_mol))


def is_molecule_linear(ob_mol: pybel.Molecule, angle_tolerance=10.0):
    """
    Check whether a molecule is linear. Adapted from _is_molecule_linear function of InchiMolAtom Mapper class (pymatgen). 

    Args:
        ob_mol: OpenBabel pybel.Molecule object.
        angle_tolerance: Tolerance for angle deviation from 180 degrees to consider a molecule linear.
    
    Returns:
        bool: True if molecule is linear, False otherwise.
    """
    if isinstance(ob_mol, Chem.rdchem.Mol):
        ob_mol = convert_rdkit_to_pybel_mol(ob_mol)

    if ob_mol.OBMol.NumAtoms() < 3:
        return True
    a1 = ob_mol.OBMol.GetAtom(1)
    a2 = ob_mol.OBMol.GetAtom(2)
    for idx in range(3, ob_mol.OBMol.NumAtoms() + 1):
        angle = float(ob_mol.OBMol.GetAtom(idx).GetAngle(a2, a1))
        if angle < 0.0:
            angle = -angle
        if angle > 90.0:
            angle = 180.0 - angle
        if angle > angle_tolerance:
            return False
    return True


def is_metal_bb(bb_mol: Chem.rdchem.Mol) -> bool:
    """
    Returns:
    - bool: True if the molecule contains a metal atom, False otherwise
    """
    if isinstance(bb_mol, Chem.rdchem.Mol):
        bb_atom_types = torch.tensor([atom.GetAtomicNum() for atom in bb_mol.GetAtoms()]).long()
    elif isinstance(bb_mol, torch.Tensor):
        bb_atom_types = bb_mol
    else:
        raise ValueError(f"Invalid type: {type(bb_mol)}")
    return torch.any(torch.isin(bb_atom_types, METALS)).item()


def bb_to_rdkit_mols(atom_types, coords, bb_num_vec):
    atom_type_list = torch.split(atom_types, bb_num_vec.tolist())
    coords_list = torch.split(coords, bb_num_vec.tolist())
    is_metal_list = [is_metal_bb(bb_atom_types) for bb_atom_types in atom_type_list]
    bb_mols = []
    for bb_atom_types, bb_coords, is_metal in zip(atom_type_list, coords_list, is_metal_list):
        bb_mol = convert_xyz_to_pybel_mol(bb_atom_types, bb_coords, is_metal)
        bb_mol = convert_pybel_to_rdkit_mol(bb_mol)
        bb_mols.append(bb_mol)
    return bb_mols, torch.tensor(is_metal_list).bool()


# from Boltz
def compute_3d_conformer(mol: Mol, version: str = "v3") -> bool:
    """Generate 3D coordinates using EKTDG method.

    Taken from `pdbeccdutils.core.component.Component`.

    Parameters
    ----------
    mol: Mol
        The RDKit molecule to process
    version: str, optional
        The ETKDG version, defaults to v3

    Returns
    -------
    bool
        Whether computation was successful.

    """
    if version == "v3":
        options = AllChem.ETKDGv3()
    elif version == "v2":
        options = AllChem.ETKDGv2()
    else:
        options = AllChem.ETKDGv2()

    options.clearConfs = False
    conf_id = -1

    try:
        conf_id = AllChem.EmbedMolecule(mol, options)

        if conf_id == -1:
            print(
                f"WARNING: RDKit ETKDGv3 failed to generate a conformer for molecule "
                f"{Chem.MolToSmiles(AllChem.RemoveHs(mol))}, so the program will start with random coordinates. "
                f"Note that the performance of the model under this behaviour was not tested."
            )
            options.useRandomCoords = True
            conf_id = AllChem.EmbedMolecule(mol, options)

        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)

    except RuntimeError:
        pass  # Force field issue here
    except ValueError:
        pass  # sanitization issue here

    if conf_id != -1:
        conformer = mol.GetConformer(conf_id)
        conformer.SetProp("name", "Computed")
        conformer.SetProp("coord_generation", f"ETKDG{version}")

        return True

    return False


def extract_features_from_rdkit_mol(mol: Mol) -> Dict[str, np.ndarray]:
    """
    Extracts atomic numbers, charges, chirality, and bond info from an RDKit Mol.
    Returns a dictionary compatible with the Block dataclass.
    """
    # Atom Features
    charges_list = []
    chirality_list = []
    
    unk_chirality_tags = const.chirality_type_ids[const.unk_chirality_type]

    for atom in mol.GetAtoms():
        charges_list.append(atom.GetFormalCharge())
        
        # Map RDKit ChiralTag enum to your integer ID
        chiral_name = atom.GetChiralTag().name
        chirality_list.append(const.chirality_type_ids.get(chiral_name, unk_chirality_tags))

    # Bond Features
    bond_indices_list = []
    bond_types_list = []

    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        bond_type = const.bond_type_ids.get(str(bond.GetBondType()), 0)
        
        bond_indices_list.append((start_idx, end_idx))
        bond_types_list.append(bond_type)

    # Formatting Arrays
    if len(bond_indices_list) > 0:
        bond_indices = np.array(bond_indices_list).T  # Shape: (2, num_bonds)
        bond_types = np.array(bond_types_list)        # Shape: (num_bonds,)
    else:
        bond_indices = np.empty((2, 0), dtype=int)
        bond_types = np.empty((0,), dtype=int)

    return {
        "formal_charges": np.array(charges_list, dtype=int),
        "chirality_tags": np.array(chirality_list, dtype=int),
        "bond_indices": bond_indices,
        "bond_types": bond_types,
    }