"""
Extracts molecular building blocks from MOFid CIF outputs.
"""
import argparse
import json
import numpy as np
import lmdb
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

from tqdm import tqdm
from joblib import Parallel, delayed
from ase import Atoms
from ase.io import read
from ase.build import niggli_reduce
from ase.data import covalent_radii
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from fairchem.core.datasets import AseDBDataset

from src.utils.lmdb_utils import write_lmdb
from src.utils.mol_utils import convert_xyz_to_canonical_smiles
from src.data.types import BlockType, Fragment, DecomposedMOF


# Helper functions
def process_cif_jimage(string: str) -> List[int]:
    """Parses the jimage string from a CIF file into a lattice vector shift."""
    if string == ".":
        return [0, 0, 0]
    else:
        # Convention: 555 corresponds to no shift [0, 0, 0]
        return [int(string[2]) - 5, int(string[3]) - 5, int(string[4]) - 5]

def get_digits(string: str) -> int:
    """Extracts the integer from a string (e.g., 'C1' -> 1)."""
    return int("".join([c for c in string if c.isdigit()]))

def read_cif_bonds(cif_path: Path) -> Tuple[List[int], List[int], List[List[int]]]:
    """Reads bond information from a CIF file."""
    with open(cif_path, "r") as f:
        lines = f.read().splitlines()
    
    # No bonds
    if "_ccdc_geom_bond_type" not in "".join(lines):
        return [], [], []

    bond_start = next(i for i, line in enumerate(lines) if "_ccdc_geom_bond_type" in line) + 1

    from_index = []
    to_index = []
    to_jimage = []
    for idx in range(bond_start, len(lines)):
        v1, v2, _, jimage, _ = [x for x in lines[idx].split(" ") if x]
        v1 = get_digits(v1)
        v2 = get_digits(v2)
        jimage = process_cif_jimage(jimage)
        from_index.append(v1)
        to_index.append(v2)
        to_jimage.append(jimage)
    
    return from_index, to_index, to_jimage

def get_edge_index(atoms: Atoms) -> np.ndarray:
    """
    Generates edges for adsorbate atoms based on covalent radii.

    Args:
        atoms (ase.Atoms): Atoms object (with only adsorbate atoms for this use case).

    Returns:
        np.ndarray: Edge index array of shape (2, num_edges).
    """
    num_atoms = len(atoms)
    atomic_numbers = atoms.get_atomic_numbers()

    row_indices, col_indices = [], []
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            # Loop over unique pairs (i, j) with i < j (upper triangle)
            distance = atoms.get_distance(i, j, mic=True)
            radius_i = covalent_radii[atomic_numbers[i]]
            radius_j = covalent_radii[atomic_numbers[j]]
            # Standard cutoff with a tolerance (OpenBabel's default)
            cutoff = radius_i + radius_j + 0.45

            if distance < cutoff:
                row_indices.append(i)
                col_indices.append(j)

    # Create an undirected graph
    all_rows = row_indices + col_indices
    all_cols = col_indices + row_indices
    return np.array([all_rows, all_cols], dtype=np.int64)

def edge_index_to_connected_components(num_nodes: int, edge_index: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Args:
        num_nodes (int): Number of nodes in the graph.
        edge_index (np.ndarray): Edge index tensor of shape (2, num_edges)
            representing edges in the graph.
    Returns:
        num_components (int): Number of connected components in the graph.
        labels (np.ndarray): Array of shape (num_nodes,) containing the component label for each node.
    """
    adj_matrix = csr_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes)
    )
    num_components, labels = connected_components(
        csgraph=adj_matrix, directed=False, return_labels=True
    )
    return num_components, labels

def get_xyz_connected(frac_coords: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """
    Remaps fractional coordinates to be contiguous via BFS.
    """
    n_nodes = frac_coords.shape[0]
    shifted_frac_coords = frac_coords.copy()
    
    # Build adjacency list for efficient neighbor lookup
    adj = [[] for _ in range(n_nodes)]
    for sender, receiver in edge_index.T:
        adj[sender.item()].append(receiver.item())
        adj[receiver.item()].append(sender.item())

    # Standard BFS implementation
    queue = [0]
    visited = {0}
    head = 0
    
    while head < len(queue):
        current_node = queue[head]
        head += 1
        
        for neighbor in adj[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                
                # Calculate shift to bring neighbor close to the current node
                delta = shifted_frac_coords[current_node] - shifted_frac_coords[neighbor]
                shift = np.round(delta) # Shift by integer lattice vectors
                shifted_frac_coords[neighbor] += shift
                
                queue.append(neighbor)
                
    return shifted_frac_coords

def _process_components(atoms: Atoms, edge_index: np.ndarray, block_type: BlockType) -> List[Fragment]:
    """
    Processes graph components into Fragment dataclasses.
    """
    # Identify connected components
    num_components, component_labels = edge_index_to_connected_components(len(atoms), edge_index)
    fragments = []

    for component_id in range(num_components):
        # Identify which atoms belong to this component
        subset_indices = np.where(component_labels == component_id)[0]
        fragment_atoms = atoms[subset_indices]

        # Get original coordinates (raw from CIF)
        cart_coords_original = fragment_atoms.get_positions()
        
        # Compute contiguous (unwrapped) coordinates
        if len(fragment_atoms) == 1:
            # Single atoms are already contiguous
            cart_coords_contiguous = cart_coords_original.reshape(1, 3)
        else: 
            # Map subset indices (0..M) -> local index (Fragment 0..K)
            global_to_local_map = {
                global_idx: local_idx 
                for local_idx, global_idx in enumerate(subset_indices)
            }   

            # Isolate edges belonging to this component
            # We filter the full edge_index to keep only edges where nodes are in this component
            edge_mask = np.isin(edge_index[0], subset_indices)
            global_edge_subset = edge_index[:, edge_mask]

            # Re-index edges to local frame (0..K) for BFS
            map_func = np.vectorize(global_to_local_map.get)
            local_edge_index = map_func(global_edge_subset)

            # Perform BFS unwrap
            wrapped_frac_coords = fragment_atoms.get_scaled_positions()
            unwrapped_frac_coords = get_xyz_connected(wrapped_frac_coords, local_edge_index)
            cart_coords_contiguous = np.matmul(unwrapped_frac_coords, atoms.get_cell().array)

        # Extract smiles
        canonical_smiles = convert_xyz_to_canonical_smiles(
            atom_types=fragment_atoms.get_atomic_numbers(), 
            positions=cart_coords_contiguous,
            add_bonds=(block_type != "NODE")
        )
        
        # Create fragment
        frag = Fragment(
            block_type=block_type,
            canonical_smiles=canonical_smiles,
            cart_coords_contiguous=cart_coords_contiguous,
            cart_coords_original=cart_coords_original,
            atomic_numbers=fragment_atoms.get_atomic_numbers(),
            original_indices=subset_indices,
        )
        fragments.append(frag)
    return fragments

# Main extraction and processing functions
def extract_building_blocks(mofid_output_dir: Path, adsorbate_atoms: Optional[Atoms] = None) -> List[Fragment]:
    """
    Extracts all building blocks, identifies molecules, and makes them contiguous.
    """
    all_fragments: List[Fragment] = []

    # Map filename prefixes to BlockType Literal
    type_map : Dict[str, BlockType] = {
        "linkers": "LINKER",
        "nodes": "NODE",
        "node_bridges": "NODE",
        "free_solvent": "SOLVENT",
        "bound_solvent": "SOLVENT",
    }

    # Process framework blocks from MOFid outputs (nodes, linkers, solvents)
    for file_name, block_type in type_map.items():
        cif_path = mofid_output_dir / "MetalOxo" / f"{file_name}.cif"
        if not cif_path.exists():
            continue
        try:
            atoms = read(str(cif_path))
            from_index, to_index, _ = read_cif_bonds(cif_path)

            if from_index:  
                # Block has defined bonds
                edge_index = np.array([from_index, to_index], dtype=np.int64)
                edge_index = np.hstack([edge_index, np.flip(edge_index, axis=0)]) # Make undirected

                frags = _process_components(atoms, edge_index, block_type)
                all_fragments.extend(frags)
            else:  
                # No bonds in CIF -> Treat each atom as a separate Fragment
                for idx, atom in enumerate(atoms):
                    canonical_smiles = convert_xyz_to_canonical_smiles(
                        atom_types=np.array([atom.number]), 
                        positions=atom.position.reshape(1, 3),
                        add_bonds=False
                    )

                    frag = Fragment(
                        block_type=block_type,
                        canonical_smiles=canonical_smiles,
                        cart_coords_contiguous=atom.position.reshape(1, 3),
                        cart_coords_original=atom.position.reshape(1, 3),
                        atomic_numbers=np.array([atom.number]),
                        original_indices=np.array([idx]),
                    )
                    all_fragments.append(frag)

        except Exception as e:
            if "need at least one array to concatenate" in str(e):
                continue  # Skip empty CIFs
            else:
                raise e  # Re-raise unexpected exceptions

    # Process adsorbate molecules
    if adsorbate_atoms is not None and len(adsorbate_atoms) > 0:
        edge_index = get_edge_index(adsorbate_atoms)
        frags = _process_components(adsorbate_atoms, edge_index, "ADSORBATE")
        all_fragments.extend(frags)

    return all_fragments

def process_single_mof(mofid_output_dir: Path, dataset_dir: Path, use_niggli_reduce: bool = True) -> Tuple[str, Union[DecomposedMOF, str], bool]:
    """
    Wrapper function to process one MOFid directory.
    Returns: (mof_identifier, DecomposedMOF object, error_flag)
    """
    try:
        # Each worker loads its own dataset instance
        dataset = AseDBDataset({'src': str(dataset_dir)})

        # Extract the full atoms object from the dataset
        data_idx = mofid_output_dir.name.split("_")[-1]
        atoms = dataset.get_atoms(int(data_idx))

        if use_niggli_reduce:
            # Set to true if used niggli reduction during extract_mofid.py
            niggli_reduce(atoms)

        # Separate adsorbate atoms (tag != 0)
        adsorbate_atoms = atoms[atoms.get_tags() != 0]

        fragments = extract_building_blocks(mofid_output_dir, adsorbate_atoms)

        # Add cell info (for reconstruction later)
        crystal = DecomposedMOF(
            fragments=fragments,
            cell=atoms.get_cell().array,
            lattice=atoms.cell.cellpar(),
            info=atoms.info
        )
        return data_idx, crystal, False
    
    except Exception as e:
        print(f"Error processing {mofid_output_dir.name}: {e}")
        return mofid_output_dir.name, str(e), True

def main(dataset_dir, num_workers):
    dataset_dir = Path(dataset_dir)

    # Determine if niggli reduction was used
    dir_suffix = "_niggli" if (dataset_dir / "mofid_niggli").exists() else ""
    use_niggli_reduce = dir_suffix == "_niggli"

    mofid_base_dir = dataset_dir / f"mofid{dir_suffix}"

    print(f"--- Processing Setup ---")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Number of workers: {num_workers}")
    print(f"Niggli reduction: {'Enabled' if use_niggli_reduce else 'Disabled'}")
    print(f"------------------------")
    
    # Get all individual MOFid output directories
    mofid_subdirs = sorted([d for d in mofid_base_dir.iterdir() if d.is_dir()])
    print(f"Found {len(mofid_subdirs)} MOFid output directories to process in {mofid_base_dir}.")

    # Parallel processing of MOFid directories
    results = Parallel(n_jobs=num_workers)(
        delayed(process_single_mof)(subdir, dataset_dir, use_niggli_reduce)
        for subdir in tqdm(mofid_subdirs, desc="Extracting building blocks")
    )

    # Collect results
    successful_results: Dict[str, DecomposedMOF] = {}
    failed_results: Dict[str, str] = {}

    for identifier, data, error in results:
        if error:
            failed_results[identifier] = data
        else:
            successful_results[identifier] = data
    
    print(f"Successfully processed {len(successful_results)}/{len(mofid_subdirs)} MOFid outputs.")

    # Save results as LMDB
    save_dir = dataset_dir / "blocks"
    save_dir.mkdir(exist_ok=True)

    if successful_results:
        lmdb_path = save_dir / "building_blocks.lmdb"
        env = write_lmdb(lmdb_path)
        with env.begin(write=True) as txn:
            for key, value in tqdm(successful_results.items(), desc="Saving to LMDB"):
                key_bytes = key.encode('ascii')
                value_bytes = pickle.dumps(value)
                txn.put(key_bytes, value_bytes)
        env.close()
    
    if failed_results:
        failed_path = save_dir / "failed.json"
        with open(failed_path, "w") as f:
            json.dump(failed_results, f, indent=4)
    else:
        print("\nSuccessfully processed all structures with no errors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    main(args.dataset_dir, args.num_workers)
