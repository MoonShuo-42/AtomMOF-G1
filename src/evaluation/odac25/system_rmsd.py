"""
Compute match rate and RMSD between predicted and true structures per MOF+adsorbate system.
"""
import pickle
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Literal

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher


def to_numpy_dict(prediction: Dict) -> Dict:
    """Converts all torch Tensors in a dict to numpy arrays."""
    if prediction is None: 
        return None
    new_pred = {}
    for k, v in prediction.items():
        if isinstance(v, torch.Tensor):
            new_pred[k] = v.detach().cpu().numpy()
        else:
            new_pred[k] = v
    return new_pred


def tensor_to_structure(
    atom_types: np.ndarray,
    coords: np.ndarray,
    lattice: np.ndarray
) -> Structure:
    atoms = Atoms(
        symbols=atom_types,
        positions=coords,
        cell=lattice,
        pbc=True
    )
    structure = AseAtomsAdaptor.get_structure(atoms)
    return structure


def compute_metrics_from_matrix(
    pred_list: List[Structure], 
    gt_list: List[Structure],
    structure_matcher: StructureMatcher
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Computes an N x M comparison to get both Precision and Coverage in one go.
    """
    if not pred_list or not gt_list:
        return {
            "precision_match_rate": 0.0, "precision_rmsd": None,
            "coverage_match_rate": 0.0, "coverage_rmsd": None
        }, np.array([])
    
    # 1. Initialize Matrix with Infinity
    # Rows = Predictions (N), Cols = Ground Truths (M)
    rmsd_matrix = np.full((len(pred_list), len(gt_list)), np.inf)

    # 2. O(N*M) loop to fill rmsd matrix
    for i, pred in enumerate(pred_list):
        for j, gt in enumerate(gt_list):
            
            result = structure_matcher.get_rms_dist(pred, gt)
            
            if result is not None:
                rms, scale = result
                rmsd_matrix[i, j] = rms

    # 3. Computpe metrics
    # Precision (per row/prediction)
    min_rmsds_per_pred = np.min(rmsd_matrix, axis=1)  # (N,)
    pred_matches_mask = min_rmsds_per_pred < np.inf
    
    cov_precision = np.sum(pred_matches_mask) / len(pred_list)
    if np.any(pred_matches_mask):
        amr_precision = np.mean(min_rmsds_per_pred[pred_matches_mask])
    else:
        amr_precision = None

    # Coverage (per column/ground truth)
    min_rmsds_per_gt = np.min(rmsd_matrix, axis=0)  # (M,)
    gt_matches_mask = min_rmsds_per_gt < np.inf

    cov_recall = np.sum(gt_matches_mask) / len(gt_list)
    if np.any(gt_matches_mask):
        amr_recall = np.mean(min_rmsds_per_gt[gt_matches_mask])
    else:
        amr_recall = None

    metrics = {
        "cov_precision": cov_precision,
        "amr_precision": amr_precision,
        "cov_recall": cov_recall,
        "amr_recall": amr_recall
    }

    return metrics, rmsd_matrix
    

def process_system(
    system_id: str,
    system_preds: List[Dict],
    matcher_args: Dict,
    num_samples: Optional[int] = None,
    mask_mode: Literal["all", "framework", "adsorbate"] = "all"
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Worker function to process a single system.
    Extracts structures -> Calculates Metrics -> Returns Dictionary. 
    """
    # Initialize matcher 
    matcher = StructureMatcher(**matcher_args)

    gt_structures = []
    pred_structures = []

    # 1. Collect all structures for this system
    for prediction in system_preds:

        if prediction is None:
            continue  # Skip OOM or failed predictions

        # Determine mask
        atom_block_type = prediction.get('atom_block_type')
        if mask_mode == 'all':
            mask = np.ones(prediction['atom_types'].shape[0], dtype=bool)
        elif mask_mode == 'framework':
            mask = atom_block_type != 3
        elif mask_mode == 'adsorbate':
            mask = atom_block_type == 3
        else:
            raise ValueError(f"Invalid mask_mode: {mask_mode}")
        
        atom_types = prediction['atom_types'][mask]

        # Ground truth structure
        gt_structure = tensor_to_structure(
            atom_types=atom_types,
            coords=prediction['true_coords'][mask],
            lattice=prediction['true_lattice']
        )
        gt_structures.append(gt_structure)

        # Predicted structure
        # generated_coords is (M, N, 3), generated lattice is (M, 6)
        limit = num_samples if num_samples is not None else prediction['generated_coords'].shape[0]
        limit = min(limit, prediction['generated_coords'].shape[0])

        for m in range(limit):
            pred_structure = tensor_to_structure(
                atom_types=atom_types,
                coords=prediction['generated_coords'][m][mask], # NOTE: Set to [m][-1][mask] for GCMC/RS
                lattice=prediction['generated_lattice'][m]
            )
            pred_structures.append(pred_structure)

    # 2. Compute metrics
    metrics, rmsd_matrix = compute_metrics_from_matrix(
        pred_list=pred_structures,
        gt_list=gt_structures,
        structure_matcher=matcher
    )

    results_dict = {
        "system_id": system_id,
        "num_gt": len(gt_structures),
        "num_pred": len(pred_structures),
        **metrics
    }

    return results_dict, rmsd_matrix
        

def summarize_results(df: pd.DataFrame) -> None:
    print("\n------ Summary Statistics ------")
    print(df.describe())

    # Simple averages
    # TODO: Weighted averages
    avg_precision_cov = df['cov_precision'].mean()
    avg_recall_cov = df['cov_recall'].mean()
    avg_precision_amr = df['amr_precision'].mean()
    avg_recall_amr = df['amr_recall'].mean()

    # --- Print Report ---
    print(f"Average Coverage Precision (COV-P): {avg_precision_cov:.4f}")
    print(f"Average AMR Precision (AMR-P): {avg_precision_amr:.4f}")
    print(f"Average Coverage Recall (COV-R): {avg_recall_cov:.4f}")
    print(f"Average AMR Recall (AMR-R): {avg_recall_amr:.4f}")


def main(args):
    print("------ Loading Inputs ------")
    # Load system index
    system_index_path = Path(args.system_index_path)
    with open(system_index_path, 'rb') as f:
        system_indices = pickle.load(f)

    # Load model predictions
    model_pred_path = Path(args.model_pred_path)
    prediction_list = torch.load(model_pred_path, map_location='cpu')
    prediction_list = [to_numpy_dict(pred) for pred in prediction_list]

    # Infer samples if not provided
    if args.num_samples is None:
        valid_pred = next((p for p in prediction_list if p is not None), None)
        if valid_pred is None:
            raise ValueError("All predictions are None due to OOM exceptions.")
        
        args.num_samples = valid_pred['generated_coords'].shape[0]
        print(f"Inferred num_samples: {args.num_samples}")

    # ------ Prepare Matcher Config ------
    matcher_args = {
        "stol": args.stol,
        "ltol": args.ltol,
        "angle_tol": args.angle_tol
    }

    # ------ Processing Systems ------
    print(f"Starting evaluation for {len(system_indices)} systems with {args.num_cpus} CPUs...")

    # Prepare for parallel processing
    tasks_data = []
    for system_id, indices in system_indices.items():
        system_preds = [prediction_list[idx] for idx in indices]
        
        if system_preds:
            tasks_data.append((system_id, system_preds))

    # Run parallel processing
    results = Parallel(n_jobs=args.num_cpus, verbose=10, backend='loky')(
        delayed(process_system)(
            system_id,
            system_preds,
            matcher_args,
            args.num_samples,
            args.mask
        ) for system_id, system_preds in tasks_data
    )

    # ------ Unpack Results ------
    metrics_list = []
    matrices_dict = {}

    for metrics_data, rmsd_matrix in results:
        metrics_list.append(metrics_data)
        matrices_dict[metrics_data['system_id']] = rmsd_matrix

    # ------ Save and Summarize Results ------
    df = pd.DataFrame(metrics_list)

    # Set output directory
    if args.output_dir is None:
        output_dir = model_pred_path.parent / "system_rmsd_results"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f'num_samples-{args.num_samples}_mask-{args.mask}_stol-{args.stol}_ltol-{args.ltol}_angle-{args.angle_tol}'

    csv_path = output_dir / f"{base_name}_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    pkl_path = output_dir / f"{base_name}_matrices.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(matrices_dict, f)
    print(f"RMSD matrices saved to {pkl_path}")

    summarize_results(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--system_index_path', type=str, required=True, default="path/to/system_index.pkl")
    parser.add_argument('--model_pred_path', type=str, required=True, default="path/to/collated_predictions.pt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--num_cpus', type=int, default=1) # NOTE: 1 is faster
    parser.add_argument('--stol', type=float, default=0.5)
    parser.add_argument('--ltol', type=float, default=0.3)
    parser.add_argument('--angle_tol', type=float, default=10)
    parser.add_argument('--mask', type=str, default='all', choices=['all', 'framework', 'adsorbate'])
    args = parser.parse_args()
    main(args)