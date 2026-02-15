"""
Split batched predictions into individual sample predictions for evaluation.
"""
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from einops import rearrange


def split_batch_predictions(batch_predictions, batch_indices):
    """
    Args:
        batch_indices (list): List or tensor of batch indices for each sample.
        batch_predictions (dict): Dictionary containing batched predictions and optionally ground truth data.
    Returns:
        results (dict): Dictionary mapping each batch index to its corresponding predictions.
    """
    results = {}
    
    # Check for exceptions
    exception = batch_predictions['exception']
    if exception:
        for batch_idx in batch_indices:
            results[batch_idx] = None
        return results
    
    # Get batched data
    generated_coords = batch_predictions['generated_coords']     # (B * M * P, N, 3)
    generated_lattices = batch_predictions['generated_lattices'] # (B * M * P, 6)
    atom_types = batch_predictions['atom_types']                 # (B, N)
    atom_mask = batch_predictions['atom_mask']                   # (B, N)
    atom_block_type = batch_predictions['atom_block_type']       # (B, N)

    # Optional data
    true_coords = batch_predictions.get('true_coords')           # (B, N, 3) or None
    true_lattices = batch_predictions.get('true_lattices')       # (B, 6) or None
    final_energy = batch_predictions.get('final_energy')         # (B * M * P) or None
    num_particles = batch_predictions.get('num_particles', 1)

    batch_size = atom_types.shape[0]
    total_multiplicity = generated_coords.shape[0] // batch_size

    # -----------------------------------------------------------------------
    # FEYNMAN-KAC REDUCTION LOGIC
    # -----------------------------------------------------------------------
    if final_energy is not None and num_particles > 1:
        # Determine multiplicity
        multiplicity = total_multiplicity // num_particles

        # 1. Reshape energy: (B * M * K) -> (B, M, K)
        energy_reshaped = rearrange(final_energy, '(b m k) -> b m k', b=batch_size, m=multiplicity, k=num_particles)

        # 2. Find best particle indices: (B, M)
        best_particle_idx = torch.argmin(energy_reshaped, dim=-1)

        # 3. Reshape coords and lattices
        coords_reshaped = rearrange(generated_coords, '(b m k) n d -> b m k n d', b=batch_size, m=multiplicity, k=num_particles)
        lattices_reshaped = rearrange(generated_lattices, '(b m k) d -> b m k d', b=batch_size, m=multiplicity, k=num_particles)

        # 4. Select best particles
        b_idx = torch.arange(batch_size, device=generated_coords.device)[:, None].expand(batch_size, multiplicity)
        m_idx = torch.arange(multiplicity, device=generated_coords.device)[None, :].expand(batch_size, multiplicity)

        # Select (B, M, N, 3) and (B, M, 6)
        generated_coords = coords_reshaped[b_idx, m_idx, best_particle_idx]
        generated_lattices = lattices_reshaped[b_idx, m_idx, best_particle_idx]
    else:
        # Standard reshaping if no steering
        multiplicity = total_multiplicity

        generated_coords = rearrange(generated_coords, '(b m) ... -> b m ...', m=multiplicity)
        generated_lattices = rearrange(generated_lattices, '(b m) ... -> b m ...', m=multiplicity)

    # Split predictions
    for i, batch_idx in enumerate(batch_indices):
        mask = atom_mask[i].bool() # Pad mask

        # Remove paddings
        sample_predictions = {
            'generated_coords': generated_coords[i][:, mask], # (M, N, 3)
            'generated_lattice': generated_lattices[i], # (M, 6)
            'atom_types': atom_types[i][mask], # (N,)
            'atom_block_type': atom_block_type[i][mask], # (N,)
        }

        if true_coords is not None:
            sample_predictions['true_coords'] = true_coords[i][mask] # (N, 6)
        if true_lattices is not None:
            sample_predictions['true_lattice'] = true_lattices[i] # (6,)
            
        results[batch_idx] = sample_predictions
    
    return results


def main(args):
    # Set directory
    pred_dir = Path(args.pred_dir)
    raw_outputs_dir = pred_dir / "raw_outputs"

    # Get prediction files
    prediction_files = sorted(list(raw_outputs_dir.glob('predictions_*.pt')))

    # Collate predictions
    results = {}
    for pred_file in tqdm(prediction_files, desc="Evaluating predictions"):
        batch_indices_file = pred_file.parent / pred_file.name.replace('predictions_', 'batch_indices_')

        # Load predictions and batch indices
        predictions = torch.load(pred_file, map_location='cpu')
        batch_indices = torch.load(batch_indices_file, map_location='cpu')

        # Split batch predictions
        batch_results = split_batch_predictions(predictions, batch_indices)
        results.update(batch_results)

    # Sort results by batch index
    results = [value for key, value in sorted(results.items())]

    # Save collated results
    collated_file = pred_dir / 'collated_predictions.pt'
    torch.save(results, collated_file)
    print(f"Collated predictions saved to {collated_file}")

    # Remove raw_outputs 
    if not args.keep_raw_outputs:
        import shutil
        shutil.rmtree(raw_outputs_dir)
        print(f"Removed raw outputs directory: {raw_outputs_dir}")

    # Report number of exceptions
    num_exceptions = sum(1 for r in results if r is None)
    print(f"Number of exceptions (OOM error): {num_exceptions}/{len(results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--keep_raw_outputs', action='store_true', help='Keep raw_outputs directory after collation') # Default: False (delete raw_outputs after collation)
    args = parser.parse_args()
    main(args)