import os
import gc
import time
import random
import multiprocessing as mp
from typing import Any, Optional

import torch
from einops import rearrange
from lightning import Callback, LightningModule
from lightning.pytorch.utilities import rank_zero_info
from torch import Tensor, nn
from pebble import ProcessPool
from concurrent.futures import TimeoutError as PebbleTimeoutError
from pymatgen.analysis.structure_matcher import StructureMatcher

from src.models.loss.validation import find_best_match_rmsd
from src.models.loss.utils import stratify_loss_by_time, stratify_losses_by_time
from src.models.simple.flow import AtomFlowMatching
from src.models.optim.ema import EMA
from src.models.optim.scheduler import AlphaFoldLRScheduler


class SimpleMOFModule(LightningModule):
    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        atom_feature_dim: int,
        flow_model_args: dict[str, Any],
        training_args: dict[str, Any],
        validation_args: dict[str, Any],
        flow_process_args: dict[str, Any],
        ema: bool = False,
        ema_decay: float = 0.999,
        predict_args: Optional[dict[str, Any]] = None,
        use_kernels: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # EMA
        self.use_ema = ema
        self.ema_decay = ema_decay

        # Args
        self.training_args = training_args
        self.validation_args = validation_args
        self.predict_args = predict_args

        # Kernels
        self.use_kernels = use_kernels

        # Validation metrics
        self.matcher_kwargs = StructureMatcher(**self.validation_args['structure_matcher_args']).as_dict()
        self.validation_step_outputs = []

        # Flow matching module
        self.structure_module = AtomFlowMatching(
            flow_model_args={
                "atom_s": atom_s,
                "atom_z": atom_z,
                "token_s": token_s,
                "token_z": token_z,
                "atom_feature_dim": atom_feature_dim,
                **flow_model_args,
            },
            **flow_process_args,
        )
    
    def setup(self, stage: str) -> None:
        """Set the model for training, validation and inference."""
        if stage == "predict" and not (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(torch.device("cuda")).major >= 8
        ):
            self.use_kernels = False

    def forward(
        self, 
        feats: dict[str, Tensor],
        # Training arguments
        multiplicity_flow_train: int = 1,
        # Sampling arguments
        multiplicity_flow_sample: int = 1,
        num_sampling_steps: Optional[int] = None,
        use_sde_coords: bool = False,
        center_during_sampling: bool = True,
        refine_final: bool = False,
        return_trajectory: bool = False,
        steering_args: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        
        dict_out = {}

        ##### Structure prediction #####
        if self.training:
            # Returns denoised predictions for loss computation
            dict_out.update(
                self.structure_module(
                    feats=feats,
                    multiplicity=multiplicity_flow_train
                )
            )
        else:
            # Returns sampled structure
            atom_mask = feats["atom_pad_mask"]

            network_condition_kwargs = dict(
                feats=feats,
            )

            dict_out.update(
                self.structure_module.sample(
                    atom_mask=atom_mask,
                    num_sampling_steps=num_sampling_steps,
                    multiplicity=multiplicity_flow_sample,
                    use_sde_coords=use_sde_coords,
                    center_coords=center_during_sampling,
                    refine_final=refine_final,
                    return_trajectory=return_trajectory,
                    steering_args=steering_args,
                    **network_condition_kwargs,
                )
            )

        return dict_out
    
    def _compute_and_log_loss(self, batch: dict[str, Tensor], prefix: str) -> tuple[Tensor, dict, dict]:
        # Compute the forward pass
        out = self.structure_module(
            feats=batch,
            multiplicity=self.training_args['train_multiplicity'],
        )

        # Compute losses
        flow_loss_dict = self.structure_module.compute_loss(
            batch,
            out,
            multiplicity=self.training_args['train_multiplicity'],
            loss_type=self.training_args['loss_type'],
            use_dist_loss=self.training_args['use_dist_loss'],
            dist_cutoff=self.training_args['dist_cutoff'],
            use_dist_schedule=self.training_args['use_dist_schedule']
        )

        # The scalar reductions also feed logging; compute each only once.
        mean_losses = {name: values.mean() for name, values in flow_loss_dict.items()}
        total_loss = (
            self.training_args['coord_loss_weight'] * mean_losses['coord_loss']
            + self.training_args['lengths_loss_weight'] * mean_losses['length_loss']
            + self.training_args['angle_loss_weight'] * mean_losses['angle_loss']
            + self.training_args['dist_loss_weight'] * mean_losses['dist_loss']
        )

        # Loggings
        batch_size = batch['ref_element'].shape[0]
        sync_dist = prefix == 'val'

        # Loss per component
        for loss_name, mean_loss in mean_losses.items():
            self.log(f'{prefix}/{loss_name}', mean_loss, batch_size=batch_size, sync_dist=sync_dist)

        # Total loss
        self.log(f'{prefix}/total_loss', total_loss, batch_size=batch_size, sync_dist=sync_dist)

        return total_loss, flow_loss_dict, out

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        step_start_time = time.time() 

        total_loss, flow_loss_dict, out = self._compute_and_log_loss(batch, prefix='train')

        # Training-specific loggings
        batch_size = batch['ref_element'].shape[0]
        self.log('train/batch_size', batch_size)

        # Stratified loss logging
        if self.structure_module.flow_model.execution_options.get("joint_time_metrics", False):
            stratified_losses = stratify_losses_by_time(out['times'], flow_loss_dict)
        else:
            stratified_losses = {}
            for loss_name, batch_loss in flow_loss_dict.items():
                stratified_losses.update(stratify_loss_by_time(
                    batch_t=out['times'], batch_loss=batch_loss, loss_name=loss_name,
                    fast=self.structure_module.flow_model.execution_options.get("vectorized_metrics", False),
                ))
        for k, v in stratified_losses.items():
            self.log(f'train/{k}', v, batch_size=batch_size)

        # Time
        step_time = time.time() - step_start_time
        self.log('train/samples_per_second', batch_size / step_time)

        return total_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        # Compute and log validation loss
        self._compute_and_log_loss(batch, prefix='val')

        # Sampling and structure matching
        sample_every_n_epochs = self.validation_args.get('sample_every_n_epochs', 1)
        if sample_every_n_epochs <= 0 or (self.current_epoch + 1) % sample_every_n_epochs != 0:
            return
        
        n_samples = self.validation_args['flow_samples']

        try:
            out = self(
                batch,
                num_sampling_steps=self.validation_args['sampling_steps'],
                center_during_sampling=self.validation_args['sampling_center_coords'],
                multiplicity_flow_sample=n_samples,
            )
        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                rank_zero_info("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return
            else:
                raise e
        
        # Aggregate outputs
        return_dict = {
            "sampled_coords": out['sampled_atom_coords'],
            "sampled_lattices": out['sampled_lattice'],
            "true_coords": batch['cart_coords'],
            "true_lattices": batch['lattice'],
            "atom_types": batch['ref_element'],
            "atom_mask": batch['atom_pad_mask'],
            "block_type": batch['block_type'],
        }
        # Keep epoch-long validation caches off the GPU on every rank.
        self.validation_step_outputs.append(
            {key: value.detach().cpu() for key, value in return_dict.items()}
        )

    def on_validation_epoch_end(self) -> None:
        """Compute local structure matches and reduce sample totals across all ranks."""
        try:
            n_samples = self.validation_args["flow_samples"]
            match_mode = self.validation_args.get("match_mode", "all")
            tasks = []

            for batch_output in self.validation_step_outputs:
                if batch_output is None:
                    continue
                sampled_coords = rearrange(
                    batch_output["sampled_coords"], "(b m) n c -> b m n c", m=n_samples
                ).cpu().numpy()
                sampled_lattices = rearrange(
                    batch_output["sampled_lattices"], "(b m) c -> b m c", m=n_samples
                ).cpu().numpy()
                true_coords = batch_output["true_coords"].cpu().numpy()
                true_lattices = batch_output["true_lattices"].cpu().numpy()
                atom_types = batch_output["atom_types"].cpu().numpy()
                atom_mask = batch_output["atom_mask"].cpu().numpy().astype(bool)
                block_type = batch_output["block_type"].cpu().numpy()

                for i in range(sampled_coords.shape[0]):
                    tasks.append(
                        (
                            sampled_coords[i],
                            sampled_lattices[i],
                            true_coords[i],
                            true_lattices[i],
                            atom_types[i],
                            atom_mask[i],
                            block_type[i],
                            match_mode,
                            self.matcher_kwargs,
                        )
                    )

            results_per_item = []
            if tasks:
                timeout_seconds = self.validation_args["timeout"]
                num_workers = self.validation_args["num_workers"]
                with ProcessPool(
                    max_workers=num_workers, context=mp.get_context("spawn")
                ) as pool:
                    futures = [
                        pool.schedule(
                            find_best_match_rmsd, args=(task,), timeout=timeout_seconds
                        )
                        for task in tasks
                    ]
                    for future in futures:
                        try:
                            results_per_item.append(future.result())
                        except PebbleTimeoutError:
                            future.cancel()
                            rank_zero_info("| WARNING: A structure matching task timed out.")
                            results_per_item.append([None] * n_samples)
                        except Exception as error:
                            future.cancel()
                            rank_zero_info(f"| WARNING: A structure matching task failed: {error}")
                            results_per_item.append([None] * n_samples)

            matched = 0
            rmsd_sum = 0.0
            for result_list in results_per_item:
                valid_rmsds = [r for r in result_list if r is not None]
                if valid_rmsds:
                    matched += 1
                    rmsd_sum += min(valid_rmsds)

            # Reduce counts/sums, not rank averages: dynamic batches can have
            # different sizes, and a rank may have no successful sampling batches.
            # All ranks must participate, including ranks with an empty cache.
            totals = torch.tensor(
                [len(results_per_item), matched, rmsd_sum],
                dtype=torch.float64,
                device=self.device,
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
            sample_count, match_count, rmsd_sum = totals.tolist()
            if sample_count == 0:
                return

            # Every rank already has the same global values; do not reduce again.
            self.log("val/match_rate", match_count / sample_count, sync_dist=False)
            self.log(
                "val/avg_rmsd",
                rmsd_sum / match_count if match_count else float("nan"),
                sync_dist=False,
            )
        finally:
            # Runs on every rank, also on empty epochs and when matching fails.
            self.validation_step_outputs.clear()

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        num_samples = self.predict_args.get('num_samples', 1)

        try:
            out = self(
                batch,
                num_sampling_steps=self.predict_args['sampling_steps'],
                use_sde_coords=self.predict_args['use_sde_coords'],
                center_during_sampling=self.predict_args['sampling_center_coords'],
                refine_final=self.predict_args['refine_final'],
                steering_args=self.predict_args.get('steering_args', None),
                multiplicity_flow_sample=num_samples,
            )
            
            prediction_output = {
                "exception": False,
                "generated_coords": out['sampled_atom_coords'],
                "generated_lattices": out['sampled_lattice'],
                "atom_types": batch['ref_element'],
                "atom_mask": batch['atom_pad_mask'],
                "atom_block_type": batch['block_type'],
            }

            if 'final_energy' in out:
                prediction_output['final_energy'] = out['final_energy']
                prediction_output['num_particles'] = self.predict_args['steering_args']['num_particles']

            if 'cart_coords' in batch:
                prediction_output['true_coords'] = batch['cart_coords']
            
            if 'lattice' in batch:
                prediction_output['true_lattices'] = batch['lattice']

            return prediction_output

        except RuntimeError as e:
            if "out of memory" in str(e):
                rank_zero_info(f"| WARNING: Ran out of memory during prediction on batch {batch_idx}. Skipping batch.")
                torch.cuda.empty_cache()
                gc.collect()
                return {"exception": True}
            else:
                raise e
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.trainer.model.parameters(), 
            lr=self.training_args.lr,
            weight_decay=self.training_args.weight_decay,
            **({"fused": True} if self.training_args.get("fused_adam", False) else {}),
        )

        if self.training_args.lr_scheduler == "af3":
            scheduler = AlphaFoldLRScheduler(
                optimizer=optimizer,
                base_lr=self.training_args.base_lr,
                max_lr=self.training_args.max_lr,
                warmup_no_steps=self.training_args.lr_warmup_no_steps,
                start_decay_after_n_steps=self.training_args.lr_start_decay_after_n_steps,
                decay_every_n_steps=self.training_args.lr_decay_every_n_steps,
                decay_factor=self.training_args.lr_decay_factor,
            )
            return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

        return {"optimizer": optimizer}

    def configure_callbacks(self) -> list[Callback]:
        """Configure model callbacks.

        Returns
        -------
        List[Callback]
            List of callbacks to be used in the model.

        """
        return [EMA(self.ema_decay)] if self.use_ema else []
