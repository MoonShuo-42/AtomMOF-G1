import torch
from pathlib import Path
from lightning.pytorch.callbacks import BasePredictionWriter


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: str, write_interval):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir) / "raw_outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        # Save predictions
        torch.save(prediction, self.output_dir / f"predictions_rank-{trainer.global_rank}_batch-{batch_idx}.pt")

        # Save batch_indices (data indices)
        torch.save(batch_indices, self.output_dir / f"batch_indices_rank-{trainer.global_rank}_batch-{batch_idx}.pt")