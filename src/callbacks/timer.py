import time
from pathlib import Path
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_info


class InferenceTimer(Callback):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.sample_count = 0
        self.start_time = 0.0

    def on_predict_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.sample_count = 0

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs and "atom_types" in outputs:
            self.sample_count += len(outputs["atom_types"])

    def on_predict_end(self, trainer, pl_module):
        total_time = time.time() - self.start_time
        
        avg_time = 0.0
        if self.sample_count > 0:
            avg_time = total_time / self.sample_count
        
        # Log to console
        rank_zero_info(f"Total inference time: {total_time:.4f}s")
        rank_zero_info(f"Number of samples: {self.sample_count}")
        rank_zero_info(f"Avg time per sample: {avg_time:.4f}s")

        # Write to file
        file_path = self.output_dir.parent / "inference_time.txt"
        
        with open(file_path, "w") as f:
            f.write(f"Total inference time: {total_time}s\n")
            f.write(f"Number of samples: {self.sample_count}\n")
            f.write(f"Avg time per sample: {avg_time}s\n")