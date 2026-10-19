"""Real training-window measurements, including backward/optimizer/DDP.

No per-step synchronize is introduced. Synchronize at the window boundaries,
report each rank independently, and aggregate using the slowest/largest rank.
This callback is for benchmark runs, not normal training.
"""
import hashlib
import json
import time
from pathlib import Path

import torch
from lightning import Callback


class ExecutionBenchmark(Callback):
    def __init__(self, warmup_batches=20):
        self.warmup_batches = warmup_batches

    @staticmethod
    def _sync(module):
        if module.device.type == "cuda":
            torch.cuda.synchronize(module.device)

    def _peak(self, module):
        if module.device.type != "cuda":
            return (0, 0)
        return (
            torch.cuda.max_memory_allocated(module.device),
            torch.cuda.max_memory_reserved(module.device),
        )

    def on_train_epoch_start(self, trainer, pl_module):
        self._sync(pl_module)
        trainer.strategy.barrier()
        self.epoch_start = time.perf_counter()
        self.window_start = None
        self.finished = False
        self.seen = 0
        self.samples = 0
        self.shapes = []
        self.indices = []
        self.before_peak = (0, 0)
        if pl_module.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx == self.warmup_batches:
            self._sync(pl_module)
            trainer.strategy.barrier()
            self.before_peak = self._peak(pl_module)
            if pl_module.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(pl_module.device)
            self.window_start = time.perf_counter()
            self.step_start = trainer.global_step
        shape = tuple(batch["ref_element"].shape)
        self.shapes.append(shape)
        sampler = trainer.train_dataloader.batch_sampler
        batches = getattr(sampler, "_batches", None)
        if batches is not None and batch_idx < len(batches):
            self.indices.append([int(i) for i in batches[batch_idx]])
        if self.window_start is not None:
            self.samples += shape[0]

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.seen = batch_idx + 1
        if self.seen >= trainer.num_training_batches:
            self._finish(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.finished:
            self._finish(trainer, pl_module)

    def _finish(self, trainer, module):
        if self.finished:
            return
        self.finished = True
        self._sync(module)
        trainer.strategy.barrier()
        end = time.perf_counter()
        peak = self._peak(module)

        def digest(value):
            return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()

        data = dict(
            rank=trainer.global_rank,
            epoch=trainer.current_epoch,
            world_size=trainer.world_size,
            device=str(module.device),
            torch_version=torch.__version__,
            gpu_name=torch.cuda.get_device_name(module.device)
            if module.device.type == "cuda"
            else None,
            train_seconds=end - self.epoch_start,
            measured_seconds=None if self.window_start is None else end - self.window_start,
            warmup_batches=self.warmup_batches,
            total_batches=self.seen,
            measured_batches=max(0, self.seen - self.warmup_batches),
            measured_samples=self.samples,
            optimizer_updates=None
            if self.window_start is None
            else trainer.global_step - self.step_start,
            accumulate_grad_batches=trainer.accumulate_grad_batches,
            precision=str(trainer.precision),
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            peak_allocated_bytes=max(self.before_peak[0], peak[0]),
            peak_reserved_bytes=max(self.before_peak[1], peak[1]),
            steady_peak_allocated_bytes=peak[0],
            shapes=self.shapes,
            batch_indices=self.indices,
            shapes_sha256=digest(self.shapes),
            batches_sha256=digest(self.indices) if len(self.indices) == self.seen else None,
            parameter_count=sum(p.numel() for p in module.parameters()),
            execution=module.structure_module.flow_model.execution_options,
        )
        directory = Path(trainer.default_root_dir) / "performance"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"rank{trainer.global_rank}_epoch{trainer.current_epoch}.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
