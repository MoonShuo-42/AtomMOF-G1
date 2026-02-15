import os
from pathlib import Path

import hydra
import rootutils
import lightning as L
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.callbacks import PredictionWriter, InferenceTimer
from src.models.simplemof_module import SimpleMOFModule
from src.utils import RankedLogger, extras

log = RankedLogger(__name__, rank_zero_only=True)


def set_dir_name(cfg: DictConfig) -> str:
    """Set directory name based on prediction configuration."""
    steering_args = cfg.predict_args.steering_args

    name_parts = [
        f"{cfg.data.predict_split}",
        f"samples-{cfg.predict_args.num_samples}",
        f"steps-{cfg.predict_args.sampling_steps}",
        f"refine-{cfg.predict_args.refine_final}",
        f"sde_coords-{cfg.predict_args.use_sde_coords}",
        f"fk_steering-{steering_args.fk_steering}",
    ]

    if steering_args.fk_steering:
        name_parts.extend([
            f"fk_lambda-{steering_args.fk_lambda}",
            f"fk_resample-{steering_args.fk_resampling_interval}",
            f"fk_start-{steering_args.fk_start_time}",
            f"potential-{steering_args.potential_mode}",
            f"num_particles-{steering_args.num_particles}",
            f"energy_fn-{steering_args.energy_fn}",
        ])
    base_name = "_".join(name_parts)
    return base_name


def set_callbacks(cfg: DictConfig, output_dir: Path) -> list:
    """Set up callbacks for prediction."""
    callbacks = []

    # Prediction Writer
    pred_writer = PredictionWriter(output_dir=str(output_dir), write_interval="batch")
    callbacks.append(pred_writer)

    # Inference Timer
    if cfg.predict_args.get("measure_time", False):
        num_devices = cfg.trainer.get("devices", 1)
        if num_devices > 1:
            log.warning("Measuring inference time. Ensure that only one device is being used for accurate measurements.")

        callbacks.append(InferenceTimer(output_dir=output_dir))

    return callbacks


def predict(cfg: DictConfig):
    assert cfg.ckpt_path
    ckpt_path = Path(cfg.ckpt_path)
    ckpt_dir = ckpt_path.parents[1]

    # Set seed
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Load datamodule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    # Load model
    log.info(f"Loading model from checkpoint <{ckpt_path}>")
    model = SimpleMOFModule.load_from_checkpoint(str(ckpt_path), predict_args=cfg.predict_args)
    model.eval()

    # Create output directory
    base_name = set_dir_name(cfg)
    output_dir = ckpt_dir / "predictions" / base_name
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Saving predictions to: {output_dir}")

    # Save config
    config_save_path = output_dir.parent / "config.yaml"
    OmegaConf.save(cfg, f=config_save_path)

    # Setup Callbacks
    callbacks = set_callbacks(cfg, output_dir)

    # Set inference mode
    inference_mode = True
    if cfg.predict_args.steering_args.energy_fn == "mlip":
        inference_mode = False

    # Setup trainer
    trainer = hydra.utils.instantiate(
        cfg.trainer, 
        logger=False,
        callbacks=callbacks,
        inference_mode=inference_mode,
    )

    # Run prediction
    trainer.predict(
        model,
        datamodule=datamodule,
        return_predictions=False
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    predict(cfg)


if __name__ == "__main__":
    main()
