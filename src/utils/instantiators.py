from typing import List, Optional

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.profilers import Profiler
from omegaconf import DictConfig

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def instantiate_profiler(profiler_cfg: DictConfig) -> Optional[Profiler]:
    """Instantiates a profiler from config.

    :param profiler_cfg: A DictConfig object containing the profiler configuration.
    :return: An instantiated profiler object or None if no config is provided.
    """
    if not profiler_cfg:
        log.info("No profiler config found. Skipping profiler instantiation.")
        return None

    if not isinstance(profiler_cfg, DictConfig):
        raise TypeError("Profiler config must be a DictConfig!")

    if "_target_" in profiler_cfg:
        log.info(f"Instantiating profiler <{profiler_cfg._target_}>")
        return hydra.utils.instantiate(profiler_cfg)

    log.warning("Profiler config is missing a '_target_' key. Skipping.")
    return None