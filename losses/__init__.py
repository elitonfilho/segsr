from omegaconf import DictConfig
from typing import Dict
from hydra.utils import instantiate

from .losses import *

def get_losses(cfg: DictConfig) -> Dict:
    losses = {}
    for loss in cfg.trainer.losses:
        losses.update({
            loss: instantiate(cfg.trainer.losses.get(loss))
        })
    return losses