import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import logging

import models
import dataloaders
import losses
from utils import distributed

logger = logging.getLogger('main')

@hydra.main()
def train(cfg: DictConfig) -> None:
    '''
    Trains the model.
    Args:
        cfg (Dict): hydra configuration file
    '''
    # Initialize distributed
    distributed.setup_dist(cfg)
    # create model
    model = models.get_models(cfg)
    # create losses
    loss = losses.get_losses(cfg)
    # create dataset
    dataloader = dataloaders.get_dataloaders(cfg)
    # create optimizer
    trainer = instantiate(cfg.trainer, cfg, model, loss, dataloader)
    # fit!
    trainer.fit()