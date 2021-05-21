import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import logging

import models
import dataloaders
import losses
from utils import distributed

logger = logging.getLogger(__name__)

@hydra.main()
def train(cfg: DictConfig) -> None:
    '''
    Trains the model.
    Args:
        cfg (Dict): hydra configuration file
    '''
    # Initialize distributed
    # distributed.setup_dist(cfg)
    # create model
    print('0')
    model = models.get_models(cfg)
    print('1')
    # create losses
    loss = losses.get_losses(cfg)
    print('2')
    # create dataset
    dataloader = dataloaders.get_dataloaders(cfg)
    print('3')
    # create optimizer
    trainer = instantiate(cfg.trainer, model, loss, dataloader)
    # fit!
    trainer.fit()