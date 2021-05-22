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
    print('s0')
    model = models.get_models(cfg)
    print('s1')
    # create losses
    loss = losses.get_losses(cfg)
    print('s2')
    # create dataset
    dataloader = dataloaders.get_dataloaders(cfg)
    print('s3')
    # create optimizer
    trainer = instantiate(cfg.trainer, cfg, model, loss, dataloader)
    # fit!
    # trainer.fit()