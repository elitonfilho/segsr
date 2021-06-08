import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import ignite.distributed as idist
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
#     distributed.setup_dist(cfg)
    # create model
    with idist.Parallel(backend=cfg.backend, nproc_per_node=len(cfg.gpus)) as parallel:
        parallel.run(runTrain, cfg)
            

    
def runTrain(rank, cfg):
    model = models.get_models(cfg)
    # create losses
    loss = losses.get_losses(cfg)
    # create dataset
    dataset = dataloaders.get_datasets(cfg)
    # create optimizer
    trainer = instantiate(cfg.trainer, cfg, model, loss, dataset)
    # fit!
    trainer.fit()