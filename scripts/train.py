import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import ignite.distributed as idist
import logging

import models
import dataloaders
import losses
from utils import distributed

@hydra.main()
def train(cfg: DictConfig) -> None:
    '''
    Trains the model.
    Args:
        cfg (Dict): hydra configuration file
    '''
    if isinstance(cfg.gpus, list):
        nproc_per_node = len(cfg.gpus)
        backend = cfg.backend
    else:
        nproc_per_node = None
        backend = None
    with idist.Parallel(backend=backend, nproc_per_node=nproc_per_node) as parallel:
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