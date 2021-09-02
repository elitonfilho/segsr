import hydra
from hydra.utils import get_original_cwd, instantiate, to_absolute_path
from omegaconf import DictConfig, ListConfig
from hydra.core.singleton import Singleton
import ignite.distributed as idist

import models
import dataloaders
import losses

@hydra.main()
def train(cfg: DictConfig) -> None:
    '''
    Trains the model.
    Args:
        cfg (Dict): hydra configuration file
    '''
    if isinstance(cfg.gpus, ListConfig) and len(cfg.gpus) > 1:
        nproc_per_node = len(cfg.gpus)
        backend = cfg.backend
        if cfg.trainer.get('path_pretrained_seg', None):
            cfg.trainer.path_pretrained_seg = to_absolute_path(cfg.trainer.path_pretrained_seg)
        if cfg.trainer.get('path_pretrained_sr', None):
            cfg.trainer.path_pretrained_sr = to_absolute_path(cfg.trainer.path_pretrained_sr)
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