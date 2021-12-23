import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, ListConfig
import ignite.distributed as idist

import models
import dataloaders

def test(cfg: DictConfig) -> None:
    '''
    Tests on data.
    Args:
        cfg (DictConfig): hydra configuration file
    '''
    if isinstance(cfg.gpus, ListConfig) and len(cfg.gpus) > 1:
        nproc_per_node = len(cfg.gpus)
        backend = cfg.backend
        if cfg.tester.get('path_pretrained_seg', None):
            cfg.tester.path_pretrained = to_absolute_path(cfg.trainer.path_pretrained)
    else:
        nproc_per_node = None
        backend = None
    with idist.Parallel(backend=backend, nproc_per_node=nproc_per_node) as parallel:
        parallel.run(runTrain, cfg)
            

    
def runTrain(rank, cfg):
    model = models.get_models(cfg)
    # create dataset
    dataset = dataloaders.get_datasets(cfg)
    # create optimizer
    tester = instantiate(cfg.tester, cfg, model, dataset)
    # fit!
    tester.run()