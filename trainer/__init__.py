from hydra.utils import instantiate
from omegaconf import DictConfig
from typing import Dict, Type, Any
from .base_optimizer import BaseOptimizer

# TODO: support dynamic import with importlib
# TODO: for now, just accepts default_optimizer
def get_dataloaders(cfg: DictConfig) -> Type[BaseOptimizer]:
    for dataloader in (x for x in cfg.dataloader if x.get('_target_')):
        dataset = instantiate(dataloader)
        dataloaders.update({
            dataloader: DataLoader(
                dataset=dataset,
                **cfg.train)
            })
    return dataloaders