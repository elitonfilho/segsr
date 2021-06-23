from torch.utils.data import DataLoader
from hydra.utils import instantiate
from omegaconf import DictConfig
from typing import Dict

from .dataloader_cgeo import CGEODataset, CGEODatasetForSegTask
from .dataloader_landcover import LandCoverDataset
from .dataloader_landcoverai import LandCoverAIDataset
# TODO: support dynamic import with importlib

def get_dataloaders(cfg: DictConfig) -> Dict:
    dataloaders = {}
    options = ('train', 'val') if cfg.mode == 'train' else ('test')
    for dataloader in options:
        dataset = instantiate(cfg.dataloader.get(dataloader))
        dataloaders.update({
            dataloader: DataLoader(
                dataset=dataset,
                batch_size=cfg.dataloader.batch_size,
                num_workers=cfg.dataloader.num_workers,
                pin_memory=cfg.dataloader.pin_memory,
                drop_last=cfg.dataloader.drop_last,
                shuffle=cfg.dataloader.shuffle,
                )
            })
    return dataloaders

def get_datasets(cfg: DictConfig) -> Dict:
    datasets = {}
    options = ('train', 'val') if cfg.mode == 'train' else ('test')
    for option in options:
        dataset = instantiate(cfg.dataloader.get(option))
        datasets.update({
            option: dataset
        })
    return datasets