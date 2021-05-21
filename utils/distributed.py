from omegaconf import DictConfig
import torch.distributed as dist
from torch import multiprocessing
from typing import Tuple


def setup_dist(cfg: DictConfig) -> None:
    if len(cfg.gpus) > 1 and cfg.dist_type == 'ddp':
        multiprocessing.spawn(
            fn=start_dist,
            args=cfg,
            nprocs=len(cfg.gpus)
        )

#TODO: dist should return gpu_id for later usage
def start_dist(id: int, cfg: Tuple(DictConfig)) -> None:
    dist.init_process_group(
        backend=cfg.backend,
        rank=id,
        world_size=len(cfg.gpus)
    )