from omegaconf import DictConfig
import torch.distributed as dist

def setup_dist(cfg: DictConfig) -> None:
    if cfg.dist_type == 'ddp' and len(cfg.gpus) > 1:
        dist.init_process_group(
            backend=cfg.backend,
            rank=cfg.rank,
            world_size=len(cfg.gpus)
    )