import hydra
from omegaconf import DictConfig, OmegaConf

from scripts import train

@hydra.main(config_name="config/default.yaml")
def main(cfg : DictConfig) -> None:
    if cfg.mode == 'train':
        return train(cfg)
    elif cfg.mode == 'test':
        return test(cfg)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()