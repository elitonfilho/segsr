import hydra
from omegaconf import DictConfig, OmegaConf

from scripts import train, test

@hydra.main(config_path="config", config_name='default')
def main(cfg : DictConfig) -> None:
    if cfg.mode == 'train':
        return train(cfg)
    elif cfg.mode == 'test':
        return test(cfg)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()