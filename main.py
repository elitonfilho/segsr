import hydra
from omegaconf import DictConfig, OmegaConf

from scripts import train, test

@hydra.main(config_path="config", config_name='default')
def main(cfg : DictConfig) -> None:
    if cfg.mode == 'train':
        # print(OmegaConf.to_yaml(cfg))
        return train(cfg)
    elif cfg.mode == 'test':
        # print(OmegaConf.to_yaml(cfg))
        return test(cfg)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()