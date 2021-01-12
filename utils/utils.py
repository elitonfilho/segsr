from pathlib import Path
from shutil import copy
import torch


def create_pretrain_folder(args, cfg):
    if cfg.TRAIN.model_save_path:
        path_save_model = Path(cfg.TRAIN.model_save_path).resolve()
        path_save_model.mkdir(exist_ok=True)
        copy(args.cfg, path_save_model / 'config.yaml')


def save_model(cfg, netG, netD):
    save_path = Path(cfg.TRAIN.model_save_path).resolve()
    torch.save(netG.state_dict(), save_path / f'{cfg.TRAIN.model_name}_encoder.pth')
    torch.save(netD.state_dict(), save_path / f'{cfg.TRAIN.model_name}_decoder.pth')