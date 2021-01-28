from pathlib import Path
from shutil import copy
import torch
import pandas as pd
from datetime import datetime


def create_pretrain_folder(args, cfg):
    if cfg.TRAIN.model_save_path:
        path_save_model = Path(cfg.TRAIN.model_save_path).resolve()
        if path_save_model.exists():
            time = datetime.now().strftime('%Y%m%d-%H%M')
            path_save_model.rename(
                path_save_model.parent / f'{path_save_model.stem}_{time}')
        path_save_model.mkdir()
        copy(args.cfg, path_save_model / 'config.yaml')


def save_model(cfg, netG, netD):
    save_path = Path(cfg.TRAIN.model_save_path).resolve()
    torch.save(netG.state_dict(), save_path / f'{cfg.TRAIN.model_name}_encoder.pth')
    torch.save(netD.state_dict(), save_path / f'{cfg.TRAIN.model_name}_decoder.pth')


def save_train_stats(cfg, epoch, stats):
    out_path = Path(cfg.TRAIN.model_save_path, 'train_stats.csv').resolve()
    data_frame = pd.DataFrame(
        data={
            'Epoch': epoch,
            'Loss_D': stats['d_loss'],
            'Loss_G': stats['g_loss'],
            'Score_D': stats['d_score'],
            'Score_G': stats['g_score'],
            'Loss_adv': stats['adv'],
            'Loss_img': stats['img'],
            'Loss_tv': stats['tv'],
            'Loss_per': stats['per'],
            'Loss_seg': stats['seg'],

        }, index=[0])
    data_frame.to_csv(out_path, index_label='Epoch', mode='a', header=not out_path.exists())


def save_val_stats(cfg, epoch, stats):
    out_path = Path(cfg.TRAIN.model_save_path, 'val_stats.csv').resolve()
    data_frame = pd.DataFrame(
        data={
            'Epoch': epoch,
            'PSNR': stats['psnr'],
            'SSIM': stats['ssim'],
        }, index=[0])
    data_frame.to_csv(out_path, index_label='Epoch', mode='a', header=not out_path.exists())
