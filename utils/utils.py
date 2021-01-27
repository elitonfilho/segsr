from pathlib import Path
from shutil import copy
import torch
import pandas as pd


def create_pretrain_folder(args, cfg):
    if cfg.TRAIN.model_save_path:
        path_save_model = Path(cfg.TRAIN.model_save_path).resolve()
        path_save_model.mkdir(exist_ok=True)
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
            'Loss_D': stats['d_loss'] / stats['batch_sizes'],
            'Loss_G': stats['g_loss'] / stats['batch_sizes'],
            'Score_D': stats['d_score'] / stats['batch_sizes'],
            'Score_G': stats['g_score'] / stats['batch_sizes'],
            'Loss_adv': stats['adv'] / stats['batch_sizes'],
            'Loss_img': stats['img'] / stats['batch_sizes'],
            'Loss_tv': stats['tv'] / stats['batch_sizes'],
            'Loss_per': stats['per'] / stats['batch_sizes'],
            'Loss_seg': stats['seg'] / stats['batch_sizes'],

        }, index=[0])
    data_frame.to_csv(out_path, index_label='Epoch', mode='a', header=not out_path.exists())


def save_val_stats(cfg, epoch, stats):
    out_path = Path(cfg.TRAIN.model_save_path, 'val_stats.csv').resolve()
    data_frame = pd.DataFrame(
        data={
            'Epoch': epoch,
            'PSNR': stats['psnr'] / stats['batch_sizes'],
            'SSIM': stats['ssim'] / stats['batch_sizes'],
        }, index=[0])
    data_frame.to_csv(out_path, index_label='Epoch', mode='a', header=not out_path.exists())
