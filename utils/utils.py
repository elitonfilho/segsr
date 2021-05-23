from pathlib import Path
from shutil import copy
import torch
import pandas as pd
from datetime import datetime

from torch.nn import init

class AverageMeter:
    def __init__(self, *args) -> None:
        self.meter = dict()
        for x in args:
            self.meter.update({x: 0})

    def update(self, delta: dict) -> None:
        for key in self.meter.keys():
            if delta.get(key):
                self.meter[key] += delta[key]
    
    def __str__(self) -> str:
        print(' '.join(f'{key}:{value}' for key, value in self.meter.items()))


def create_pretrain_folder(args, cfg):
    if cfg.TRAIN.model_save_path:
        path_save_model = Path(cfg.TRAIN.model_save_path).resolve()
        if path_save_model.exists():
            time = datetime.now().strftime('%Y%m%d-%H%M')
            path_save_model.rename(
                path_save_model.parent / f'{path_save_model.stem}_{time}')
        path_save_model.mkdir()
        copy(args.cfg, path_save_model / 'config.yaml')
        return path_save_model


def save_model(cfg, best_results, netG, netD):
    save_path = Path(cfg.TRAIN.model_save_path).resolve()
    torch.save(netG, save_path / f'{cfg.TRAIN.model_name}_g_{best_results["epoch"]}.pth')
    torch.save(netD, save_path / f'{cfg.TRAIN.model_name}_d_{best_results["epoch"]}.pth')


def save_train_stats(cfg, epoch, stats):
    out_path = Path(cfg.TRAIN.model_save_path, 'train_stats.csv').resolve()
    len_ds = stats['batch_sizes']
    data_frame = pd.DataFrame(
        data={
            'Epoch': epoch,
            'Loss_D': stats['d_loss']/len_ds,
            'Loss_G': stats['g_loss']/len_ds,
            'Score_D': stats['d_score']/len_ds,
            'Score_G': stats['g_score']/len_ds,
            'Loss_adv': stats['adv']/len_ds,
            'Loss_img': stats['img']/len_ds,
            'Loss_tv': stats['tv']/len_ds,
            'Loss_per': stats['per']/len_ds,
            'Loss_seg': stats['seg']/len_ds,

        }, index=[0])
    data_frame.to_csv(out_path, index_label='Epoch', mode='a', header=not out_path.exists())


def save_val_stats(cfg, epoch, stats):
    out_path = Path(cfg.TRAIN.model_save_path, 'val_stats.csv').resolve()
    if cfg.TRAIN.use_seg:
        data_frame = pd.DataFrame(
            data={
                'Epoch': epoch,
                'PSNR': stats['psnr'],
                'SSIM': stats['ssim'],
                'IoU': stats['iou'],
                'Acc': stats['acc']
            }, index=[0])
    else:
        data_frame = pd.DataFrame(
            data={
                'Epoch': epoch,
                'PSNR': stats['psnr'],
                'SSIM': stats['ssim'],
            }, index=[0])
    data_frame.to_csv(out_path, index_label='Epoch', mode='a', header=not out_path.exists())