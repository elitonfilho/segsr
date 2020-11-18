import argparse
import os
from math import log10
import yaml
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from shutil import copy
from pathlib import Path

import torch
import torchvision
import torch.optim as optim
from torch.tensor import Tensor
from torch.utils.data import DataLoader
from config import cfg
from utils.data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from utils import pytorch_ssim
from models.loss_sr import GeneratorLoss, criterionD
from models.model_sr import Generator, Discriminator
from models.model_hrnet import HRNet
from models.models_hrnetv2 import SegmentationModule, getHrnetv2, getC1
from models.model_unet_resnet import UNetResNet
from models.model_unet import UNet

def build_models(cfg):
    netG = Generator(cfg.TRAIN.upscale_factor)
    netD = Discriminator()
    if cfg.TRAIN.arch_enc == 'hrnet':
        # TODO: Better organize load_state_dict on HRNet
        netSeg = SegmentationModule(net_enc=getHrnetv2(cfg.DATASET.n_classes),
                                    net_dec=getC1(cfg.DATASET.n_classes),
                                    crit=nn.NLLLoss(ignore_index=1))
    elif cfg.TRAIN.arch_enc == 'unet':
        netSeg = UNetResNet(num_classes=cfg.DATASET.n_classes)
    else:
        netSeg = None
        print('Not using a segmentation module')

    if cfg.TRAIN.use_pretrained_sr:
        netG.load_state_dict(torch.load(f'{cfg.TRAIN.path_pretrained_sr}_encoder.pth'), strict=False)
        netD.load_state_dict(torch.load(f'{cfg.TRAIN.path_pretrained_sr}_decoder.pth'), strict=False)

    if cfg.TRAIN.use_pretrained_seg:
        netSeg.load_state_dict(torch.load(f'{cfg.TRAIN.path_pretrained_seg}.pth'), strict=False)

    return netG, netD, netSeg


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train Super Resolution Models')

    parser.add_argument(
        "--cfg",
        default="config/exp1.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    train_set = TrainDatasetFromFolder(cfg.DATASET.train_dir, crop_size=cfg.TRAIN.crop_size,
                                       upscale_factor=cfg.TRAIN.upscale_factor, use_aug=cfg.TRAIN.use_aug)
    val_set = ValDatasetFromFolder(cfg.DATASET.val_dir, upscale_factor=cfg.TRAIN.upscale_factor, crop_size=cfg.TRAIN.crop_size)

    train_loader = DataLoader(dataset=train_set, num_workers=4,
                              batch_size=cfg.TRAIN.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4,
                            batch_size=cfg.VAL.batch_size, shuffle=False)

    netG, netD, netSeg = build_models(cfg)

    generator_criterion = GeneratorLoss(seg=cfg.TRAIN.arch_enc, loss_factor=cfg.TRAIN.loss_factor)

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        try:
            netSeg.cuda()
        except (NameError, AttributeError):
            pass
        generator_criterion.cuda()

    optimizerG = optim.Adam(netG.parameters(), lr=cfg.TRAIN.lr)
    optimizerD = optim.Adam(netD.parameters(), lr=cfg.TRAIN.lr)

    schedulerG = optim.lr_scheduler.MultiStepLR(
        optimizerG, cfg.TRAIN.scheduler_milestones, cfg.TRAIN.scheduler_gamma)
    schedulerD = optim.lr_scheduler.MultiStepLR(
        optimizerD, cfg.TRAIN.scheduler_milestones, cfg.TRAIN.scheduler_gamma)

    results = {'d_loss': [], 'g_loss': [], 'd_score': [],
               'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, cfg.TRAIN.num_epochs + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0,
                           'g_loss': 0, 'd_score': 0, 'g_score': 0, 'SL': 0,
                           'seg': 0, 'adv': 0, 'img': 0, 'per': 0, 'tv': 0}

        netG.train()
        netD.train()
        try:
            netSeg.eval()
        except (NameError, AttributeError):
            pass

        for index, (lr, hr, label) in enumerate(train_bar):
            batch_size = lr.size(0)
            running_results['batch_sizes'] += batch_size

            ############################
            # (1) Update D network: maximize D(x) + 1-D(G(z))
            # TODO: As proposed in https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html, optimize D in two steps
            ###########################

            if torch.cuda.is_available():
                hr = hr.cuda()
                lr = lr.cuda()
                label = label.cuda()

            real_img = hr.float()
            lr = lr.float()
            label = label.long()

            fake_img = netG(lr)
            netD.zero_grad()
            # TODO: Relativistic GAN. See https://github.com/xinntao/BasicSR/blob/master/basicsr/models/esrgan_model.py
            real_out = netD(real_img)
            d_real_out = criterionD(real_out, True)
            d_real_out.backward(retain_graph=True)
            fake_out = netD(fake_img)
            d_fake_out = criterionD(fake_out, False)
            d_fake_out.backward(retain_graph=True)
            # d_loss = -(torch.log(real_out) + torch.log(1-fake_out)) is ideal, but D=1 means loss=inf
            # d_loss = 1 - real_out + fake_out
            # d_loss.backward(retain_graph=True)

            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################

            netG.zero_grad()

            _use_seg = True if (cfg.TRAIN.use_seg and float(
                epoch / cfg.TRAIN.num_epochs) >= cfg.TRAIN.begin_seg) else False
            if _use_seg and cfg.TRAIN.arch_enc == 'hrnet':
                feed = {
                    'img_data': fake_img,
                    'seg_label': label
                }
                segSize = (label.shape[0], label.shape[1])
                label_pred = netSeg(feed, segSize=segSize)
                label = label.long().squeeze(1)
                g_loss, losses = generator_criterion(
                    fake_out.detach(), fake_img, real_img, label, label_pred, use_seg=cfg.TRAIN.use_seg)
            elif _use_seg and cfg.TRAIN.arch_enc == 'unet':
                label_pred = netSeg(fake_img)
                g_loss, losses = generator_criterion(
                    fake_out.detach(), fake_img, real_img, label, label_pred, use_seg=cfg.TRAIN.use_seg)
            else:
                g_loss, losses = generator_criterion(
                    fake_out.detach(), fake_img, real_img, use_seg=_use_seg)

            g_loss.backward()

            fake_img = netG(lr)
            fake_out = netD(fake_img).mean()

            optimizerG.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            # running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_loss'] += (d_real_out.item() + d_fake_out.item()) * batch_size / 2
            # running_results['d_score'] += real_out.item() * batch_size
            running_results['d_score'] += real_out.mean().item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
            running_results['seg'] += losses['seg_loss'] * batch_size
            running_results['adv'] += losses['adversarial_loss'] * batch_size
            running_results['img'] += losses['image_loss'] * batch_size
            running_results['per'] += losses['perception_loss'] * batch_size
            running_results['tv'] += losses['tv_loss'] * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f \
            Seg: %.4f Adv: %.4f  Img: %.4f  Per: %.4f Tv: %.4f LR: %f' % (
                epoch, cfg.TRAIN.num_epochs,
                running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes'],
                running_results['seg'] / running_results['batch_sizes'],
                running_results['adv'] / running_results['batch_sizes'],
                running_results['img'] / running_results['batch_sizes'],
                running_results['per'] / running_results['batch_sizes'],
                running_results['tv'] / running_results['batch_sizes'],
                schedulerG.get_last_lr()
            ))

        if cfg.TRAIN.visualize and epoch % cfg.VAL.freq == 0:

            netG.eval()

            out_path = 'results/train_' + str(cfg.TRAIN.model_name) + '/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            with torch.no_grad():
                val_bar = tqdm(val_loader)
                valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
                val_images = []
                for val_lr, val_hr_restore, val_hr, val_seg in val_bar:
                    batch_size = val_lr.size(0)
                    valing_results['batch_sizes'] += batch_size
                    lr = val_lr
                    hr = val_hr
                    if torch.cuda.is_available():
                        lr = lr.cuda()
                        hr = hr.cuda()
                    sr = netG(lr)

                    batch_mse = ((sr - hr) ** 2).data.mean()
                    valing_results['mse'] += batch_mse * batch_size
                    batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                    valing_results['ssims'] += batch_ssim * batch_size
                    valing_results['psnr'] = 10 * log10((hr.max()**2) /
                                                        (valing_results['mse'] / valing_results['batch_sizes']))
                    valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                    val_bar.set_description(
                        desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                            valing_results['psnr'], valing_results['ssim']))

                    val_images.extend(
                        [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                         display_transform()(sr.data.cpu().squeeze(0))])
                val_images = torch.stack(val_images)
                val_images = torch.chunk(val_images, val_images.size(0) // (3*cfg.VAL.n_rows))
                val_save_bar = tqdm(val_images, desc='[saving training results]')
                index = 0
                for image in val_save_bar:
                    image = torchvision.utils.make_grid(image, nrow=3, padding=2)
                    torchvision.utils.save_image(image, out_path + 'val_epoch_%d_index_%d.png' %
                                     (epoch, index), padding=5)
                    index += 1

        schedulerD.step()
        schedulerG.step()

        # save model parameters and configs for current run
        if cfg.TRAIN.model_save_path and epoch == 0:
            path_save_model = Path(cfg.TRAIN.model_save_path).resolve()
            path_save_model.mkdir(exist_ok=True)
            copy(args.cfg, path_save_model / 'config.yaml')
        if epoch == cfg.TRAIN.num_epochs:
            torch.save(netG.state_dict(),
                        f'{cfg.TRAIN.model_save_path}{cfg.TRAIN.model_name}_encoder.pth')
            torch.save(netD.state_dict(),
                        f'{cfg.TRAIN.model_save_path}{cfg.TRAIN.model_name}_decoder.pth')
        # save loss\scores\psnr\ssim
        # results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        # results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        # results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        # results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        # results['psnr'].append(valing_results['psnr'])
        # results['ssim'].append(valing_results['ssim'])

        # if epoch % 10 == 0 and epoch != 0:
        #     out_path = 'statistics/'
        #     data_frame = pd.DataFrame(
        #         data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
        #               'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
        #         index=range(1, epoch + 1))
        #     data_frame.to_csv(out_path + 'srf_' + str(cfg.TRAIN.upscale_factor) + '_train_results.csv', index_label='Epoch')
