import argparse
import os
from math import log10
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
import torchvision
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from models.loss_sr import GeneratorLoss, criterion
from models.model_hrnet import HRNet
from models.model_sr import Discriminator, Generator
from models.model_unet import UNet
from models.model_unet_resnet import UNetResNet
from models.models_hrnetv2 import SegmentationModule, getC1, getHrnetv2
from models.rrdb_arch import RRDBNet
from models.vgg_arch import VGG128
from models.losses import L1Loss, MSELoss, WeightedTVLoss, PerceptualLoss, GANLoss, SegLoss

from datasets import *
from utils import pytorch_ssim
from utils.img_utils import compose_val, tensor2img
from utils.utils import *

# TODO: Dynamic instantiation


def build_models(cfg):
    # netG = Generator(cfg.TRAIN.upscale_factor)
    # netG = Discriminator()
    netG = RRDBNet(
        num_in_ch=cfg.ARCHS.netG.num_in_ch,
        num_out_ch=cfg.ARCHS.netG.num_out_ch,
        num_feat=cfg.ARCHS.netG.num_feat,
        num_block=cfg.ARCHS.netG.num_block)
    netD = VGG128(
        num_feat=cfg.ARCHS.netD.num_feat,
        num_in_ch=cfg.ARCHS.netD.num_in_ch)
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

    # TODO: Individual load paths
    if cfg.TRAIN.use_pretrained_sr:
        netG.load_state_dict(torch.load(
            f'{cfg.TRAIN.path_pretrained_sr}_encoder.pth'), strict=False)
        netD.load_state_dict(torch.load(
            f'{cfg.TRAIN.path_pretrained_sr}_decoder.pth'), strict=False)

    if cfg.TRAIN.use_pretrained_seg:
        netSeg.load_state_dict(torch.load(cfg.TRAIN.path_pretrained_seg), strict=False)

    return netG, netD, netSeg

def build_loss_criterion(cfg):
    losses = cfg.TRAIN.losses
    img_loss = L1Loss(losses.il) if losses.il else None
    per_loss = PerceptualLoss({'conv5_4':1}, perceptual_weight=losses.per) if losses.per else None
    adv_loss = GANLoss('vanilla', loss_weight=losses.adv) if losses.adv else None
    tv_loss = WeightedTVLoss(losses.tv) if losses.adv else None
    seg_loss = SegLoss(losses.seg) if losses.adv else None
    return (
        img_loss, per_loss, adv_loss, tv_loss, seg_loss
    )


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

    train_set = eval(cfg.DATASET.type)(
        cfg.DATASET.train_dir,
        crop_size=cfg.TRAIN.crop_size,
        upscale_factor=cfg.TRAIN.upscale_factor,
        use_aug=cfg.TRAIN.use_aug)

    val_set = eval(cfg.DATASET.type)(
        cfg.DATASET.val_dir,
        upscale_factor=cfg.TRAIN.upscale_factor,
        crop_size=cfg.TRAIN.crop_size,
        use_aug=cfg.TRAIN.use_aug)

    train_loader = DataLoader(dataset=train_set, num_workers=4,
                              batch_size=cfg.TRAIN.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4,
                            batch_size=cfg.VAL.batch_size, shuffle=False)

    netG, netD, netSeg = build_models(cfg)

    create_pretrain_folder(args, cfg)

    criterions = build_loss_criterion(cfg)

    # generator_criterion = GeneratorLoss(seg=cfg.TRAIN.arch_enc, loss_factor=cfg.TRAIN.losses)

    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        try:
            netSeg.cuda()
        except (NameError, AttributeError):
            pass
        for crit in criterions:
            if crit:
                crit.cuda()
        # generator_criterion.cuda()

    img_loss, per_loss, adv_loss, tv_loss, seg_loss = criterions

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

            if torch.cuda.is_available():
                hr = hr.cuda()
                lr = lr.cuda()
                label = label.cuda()

            real_img = hr.float()
            lr = lr.float()
            label = label.long()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Losses
            ###########################

            for p in netD.parameters():
                p.requires_grad = False

            netG.zero_grad()
            fake_img = netG(lr)
            d_fake = netD(fake_img)

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
                    d_fake.detach(), fake_img, real_img, label, label_pred, use_seg=cfg.TRAIN.use_seg)
            elif _use_seg and cfg.TRAIN.arch_enc == 'unet':
                label_pred = netSeg(fake_img)
                g_loss, losses = generator_criterion(
                    d_fake.detach(), fake_img, real_img, label, label_pred, use_seg=cfg.TRAIN.use_seg)
            else:
                l_img = img_loss(fake_img, real_img)
                l_per = per_loss(fake_img, real_img)[0]
                l_adv = adv_loss(d_fake, True, is_disc=False)
                l_tv = tv_loss(fake_img, real_img)
                l_seg = seg_loss(label_pred, label) if 'label_pred' in locals() else torch.tensor(0)
                g_loss = l_img + l_per + l_adv + l_tv
                # g_loss, losses = generator_criterion(
                #     d_fake.detach(), fake_img, real_img, use_seg=_use_seg)

            g_loss.backward()

            optimizerG.step()

            ############################
            # (1) Update D network: maximize D(x) + 1-D(G(z))
            # TODO: As proposed in https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html, optimize D in two steps
            ###########################

            for p in netD.parameters():
                p.requires_grad = True

            fake_img = netG(lr)

            netD.zero_grad()
            # TODO: Relativistic GAN. See https://github.com/xinntao/BasicSR/blob/master/basicsr/models/esrgan_model.py
            d_real = netD(real_img)
            l_d_real = adv_loss(d_real, True, is_disc=True)
            # l_d_real = criterion(d_real, True)
            l_d_real.backward()
            d_fake = netD(fake_img)
            l_d_fake = adv_loss(d_fake, False, is_disc=True)
            # l_d_fake = criterion(d_fake, False)
            l_d_fake.backward()
            # d_loss = -(torch.log(d_real) + torch.log(1-d_fake)) is ideal, but D=1 means loss=inf
            # d_loss = 1 - d_real + d_fake
            # d_loss.backward(retain_graph=True)

            optimizerD.step()

            fake_img = netG(lr)
            d_fake = netD(fake_img).mean()
            d_real = netD(real_img).mean()

            # Statistics for current batch
            running_results['g_loss'] += g_loss.item() * batch_size
            # running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_loss'] += (l_d_real.item() + l_d_fake.item()) * batch_size
            running_results['d_score'] += d_real.item() * batch_size
            running_results['g_score'] += d_fake.item() * batch_size
            running_results['seg'] += l_seg.item() * batch_size
            running_results['adv'] += l_adv.item() * batch_size
            running_results['img'] += l_img.item() * batch_size
            running_results['per'] += l_per.item() * batch_size
            running_results['tv'] += l_tv.item() * batch_size
            # running_results['seg'] += losses['seg_loss'] * batch_size
            # running_results['adv'] += losses['adversarial_loss'] * batch_size
            # running_results['img'] += losses['image_loss'] * batch_size
            # running_results['per'] += losses['perception_loss'] * batch_size
            # running_results['tv'] += losses['tv_loss'] * batch_size

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
                schedulerG.get_last_lr()[0]
            ))

        # TODO: Save val stats
        if epoch % cfg.VAL.freq == 0:

            netG.eval()

            val_out_path = Path('results', f'val_{str(cfg.TRAIN.model_name)}').resolve()
            if not os.path.exists(val_out_path) and cfg.VAL.visualize:
                os.makedirs(val_out_path)

            with torch.no_grad():
                val_bar = tqdm(val_loader)
                valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
                val_images = []
                for val_lr, val_hr, val_seg in val_bar:
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
                        desc='[Stats on validation set] PSNR: %.4f dB SSIM: %.4f' % (
                            valing_results['psnr'], valing_results['ssim']))

                    if cfg.VAL.visualize:
                        val_images.extend([
                            compose_val()(hr),
                            compose_val()(sr)])

                # Saving validation results
                save_val_stats(cfg, epoch, valing_results)

                # Saving SR images from validation set if visualize=True
                if cfg.VAL.visualize:
                    val_images = torch.stack(val_images)
                    val_images = torch.chunk(val_images, cfg.VAL.n_chunks)
                    val_save_bar = tqdm(val_images, desc='[saving training results]')
                    index = 0
                    for image in val_save_bar:
                        image = torchvision.utils.make_grid(image, nrow=2, padding=2)
                        torchvision.utils.save_image(
                            image, val_out_path / f'val_epoch_{epoch}_{index}.png', padding=5)
                        index += 1

        schedulerD.step()
        schedulerG.step()

        save_train_stats(cfg, epoch, running_results)

        if epoch == cfg.TRAIN.num_epochs:
            save_model(cfg, netG, netD)
