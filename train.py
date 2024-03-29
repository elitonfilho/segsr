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
from tensorboardX import SummaryWriter

from config import cfg
from models.loss_sr import GeneratorLoss, criterion
from models.model_sr import Discriminator, Generator
from models.model_unet import UNet
from models.model_unet_resnet import UNetResNet
from models.model_hrnet_C1 import SegmentationModule, getC1, getHrnetv2
from models.model_hrnetv1 import get_seg_model
from models.rrdb_arch import RRDBNet
from models.vgg_arch import VGG128
from models.losses import L1Loss, MSELoss, WeightedTVLoss, PerceptualLoss, GANLoss, SegLoss

from datasets import *
from utils import pytorch_ssim
from utils.val_utils import validate_seg
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
        netSeg = get_seg_model(cfg)
    elif cfg.TRAIN.arch_enc == 'unet':
        netSeg = UNetResNet(num_classes=cfg.DATASET.n_classes)
    else:
        netSeg = None
        print('Not using a segmentation module')

    # TODO: Individual load paths
    if cfg.TRAIN.use_pretrained_sr:
        netG.load_state_dict(torch.load(cfg.TRAIN.path_pretrained_g))
        netD.load_state_dict(torch.load(cfg.TRAIN.path_pretrained_d))
    if cfg.TRAIN.use_pretrained_seg:
        netSeg.load_state_dict(torch.load(cfg.TRAIN.path_pretrained_seg))

    return netG, netD, netSeg


def build_loss_criterion(cfg):
    losses = cfg.TRAIN.losses
    weight_classes = cfg.DATASET.weight_classes
    img_loss = L1Loss(losses.il) if losses.il else None
    per_loss = PerceptualLoss({'conv5_4': 1}, perceptual_weight=losses.per) if losses.per else None
    adv_loss = GANLoss('vanilla', loss_weight=losses.adv) if losses.adv else None
    tv_loss = WeightedTVLoss(losses.tv) if losses.tv else None
    seg_loss = SegLoss(losses.seg, weight_classes) if losses.seg else None
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

    path_save_model = create_pretrain_folder(args, cfg)

    criterions = build_loss_criterion(cfg)

    writer = SummaryWriter(path_save_model)

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
                           'g_loss': 0, 'd_score': 0, 'g_score': 0,
                           'seg': 0, 'adv': 0, 'img': 0, 'per': 0, 'tv': 0}
        best_results = {epoch: -1, 'psnr': 0, 'ssim': -1}

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
            # (1) Update G network: minimize 1-D(G(z)) + Losses
            ###########################

            for p in netD.parameters():
                p.requires_grad = False

            netG.zero_grad()
            fake_img = netG(lr)
            d_fake = netD(fake_img)
            d_real = netD(real_img).detach()

            # Getting losses
            l_img = img_loss(fake_img, real_img)
            l_per = per_loss(fake_img, real_img)[0]
            l_tv = tv_loss(fake_img, real_img)
            # l_adv = adv_loss(d_fake, True, is_disc=False)
            l_g_real = adv_loss(d_real - torch.mean(d_fake), False, is_disc=False)
            l_g_fake = adv_loss(d_fake - torch.mean(d_real), True, is_disc=False)
            l_adv = (l_g_real + l_g_fake)/2

            _use_seg = True if (cfg.TRAIN.use_seg and float(
                epoch / cfg.TRAIN.num_epochs) >= cfg.TRAIN.begin_seg) else False
            if _use_seg and cfg.TRAIN.arch_enc == 'hrnet':
                label_pred = netSeg(fake_img)[:,:cfg.DATASET.n_classes,...]
                l_seg = seg_loss(label_pred, label)
                g_loss = l_img + l_per + l_adv + l_tv + l_seg
            elif _use_seg and cfg.TRAIN.arch_enc == 'unet':
                label_pred = netSeg(fake_img)
                l_seg = seg_loss(label_pred, label)
                g_loss = l_img + l_per + l_adv + l_tv + l_seg
            else:
                l_seg = seg_loss(label_pred, label) if 'label_pred' in locals() else torch.tensor(0)
                g_loss = l_img + l_per + l_adv + l_tv

            g_loss.backward()

            optimizerG.step()

            ############################
            # (2) Update D network: maximize D(x) + 1-D(G(z))
            # Optimizing D in two steps, see https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
            ###########################

            for p in netD.parameters():
                p.requires_grad = True

            # fake_img = netG(lr)

            netD.zero_grad()
            d_fake = netD(fake_img).detach()
            d_real = netD(real_img)
            # l_d_real = adv_loss(d_real, True, is_disc=True)
            l_d_real = adv_loss(d_real - torch.mean(d_fake), True, is_disc=True) * 0.5
            l_d_real.backward()
            d_fake = netD(fake_img.detach())
            # l_d_fake = adv_loss(d_fake, False, is_disc=True)
            l_d_fake = adv_loss(d_fake - torch.mean(d_real.detach()), False, is_disc=True) * 0.5
            l_d_fake.backward()

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

            writer.add_scalars('stats/train', running_results, epoch)

        if epoch % cfg.VAL.freq == 0:

            netG.eval()

            val_out_path = Path('results', f'val_{str(cfg.TRAIN.model_name)}').resolve()
            if not os.path.exists(val_out_path) and cfg.VAL.visualize:
                os.makedirs(val_out_path)

            with torch.no_grad():
                val_bar = tqdm(val_loader)
                valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 
                'batch_sizes': 0, 'acc': 0, 'iou': 0, 'accs': 0, 'ious': 0, 
                'cmatrix': torch.zeros(cfg.DATASET.n_classes,cfg.DATASET.n_classes)}

                val_images = []
                for val_lr, val_hr, val_seg in val_bar:
                    batch_size = val_lr.size(0)
                    valing_results['batch_sizes'] += batch_size
                    lr = val_lr.float()
                    hr = val_hr.float()
                    if torch.cuda.is_available():
                        lr = lr.cuda()
                        hr = hr.cuda()
                    sr = netG(lr)
                    if cfg.TRAIN.use_seg:
                        seg = netSeg(sr)[:,:cfg.DATASET.n_classes,...].cuda()
                        val_seg = val_seg.cuda()
                        validate_seg(val_seg, seg, valing_results, batch_size, cfg)
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

                writer.add_scalars('stats/val', valing_results, epoch)

                # Saving validation results
                save_val_stats(cfg, epoch, valing_results)

                if cfg.TRAIN.save_best:
                    metric = cfg.TRAIN.save_best
                    if valing_results[metric] > best_results[metric]:
                        best_results[metric] = valing_results[metric]
                        best_results['epoch'] = epoch
                        best_netG = netG.state_dict()
                        best_netD = netD.state_dict()

                # Saving SR images from validation set if visualize=True
                if cfg.VAL.visualize:
                    if cfg.TRAIN.use_seg:
                        val_images.extend([
                            compose_val()(hr),
                            compose_val()(sr),
                            compose_val()(val_seg),
                            compose_val()(seg)])
                    else:
                        val_images.extend([
                            compose_val()(hr),
                            compose_val()(sr),])
                    val_images = torch.stack(val_images)
                    val_images = torch.chunk(val_images, cfg.VAL.n_chunks)
                    val_save_bar = tqdm(val_images, desc='[saving training results]')
                    index = 0
                    for image in val_save_bar:
                        if cfg.TRAIN.use_seg:
                            image = torchvision.utils.make_grid(image, nrow=4, padding=2)
                        else:
                            image = torchvision.utils.make_grid(image, nrow=2, padding=2)
                        torchvision.utils.save_image(
                            image, val_out_path / f'val_epoch_{epoch}_{index}.png', padding=5)
                        index += 1
                        writer.add_image(f'ep{epoch}_{index}', image, epoch)

        schedulerD.step()
        schedulerG.step()

        save_train_stats(cfg, epoch, running_results)

    try:
        save_model(cfg, best_results, best_netG, best_netD)
    except NameError:
        save_model(cfg, best_results, netG.state_dict(), netD.state_dict())

    writer.close()
