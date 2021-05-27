import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from scripts.train import train
from .base_trainer import BaseTrainer
from hydra.utils import instantiate
from utils.utils import AverageMeter
import logging

logger = logging.getLogger('main')

class DefaultTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def fit(self):

        train_set: DataLoader = self.dataloaders['train']

        netG: Module = self.models['netG'].cuda().train()
        netD: Module = self.models['netD'].cuda().train()
        netSeg: Module = self.models['netSeg'].cuda().eval()

        optimizerG: Optimizer = self.optimizers['netG']
        optimizerD: Optimizer = self.optimizers['netD']

        schedulerG = self.schedulers['netG']
        schedulerD = self.schedulers['netD']

        img_loss = self.losses['il'].cuda()
        adv_loss = self.losses['adv'].cuda()
        per_loss = self.losses['per'].cuda()
        tv_loss = self.losses['tv'].cuda()
        seg_loss = self.losses['seg'].cuda()

        for epoch in range(self.cfg.trainer.num_epochs):
            for lr_img, hr_img, seg_img in train_set:

                lr_img = lr_img.cuda().float()
                hr_img = hr_img.cuda().float()
                seg_img = seg_img.cuda().long()

                netD.eval()
                netG.zero_grad()
                fake_img = netG(lr_img)
                d_fake = netD(fake_img)
                d_real = netD(hr_img).detach()

                l_img = img_loss(fake_img, hr_img)
                l_per = per_loss(fake_img, hr_img)[0]
                l_tv = tv_loss(fake_img, hr_img)
                # l_adv = adv_loss(d_fake, True, is_disc=False)
                l_g_real = adv_loss(d_real - torch.mean(d_fake), False, is_disc=False)
                l_g_fake = adv_loss(d_fake - torch.mean(d_real), True, is_disc=False)
                l_adv = (l_g_real + l_g_fake)/2

                label_pred = netSeg(fake_img)
                l_seg = seg_loss(label_pred, seg_img).long()
                g_loss = l_img + l_per + l_adv + l_tv + l_seg
                g_loss.backward()

                optimizerG.step()

                netD.train()
                netD.zero_grad()
                
                d_fake = netD(fake_img).detach()
                d_real = netD(hr_img)
                l_d_real = adv_loss(d_real - torch.mean(d_fake), True, is_disc=True) * 0.5
                l_d_real.backward()
                d_fake = netD(fake_img.detach())
                l_d_fake = adv_loss(d_fake - torch.mean(d_real.detach()), False, is_disc=True) * 0.5
                l_d_fake.backward()

                optimizerD.step()

                fake_img = netG(lr_img)
                d_fake = netD(fake_img).mean()
                d_real = netD(hr_img).mean()

                print(self.val_metrics,self.val_metrics[0].__name__)

                print(l_img, l_per, l_tv, l_adv)
                if epoch % self.cfg.trainer.validation.freq == 0:
                    self.validate()

            schedulerD.step()
            schedulerG.step()

    def validate(self):
        val_set: DataLoader = self.dataloaders['val']

        netG: Module = self.models['netG'].eval()
        netSeg: Module = self.models['netSeg'].eval()

        val_metrics = self.val_metrics

        val_stats = AverageMeter(('count', *(x.__name__ for x in self.val_metrics)))

        for lr_img, hr_img, seg_img in val_set:
            hr_img = hr_img.float().cuda()
            lr_img = lr_img.float().cuda()
            sr_img = netG(lr_img)
            seg_img = netSeg(sr_img)

            results = {'count': lr_img.shape[0]}
            results.update({f.__name__: f(sr_img, hr_img) for f in self.val_metrics})
            val_stats.update(results)
        print(val_stats)