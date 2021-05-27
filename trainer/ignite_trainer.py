import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from scripts.train import train
from .base_trainer import BaseTrainer
from hydra.utils import instantiate
from utils.utils import AverageMeter
import logging

from ignite.engine.engine import Engine
from ignite.engine.events import Events

logger = logging.getLogger('main')

class IgniteTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.netG: Module = self.models['netG'].cuda().train()
        self.netD: Module = self.models['netD'].cuda().train()
        self.netSeg: Module = self.models['netSeg'].cuda().eval()

        self.optimizerG: Optimizer = self.optimizers['netG']
        self.optimizerD: Optimizer = self.optimizers['netD']

        self.schedulerG = self.schedulers['netG']
        self.schedulerD = self.schedulers['netD']

        self.img_loss = self.losses['il'].cuda()
        self.adv_loss = self.losses['adv'].cuda()
        self.per_loss = self.losses['per'].cuda()
        self.tv_loss = self.losses['tv'].cuda()
        self.seg_loss = self.losses['seg'].cuda()




    def train_step(self, engine, batch):

        lr_img, hr_img, seg_img = batch

        lr_img = lr_img.cuda().float()
        hr_img = hr_img.cuda().float()
        seg_img = seg_img.cuda().long()

        self.netD.eval()
        self.netG.zero_grad()
        fake_img = self.netG(lr_img)
        d_fake = self.netD(fake_img)
        d_real = self.netD(hr_img).detach()

        l_img = self.img_loss(fake_img, hr_img)
        l_per = self.per_loss(fake_img, hr_img)[0]
        l_tv = self.tv_loss(fake_img, hr_img)
        # l_adv = adv_loss(d_fake, True, is_disc=False)
        l_g_real = self.adv_loss(d_real - torch.mean(d_fake), False, is_disc=False)
        l_g_fake = self.adv_loss(d_fake - torch.mean(d_real), True, is_disc=False)
        l_adv = (l_g_real + l_g_fake)/2

        label_pred = self.netSeg(fake_img)
        l_seg = self.seg_loss(label_pred, seg_img).long()
        g_loss = l_img + l_per + l_adv + l_tv + l_seg
        g_loss.backward()

        self.optimizerG.step()

        self.netD.train()
        self.netD.zero_grad()
        
        d_fake = self.netD(fake_img).detach()
        d_real = self.netD(hr_img)
        l_d_real = self.adv_loss(d_real - torch.mean(d_fake), True, is_disc=True) * 0.5
        l_d_real.backward()
        d_fake = self.netD(fake_img.detach())
        l_d_fake = self.adv_loss(d_fake - torch.mean(d_real.detach()), False, is_disc=True) * 0.5
        l_d_fake.backward()

        self.optimizerD.step()

        fake_img = self.netG(lr_img)
        d_fake = self.netD(fake_img).mean()
        d_real = self.netD(hr_img).mean()

        print(self.val_metrics,self.val_metrics[0].__name__)

        print(l_img, l_per, l_tv, l_adv)


        self.schedulerD.step()
        self.schedulerG.step()


    def fit(self):
        self.train_loader: DataLoader = self.dataloaders['train']
        self.trainer = Engine(self.train_step)
        self.trainer.run(self.train_loader)

    @self.trainer.on(Events.ITERATION_COMPLETED(every=1))
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