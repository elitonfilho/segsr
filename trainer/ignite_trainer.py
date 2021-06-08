import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader, Dataset
from scripts.train import train
from .base_trainer import BaseTrainer
from hydra.utils import instantiate
from utils.utils import AverageMeter
from typing import List
import logging

from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.metrics import Metric
import ignite.distributed as idist

logger = logging.getLogger(__file__)

class IgniteTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.netG: Module = self.models['netG'].cuda().train()
        self.netD: Module = self.models['netD'].cuda().train()
        self.netSeg: Module = self.models['netSeg'].cuda().eval()
        
        self.netG = idist.auto_model(self.netG)
        self.netD = idist.auto_model(self.netD)
        self.netSeg = idist.auto_model(self.netSeg)

        self.optimizerG: Optimizer = self.optimizers['netG']
        self.optimizerD: Optimizer = self.optimizers['netD']
        
        self.optimizerG = idist.auto_optim(self.optimizerG)
        self.optimizerD = idist.auto_optim(self.optimizerD)

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
        
        self.netG.zero_grad()
        
        self.netD.eval()
        self.netD.requires_grad_(False)

        fake_img = self.netG(lr_img)
        
        d_fake = self.netD(fake_img.clone().detach())
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
        self.netD.requires_grad_(True)
        self.netD.zero_grad()
        
        fake_img = self.netG(lr_img)
        
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

        for metric in self.train_metrics:
            metric.update((fake_img, hr_img))

        self.schedulerD.step()
        self.schedulerG.step()

    def validate_step(self, engine, batch):
        lr_img, hr_img, seg_img = batch

        netG: Module = self.models['netG'].eval()
        netSeg: Module = self.models['netSeg'].eval()

        hr_img = hr_img.float().cuda()
        lr_img = lr_img.float().cuda()
        sr_img = netG(lr_img)
        seg_sr_img = netSeg(sr_img)

        for metric in self.val_metrics:
            metric.update((sr_img, hr_img))
            
    def run_validation(self, engine, batch):
        self.validator.run(batch)

    def setup_metrics(self):
        self.train_metrics : List[Metric] = []
        self.val_metrics : List[Metric] = []
        for metric in self.cfg.trainer.metrics.train:
            self.train_metrics.append(instantiate(self.cfg.trainer.metrics.train.get(metric)))
        for metric in self.cfg.trainer.metrics.val:
            self.val_metrics.append(instantiate(self.cfg.trainer.metrics.val.get(metric)))

    def fit(self):
        torch.autograd.set_detect_anomaly(True)
        train_dataset: Dataset = self.dataloaders['train']
        val_dataset: Dataset = self.dataloaders['val']
        train_loader = idist.auto_dataloader(train_dataset, batch_size=self.cfg.trainer.batch_size)
        val_loader = idist.auto_dataloader(val_dataset, batch_size=self.cfg.trainer.validation.batch_size)
        self.trainer = Engine(self.train_step)
        self.validator = Engine(self.validate_step)
        self.setup_metrics()
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.run_validation, val_loader)
        self.trainer.run(train_loader, max_epochs=self.cfg.trainer.num_epochs)
        

