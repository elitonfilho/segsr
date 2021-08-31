from ignite import metrics
import ignite
from ignite.distributed.utils import get_rank, one_rank_only
import omegaconf
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader, Dataset
from scripts.train import train
from .base_trainer import BaseTrainer
from hydra.utils import instantiate
from utils.utils import AverageMeter
from typing import List, Dict, Iterable
import logging

from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.metrics import Metric
from ignite.utils import setup_logger
import ignite.distributed as idist
from ignite.handlers import Checkpoint, DiskSaver
from ignite.contrib.handlers.tqdm_logger import ProgressBar

class IgniteTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.netG: Module = self.models['netG'].cuda().train()
        self.netD: Module = self.models['netD'].cuda().train()
        # self.netSeg: Module = self.models['netSeg'].cuda().eval()
        
        self.netG = idist.auto_model(self.netG)
        self.netD = idist.auto_model(self.netD)
        # self.netSeg = idist.auto_model(self.netSeg)

        self.optimizerG: Optimizer = self.optimizers['netG']
        self.optimizerD: Optimizer = self.optimizers['netD']
        
        self.optimizerG = idist.auto_optim(self.optimizerG)
        self.optimizerD = idist.auto_optim(self.optimizerD)

        # self.schedulerG = self.schedulers['netG']
        # self.schedulerD = self.schedulers['netD']

        self.img_loss = self.losses['il'].cuda()
        self.adv_loss = self.losses['adv'].cuda()
        self.per_loss = self.losses['per'].cuda()
        self.tv_loss = self.losses['tv'].cuda()
        # self.seg_loss = self.losses['seg'].cuda()

    def train_step(self, engine, batch):

        lr_img, hr_img, seg_img = batch

        lr_img = lr_img.cuda().float()
        hr_img = hr_img.cuda().float()
        seg_img = seg_img.cuda().long()
        
        self.netG.zero_grad()
        self.netG.requires_grad_(True)
        
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

        # label_pred = self.netSeg(fake_img)
        # l_seg = self.seg_loss(label_pred, seg_img).long()
        # g_loss = l_img + l_per + l_adv + l_tv + l_seg
        g_loss = l_img + l_per + l_adv + l_tv

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

        # self.schedulerD.step()
        # self.schedulerG.step()

        return fake_img, hr_img

    def validate_step(self, engine: Engine, batch: Iterable):
        lr_img, hr_img, seg_img = batch

        self.netG.eval()
        self.netG.requires_grad_(False)

        hr_img = hr_img.float().cuda()
        lr_img = lr_img.float().cuda()
        sr_img = self.netG(lr_img)
        # seg_sr_img = self.netSeg(sr_img)

        return sr_img, hr_img

    def run_validation(self, engine: Engine, data: Iterable):
        engine.run(data)
        lossesString = ' '.join(f'{key.upper()}:{value}' for key, value in engine.state.metrics.items()) \
            if engine.state.metrics.values else None
        if lossesString:
            engine.logger.info(lossesString)

    def setup_metrics(self, engine: Engine, type: str = 'train'):
        if type == 'train':
            for metric in self.cfg.trainer.metrics.train:
                _instance = instantiate(self.cfg.trainer.metrics.train.get(metric))
                _instance.attach(engine, metric)
        elif type == 'val':
            for metric in self.cfg.trainer.metrics.val:
                _instance = instantiate(self.cfg.trainer.metrics.val.get(metric))
                _instance.attach(engine, metric)

    def setup_handlers(self, engine: Engine):
        saveDict = {
            'netG': self.netG,
            'netD': self.netD
            }
        _glb = lambda *_: engine.state.epoch
        _checkpoint = Checkpoint(
            to_save= saveDict,
            save_handler= DiskSaver(self.cfg.trainer.save_path, create_dir=True),
            filename_prefix=self.cfg.name,
            score_function=Checkpoint.get_default_score_fn(self.cfg.trainer.validation.save_best),
            score_name=self.cfg.trainer.validation.save_best,
            n_saved=self.cfg.trainer.validation.n_saved,
            greater_or_equal=True
        )
        engine.add_event_handler(Events.EPOCH_COMPLETED, _checkpoint)

    def setup_load_state(self):
        if path:=self.cfg.trainer.get('path_pretrained_seg', None):
            loadDict = {
                'netSeg': self.netSeg
            }
            ckpt = torch.load(path, map_location=f'cuda:{idist.get_rank()}')
            Checkpoint.load_objects(to_load=loadDict, checkpoint=ckpt)
        if path:=self.cfg.trainer.get('path_pretrained_sr', None):
            loadDict = {
                'netG': self.netG,
                'netD': self.netD
            }
            ckpt = torch.load(path, map_location=f'cuda:{idist.get_rank()}')
            Checkpoint.load_objects(to_load=loadDict, checkpoint=ckpt)
    
    def setup_pbar(self, trainer):
        pbar = ProgressBar()
        pbar.attach(trainer)

    def fit(self):
        # torch.autograd.set_detect_anomaly(True)
        train_dataset: Dataset = self.dataloaders['train']
        val_dataset: Dataset = self.dataloaders['val']
        train_loader = idist.auto_dataloader(train_dataset, batch_size=self.cfg.trainer.batch_size, drop_last=True)
        val_loader = idist.auto_dataloader(val_dataset, batch_size=self.cfg.trainer.validation.batch_size, drop_last=True)
        trainer = Engine(self.train_step)
        trainer.logger = setup_logger('trainer')
        validator = Engine(self.validate_step)
        validator.logger = setup_logger('validator')

        self.setup_metrics(trainer, 'train')
        self.setup_metrics(validator, 'val')
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.cfg.trainer.validation.freq) | Events.COMPLETED,
            self.run_validation, validator, val_loader)
        self.setup_handlers(trainer)
        self.setup_load_state()
        self.setup_pbar(trainer)
        self.setup_pbar(validator)
        trainer.run(train_loader, max_epochs=self.cfg.trainer.num_epochs)