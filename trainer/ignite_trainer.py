import logging
from typing import Iterable, List

import ignite
import ignite.distributed as idist
from ignite.distributed import one_rank_only
import torch
from hydra.utils import instantiate
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers.terminate_on_nan import TerminateOnNan
from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.handlers.param_scheduler import LRScheduler
from ignite.metrics import Metric
from ignite.utils import setup_logger
from torch.functional import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader, Dataset

from .base_trainer import BaseTrainer
from tensorboardX import SummaryWriter


class IgniteTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.netG: Module = self.models['netG'].cuda().train()
        self.netD: Module = self.models['netD'].cuda().train()
        
        self.netG = idist.auto_model(self.netG)
        self.netD = idist.auto_model(self.netD)
        
        if 'netSeg' in self.models:
            self.netSeg: Module = self.models['netSeg'].cuda().eval()
            self.netSeg = idist.auto_model(self.netSeg)

        self.optimizerG: Optimizer = self.optimizers['netG']
        self.optimizerD: Optimizer = self.optimizers['netD']
        
        self.optimizerG = idist.auto_optim(self.optimizerG)
        self.optimizerD = idist.auto_optim(self.optimizerD)

        self.img_loss = self.losses['il'].cuda()
        self.adv_loss = self.losses['adv'].cuda()
        self.per_loss = self.losses['per'].cuda()
        self.tv_loss = self.losses['tv'].cuda()
        if 'seg' in self.losses:
            self.seg_loss = self.losses['seg'].cuda()

    def train_step(self, engine: Engine, batch: List[Tensor]):

        lr_img, hr_img, seg_img, _ = batch

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

        
        if hasattr(self, 'netSeg'):
            label_pred = self.netSeg(fake_img)
            l_seg = self.seg_loss(label_pred, seg_img)
            g_loss = l_img + l_per + l_adv + l_tv + l_seg
            self.call_summary(self.writer, 'train/losses', engine.state.epoch, \
                l_img=l_img.item(), l_per=l_per.item(), l_adv=l_adv.item(), l_tv=l_tv.item(), l_seg=l_seg.item())
        else:
            g_loss = l_img + l_per + l_adv + l_tv
            self.call_summary(self.writer, 'train/losses', engine.state.epoch, \
                l_img=l_img.item(), l_per=l_per.item(), l_adv=l_adv.item(), l_tv=l_tv.item())

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
        # d_fake = self.netD(fake_img).mean()
        # d_real = self.netD(hr_img).mean()

        return fake_img, hr_img

    def validate_step(self, engine: Engine, batch: Iterable):
        lr_img, hr_img, seg_img, _ = batch

        self.netG.eval()
        self.netG.requires_grad_(False)

        hr_img = hr_img.float().cuda()
        lr_img = lr_img.float().cuda()
        sr_img = self.netG(lr_img)
        # seg_sr_img = self.netSeg(sr_img)

        return sr_img, hr_img

    def run_validation(self, engine: Engine, data: Iterable, engineRef: Engine):
        status = engine.run(data)
        self.call_summary(self.writer, 'val/metrics', engineRef.state.epoch, **status.metrics )

    def setup_metrics(self, engine: Engine, type: str = 'train'):
        if type == 'train':
            for metric in self.cfg.trainer.metrics.train:
                _instance = instantiate(self.cfg.trainer.metrics.train.get(metric))
                _instance.attach(engine, metric)
        elif type == 'val':
            for metric in self.cfg.trainer.metrics.val:
                _instance = instantiate(self.cfg.trainer.metrics.val.get(metric))
                _instance.attach(engine, metric)

    def setup_save_state(self, engine: Engine, engineRef: Engine):
        saveDict = {
            'netG': self.netG,
            'netD': self.netD
            }
        _checkpoint = Checkpoint(
            to_save= saveDict,
            save_handler= DiskSaver(self.cfg.trainer.save_path, create_dir=True),
            filename_prefix=self.cfg.name,
            score_function=Checkpoint.get_default_score_fn(self.cfg.trainer.validation.save_best),
            score_name=self.cfg.trainer.validation.save_best,
            n_saved=self.cfg.trainer.validation.n_saved,
            greater_or_equal=True,
            global_step_transform=global_step_from_engine(engineRef)
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
    
    @one_rank_only()
    def setup_pbar(self, engine: Engine):
        pbar = ProgressBar()
        pbar.attach(engine)

    def setup_schedulers(self, engine: Engine, optimizer: Optimizer):
        _instance = instantiate(self.cfg.trainer.get('scheduler'), optimizer=optimizer)
        _wrapped_instance = LRScheduler(_instance)
        engine.add_event_handler(Events.ITERATION_COMPLETED, _wrapped_instance)

    @staticmethod
    @one_rank_only()
    def call_summary(writer: SummaryWriter, tag: str, globalstep: int, /, **tensors):
        results = {x:y for x,y in tensors.items()}
        writer.add_scalars(tag, results, globalstep)

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
            self.run_validation, validator, val_loader, trainer)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        self.setup_save_state(validator, trainer)
        self.setup_load_state()
        self.setup_pbar(trainer)
        self.setup_pbar(validator)
        self.setup_schedulers(trainer, self.optimizerG)
        self.setup_schedulers(trainer, self.optimizerD)
        self.writer = SummaryWriter('tensorboard')
        trainer.run(train_loader, max_epochs=self.cfg.trainer.num_epochs)
