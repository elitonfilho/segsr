from typing import Callable, Iterable, List, Tuple

import ignite
import ignite.distributed as idist
import torch
from hydra.utils import instantiate
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.distributed import one_rank_only
from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.handlers.param_scheduler import LRScheduler
from ignite.handlers.terminate_on_nan import TerminateOnNan
from ignite.metrics import Metric
from ignite.utils import setup_logger
from tensorboardX import SummaryWriter
from torch.functional import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import Dataset

from .base_trainer import BaseTrainer


class IgniteMultipleTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Loading the module on correct device
        for model_name, model in self.models.items():
            model = idist.auto_model(model).train()
            setattr(self, model_name, model)

        # Adapt optimizer for distributed scenario
        for model_name, optim in self.optimizers.items():
            optim = idist.auto_optim(optim)
            setattr(self, f'optim{model_name[-1]}', optim)

        # Setting up losses
        for loss_name, loss in self.losses.items():
            loss = loss.cuda()
            setattr(self, loss_name, loss)

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

    def train_step_edsr(self, engine: Engine, batch: Iterable[Tensor]):
        lr_img, hr_img, _, _ = batch

        lr_img = lr_img.cuda().float()
        hr_img = hr_img.cuda().float()

        netG : torch.nn.Module = getattr(self, 'netG')
        fake_img = netG(lr_img)

        loss = torch.tensor(0, device=idist.device(), dtype=torch.float)
        for loss_name in self.losses:
            loss_fn = getattr(self, loss_name)
            loss += loss_fn(fake_img, hr_img)
        self.call_summary(self.writer, 'train/losses', engine.state.epoch, l_img=loss.item())
        loss.backward()
        
        optimizer : torch.optim.Optimizer = getattr(self, 'optimG')
        optimizer.step()

        return fake_img, hr_img

    def validation_step_edsr(self, engine: Engine, batch: Iterable[Tensor]):
        lr_img, hr_img, _, _ = batch

        netG : torch.nn.Module = getattr(self, 'netG')

        hr_img = hr_img.float().cuda()
        lr_img = lr_img.float().cuda()
        sr_img = netG(lr_img)

        return sr_img, hr_img

    def train_step_rcan(self, engine: Engine, batch: Iterable[Tensor]):
        lr_img, hr_img, _, _ = batch

        lr_img = lr_img.cuda().float()
        hr_img = hr_img.cuda().float()

        netG : torch.nn.Module = getattr(self, 'netG')
        fake_img = netG(lr_img)

        loss = torch.tensor(0, device=idist.device(), dtype=torch.float)
        for loss_name in self.losses:
            loss_fn = getattr(self, loss_name)
            loss += loss_fn(fake_img, hr_img)
        self.call_summary(self.writer, 'train/losses', engine.state.epoch, l_img=loss.item())
        loss.backward()
        
        optimizer : torch.optim.Optimizer = getattr(self, 'optimG')
        optimizer.step()

        return fake_img, hr_img

    def validation_step_rcan(self, engine: Engine, batch: Iterable[Tensor]):
        lr_img, hr_img, _, _ = batch

        netG : torch.nn.Module = getattr(self, 'netG')

        hr_img = hr_img.float().cuda()
        lr_img = lr_img.float().cuda()
        sr_img = netG(lr_img)

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
        save_dict = dict()
        for model_name in self.models:
            save_dict.update({model_name: getattr(self, model_name)})
        _checkpoint = Checkpoint(
            to_save= save_dict,
            save_handler= DiskSaver(self.cfg.trainer.save_path, create_dir=True),
            filename_prefix=self.cfg.name,
            score_function=Checkpoint.get_default_score_fn(self.cfg.trainer.validation.save_best),
            score_name=self.cfg.trainer.validation.save_best,
            n_saved=self.cfg.trainer.validation.n_saved,
            greater_or_equal=True,
            global_step_transform=global_step_from_engine(engineRef)
        )
        engine.add_event_handler(Events.EPOCH_COMPLETED, _checkpoint)
    
    @one_rank_only()
    def setup_pbar(self, engine: Engine):
        pbar = ProgressBar()
        pbar.attach(engine)

    def setup_schedulers(self, engine: Engine):
        for model_name in self.optimizers:
            optim = getattr(self, f'optim{model_name[-1]}')
            _instance = instantiate(self.cfg.trainer.get('scheduler'), optimizer=optim)
            _wrapped_instance = LRScheduler(_instance)
            engine.add_event_handler(Events.ITERATION_COMPLETED, _wrapped_instance)

    @staticmethod
    @one_rank_only()
    def call_summary(writer: SummaryWriter, tag: str, globalstep: int, /, **tensors):
        results = {x:y for x,y in tensors.items()}
        writer.add_scalars(tag, results, globalstep)

    def get_run_step_fn(self, name: str) -> Tuple[Callable]:
        train_func = getattr(self, f'train_step_{name}')
        val_func = getattr(self, f'validation_step_{name}')
        return train_func, val_func

    def fit(self):
        # torch.autograd.set_detect_anomaly(True)
        train_dataset: Dataset = self.dataloaders['train']
        val_dataset: Dataset = self.dataloaders['val']
        train_loader = idist.auto_dataloader(train_dataset, batch_size=self.cfg.trainer.batch_size, drop_last=True)
        val_loader = idist.auto_dataloader(val_dataset, batch_size=self.cfg.trainer.validation.batch_size, drop_last=True)
        train_step, validate_step = self.get_run_step_fn(self.cfg.trainer.model_name)
        trainer = Engine(train_step)
        trainer.logger = setup_logger('trainer')
        validator = Engine(validate_step)
        validator.logger = setup_logger('validator')

        self.setup_metrics(trainer, 'train')
        self.setup_metrics(validator, 'val')
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.cfg.trainer.validation.freq) | Events.COMPLETED,
            self.run_validation, validator, val_loader, trainer)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        self.setup_save_state(validator, trainer)
        self.setup_pbar(trainer)
        self.setup_pbar(validator)
        self.setup_schedulers(trainer)
        self.writer = SummaryWriter('tensorboard')
        trainer.run(train_loader, max_epochs=self.cfg.trainer.num_epochs)
