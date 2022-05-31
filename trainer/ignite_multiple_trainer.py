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
            if model_name == 'netSeg':
                self.netSeg.eval()
                self.netSeg.requires_grad_(False)

        # Adapt optimizer for distributed scenario
        for model_name, optim in self.optimizers.items():
            optim = idist.auto_optim(optim)
            setattr(self, f'optim{model_name[-1]}', optim)

        # Setting up losses
        for loss_name, loss in self.losses.items():
            loss = loss.cuda()
            setattr(self, loss_name, loss)

    def train_step(self, engine: Engine, batch: List[Tensor]):
        lr_img, hr_img, _, _ = batch

        lr_img = lr_img.cuda().float()
        hr_img = hr_img.cuda().float()

        netG : torch.nn.Module = getattr(self, 'netG')
        netG.zero_grad()
        fake_img = netG(lr_img)

        loss: torch.Tensor = self.il(fake_img, hr_img)
        self.call_summary(self.writer, 'train/losses', engine.state.epoch, l_img=loss.item())
        loss.backward()
        
        optimizer : torch.optim.Optimizer = getattr(self, 'optimG')
        optimizer.step()

        return fake_img, hr_img

    def validate_step(self, engine: Engine, batch: Iterable):
        lr_img, hr_img, _, _ = batch

        netG : torch.nn.Module = getattr(self, 'netG')

        with torch.no_grad():
            hr_img = hr_img.float().cuda()
            lr_img = lr_img.float().cuda()
            sr_img = netG(lr_img)

        return sr_img, hr_img

    def validate_step_seg(self, engine: Engine, batch: Iterable):
        lr_img, _, label_img, _ = batch

        netG : torch.nn.Module = getattr(self, 'netG')
        netSeg: torch.nn.Module = getattr(self, 'netSeg')

        with torch.no_grad():
            label_img = label_img.long().cuda()
            lr_img = lr_img.float().cuda()
            sr_img = netG(lr_img)
            seg_img = netSeg(sr_img)

        return seg_img, label_img
    
    def train_step_abpnseg(self, engine: Engine, batch: List[Tensor]):
        lr_img, hr_img, label_hr, _ = batch

        lr_img = lr_img.cuda().float()
        hr_img = hr_img.cuda().float()

        netG : torch.nn.Module = getattr(self, 'netG')
        netSeg : torch.nn.Module = getattr(self, 'netSeg')
        netG.zero_grad()
        fake_img = netG(lr_img)
        label_seg = netSeg(fake_img)
        loss_il: torch.Tensor = self.il(fake_img, hr_img)
        loss_seg: torch.Tensor = self.seg(label_seg, label_hr.long().cuda())
        loss = loss_il + loss_seg
        self.call_summary(self.writer, 'train/losses', engine.state.epoch, l_il=loss_il.item(), l_seg=loss_seg.item())
        loss.backward()
        
        optimizer : torch.optim.Optimizer = getattr(self, 'optimG')
        optimizer.step()

        return fake_img, hr_img
    
    def validation_step_abpnseg(self, engine: Engine, batch: Iterable):
        lr_img, hr_img, _, _ = batch

        netG : torch.nn.Module = getattr(self, 'netG')

        with torch.no_grad():
            hr_img = hr_img.float().cuda()
            lr_img = lr_img.float().cuda()
            sr_img = netG(lr_img)

        return sr_img, hr_img

    def train_step_srresnet(self, engine: Engine, batch: Iterable[Tensor]):
        lr_img, hr_img, _, _ = batch

        lr_img = lr_img.cuda().float()
        hr_img = hr_img.cuda().float()

        netG : torch.nn.Module = getattr(self, 'netG')
        optimG: torch.optim.Optimizer = getattr(self, 'optimG')
        loss_il = getattr(self, 'il')
        loss_per = getattr(self, 'per')
        netG.zero_grad()

        fake_img = netG(lr_img)

        l_il = loss_il(fake_img, hr_img)
        l_per = loss_per(fake_img, hr_img)[0]
        g_loss = l_il + l_per

        g_loss.backward()
        optimG.step()

        return fake_img, hr_img

    def validation_step_srresnet(self, engine: Engine, batch: Iterable[Tensor]):
        lr_img, hr_img, _, _ = batch

        netG : torch.nn.Module = getattr(self, 'netG')

        with torch.no_grad():
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
        elif type == 'val_seg':
            n_classes = self.cfg.dataloader.n_classes
            for metric in self.cfg.trainer.metrics.val_seg:
                if metric in ('iou','miou','dice','jaccard'):
                    cm = ignite.metrics.ConfusionMatrix(n_classes)
                    _instance = instantiate(self.cfg.trainer.metrics.val_seg.get(metric), cm)
                else:
                    _instance = instantiate(self.cfg.trainer.metrics.val_seg.get(metric))
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
        scalars = {}
        for x, y in tensors.items():
            if isinstance(y, torch.Tensor):
                text = f'{x}: {"".join(str(z.item()) for z in y.ravel())}'
                writer.add_text(tag, text, globalstep)
            elif isinstance(y, dict):
                for ykey, yvalue in y.items():
                    text = f'{x}_{ykey}: {yvalue}'
                    writer.add_text(tag, text, globalstep)
            elif isinstance(y, str):
                writer.add_text(tag, f'{x}: {text}', globalstep)
            else:
                scalars.update({x:y})
        writer.add_scalars(tag, scalars, globalstep)

    def get_run_step_fn(self, name: str) -> Tuple[Callable,Callable]:
        if name in ('edsr', 'rcan', 'rdn', 'dbpn', 'csnln', 'drln', 'abpn', 'paedsr'):
            train_func = self.train_step
            val_func = self.validate_step
        else:
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
        validator_seg = Engine(self.validate_step_seg)
        validator_seg.logger = setup_logger('validator_seg')

        self.setup_metrics(trainer, 'train')
        self.setup_metrics(validator, 'val')
        self.setup_metrics(validator_seg, 'val_seg')
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.cfg.trainer.validation.freq) | Events.COMPLETED,
            self.run_validation, validator, val_loader, trainer)
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.cfg.trainer.validation.freq) | Events.COMPLETED,
            self.run_validation, validator_seg, val_loader, trainer)
#         trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        self.setup_save_state(validator, trainer)
        self.setup_pbar(trainer)
        self.setup_pbar(validator)
        self.setup_pbar(validator_seg)
        self.setup_schedulers(trainer)
        self.writer = SummaryWriter('tensorboard')
        trainer.run(train_loader, max_epochs=self.cfg.trainer.num_epochs)
