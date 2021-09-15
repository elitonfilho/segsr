import logging
from pathlib import Path
from typing import Dict, Iterable, List, Union

import ignite
import ignite.distributed as idist
from ignite.distributed import one_rank_only
from hydra.utils import instantiate
from ignite.engine import (create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.handlers.param_scheduler import LRScheduler
from ignite.utils import setup_logger
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from numpy import PINF
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import Dataset
from tensorboardX import SummaryWriter

from .base_trainer import BaseTrainer
from ignite.handlers import FastaiLRFinder

class IgniteTrainerSeg(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.netSeg: Module = self.models['netSeg'].cuda().train()
        
        self.netSeg = idist.auto_model(self.netSeg)

        self.optimizer: Optimizer = self.optimizers['netSeg']
        
        self.optimizer = idist.auto_optim(self.optimizer)

        # self.scheduler = self.schedulers['netSeg']

        self.loss = self.losses['cee'].cuda(device=f'cuda:{idist.get_rank()}')

    def run_validation(self, engine: Engine,  data: Iterable, engineRef: Engine) -> None:
        status = engine.run(data)
        self.call_summary(self.writer, 'val/metrics', engineRef.state.epoch, **status.metrics )

    def prepare_batch(self, batch, *args, **kwargs):
        return (
            batch[0].float().cuda(),
            batch[1].long().cuda()
        )

    def setup_metrics(self, type: str = 'train') -> Union[List[ignite.ignite.metrics.Metric], Dict[str, ignite.ignite.metrics.Metric]]:
        if type == 'train':
            metrics = []
            for metric in self.cfg.trainer.metrics.train:
                _instance = instantiate(self.cfg.trainer.metrics.train.get(metric))
                metrics.append(_instance)
        elif type == 'val':
            require_cm = ['iou','miou']
            metrics = {}
            for metric in self.cfg.trainer.metrics.val:
                if metric in require_cm and metrics.get('cm'):
                    _instance = instantiate(self.cfg.trainer.metrics.val.get(metric), cm=metrics.get('cm'))
                    metrics.update({metric:_instance})
                    continue
                _instance = instantiate(self.cfg.trainer.metrics.val.get(metric))
                metrics.update({metric:_instance})
        return metrics

    def debug_train(self, x, y, ypred, loss):
        return loss.item()

    def setup_save_handler(self, engine: Engine, trainer: Engine) -> None:
        saveDict = {
            'netSeg': self.netSeg
            }
        _checkpoint = Checkpoint(
            to_save= saveDict,
            save_handler= DiskSaver(self.cfg.trainer.save_path, create_dir=True),
            filename_prefix=self.cfg.name,
            global_step_transform=global_step_from_engine(trainer),
            score_function=Checkpoint.get_default_score_fn(self.cfg.trainer.validation.save_best),
            score_name=self.cfg.trainer.validation.save_best,
            n_saved=self.cfg.trainer.validation.n_saved,
            greater_or_equal=True
        )
        engine.add_event_handler(Events.EPOCH_COMPLETED, _checkpoint)

    def setup_load_state(self, path: str):
        loadDict = {
            'netSeg': self.netSeg
            }
        ckpt = torch.load(path, map_location=f'cuda:{idist.get_rank()}')
        Checkpoint.load_objects(to_load=loadDict, checkpoint=ckpt)

    def setup_schedulers(self, engine: Engine):
        _instance = instantiate(self.cfg.trainer.get('scheduler'), optimizer=self.optimizer)
        _wrapped_instance = LRScheduler(_instance)
        engine.add_event_handler(Events.ITERATION_COMPLETED, _wrapped_instance)

    def call_summary_trainer(self, engine: Engine):
        self.call_summary(self.writer, 'train/metrics',engine.state.epoch, **engine.state.metrics)
        self.call_summary(self.writer, 'train/loss', engine.state.epoch, l_seg=engine.state.output)

    @staticmethod
    def setup_pbar(engine: Engine):
        pbar = ProgressBar()
        pbar.attach(engine)

    @staticmethod
    @one_rank_only()
    def call_summary(writer: SummaryWriter, tag: str, globalstep: int, /, **tensors):
        scalars = {}
        for x, y in tensors.items():
            if isinstance(y, torch.Tensor):
                text = f'{x}: {(str(z.item()) for z in y.ravel())}'
                writer.add_text(tag, text, globalstep)
            elif isinstance(y, dict):
                for ykey, yvalue in y.items():
                    text = f'{x}_{ykey}: {yvalue}'
                    writer.add_text(tag, text, globalstep)
            elif isinstance(y, str):
                writer.add_text(tag, f'{x}: {text}', globalstep)
            else:
                scalars.update({x:y})
        # results = {x:y for x,y in tensors.items()}
        writer.add_scalars(tag, scalars, globalstep)

    def fit(self) -> None:
        # torch.autograd.set_detect_anomaly(True)
        train_dataset: Dataset = self.dataloaders['train']
        val_dataset: Dataset = self.dataloaders['val']
        train_loader = idist.auto_dataloader(train_dataset, batch_size=self.cfg.trainer.batch_size)
        val_loader = idist.auto_dataloader(val_dataset, batch_size=self.cfg.trainer.validation.batch_size)

        trainer = create_supervised_trainer(self.netSeg, self.optimizer, self.loss, prepare_batch=self.prepare_batch, device=f'cuda:{idist.get_rank()}', output_transform=self.debug_train)
        trainer.logger = setup_logger('trainer')
        evaluator = create_supervised_evaluator(self.netSeg, self.setup_metrics('val'), prepare_batch=self.prepare_batch, device=f'cuda:{idist.get_rank()}')
        evaluator.logger = setup_logger('validator')

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.cfg.trainer.validation.freq),
            self.run_validation, evaluator, val_loader, trainer)
        self.setup_save_handler(evaluator, trainer)
        if path_pretrained:=self.cfg.trainer.get('path_pretrained_seg', None):
            trainer.add_event_handler(Events.STARTED, self.setup_load_state, path_pretrained)
        if self.cfg.trainer.scheduler:
            self.setup_schedulers(trainer)
        self.setup_pbar(trainer)
        self.setup_pbar(evaluator)
        self.writer = SummaryWriter('tensorboard')
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.call_summary_trainer, trainer)
        trainer.run(train_loader, max_epochs=self.cfg.trainer.num_epochs)
