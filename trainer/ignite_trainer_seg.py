from pathlib import Path
from typing import Dict, Iterable, List, Union

import ignite
import ignite.distributed as idist
from hydra.utils import instantiate, to_absolute_path, get_original_cwd
from ignite.engine import (create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.utils import setup_logger
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import Dataset

from .base_trainer import BaseTrainer


class IgniteTrainerSeg(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.netSeg: Module = self.models['netSeg'].cuda().train()
        
        self.netSeg = idist.auto_model(self.netSeg)

        self.optimizer: Optimizer = self.optimizers['netSeg']
        
        self.optimizer = idist.auto_optim(self.optimizer)

        self.scheduler = self.schedulers['netSeg']

        self.loss = self.losses['cee'].cuda(device=f'cuda:{idist.get_rank()}')

    def train_step(self, engine: Engine, batch: Iterable):

        hr_img, hr_label = batch

        hr_img = hr_img.cuda().float()
        hr_label = hr_label.cuda().long()

        self.netSeg.train()
        self.netSeg.zero_grad()

        pred_label = self.netSeg(hr_img)
        # loss = self.loss(pred_label, hr_label)
        # loss.backward()

        return pred_label, hr_label

    def validate_step(self, engine: Engine, batch: Iterable):
        hr_img, hr_label = batch

        netSeg: Module = self.netSeg.eval()

        hr_img = hr_img.float().cuda()
        pred_label = netSeg(hr_img)

        return pred_label, hr_label

    def run_validation(self, engine: Engine, data: Iterable) -> None:
        engine.run(data)
        lossesString = '\n'.join(f'{key.upper()}:{value}' for key, value in engine.state.metrics.items()) \
            if engine.state.metrics.values else None
        if lossesString:
            engine.logger.info(lossesString)

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
            require_cm = ['iou']
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
#         print('Loss: ', loss.item())
        return loss.item()

    def setup_handlers(self, engine: Engine, trainer: Engine) -> None:
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
            self.run_validation, evaluator, val_loader)
        self.setup_handlers(evaluator, trainer)
        if self.cfg.trainer.pretrained:
            trainer.add_event_handler(Events.STARTED, self.setup_load_state, self.cfg.trainer.path_pretrained)
        trainer.run(train_loader, max_epochs=self.cfg.trainer.num_epochs)
