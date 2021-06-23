import ignite
from ignite.engine import create_supervised_evaluator, create_supervised_trainer
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import Dataset
from .base_trainer import BaseTrainer
from hydra.utils import instantiate
from typing import List, Dict, Iterable

from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.metrics import Metric
from ignite.utils import setup_logger
import ignite.distributed as idist
from ignite.handlers import Checkpoint, DiskSaver

class IgniteTrainerSeg(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.netSeg: Module = self.models['netSeg'].cuda().eval()
        
        self.netSeg = idist.auto_model(self.netSeg)

        self.optimizer: Optimizer = self.optimizers['netSeg']
        
        self.optimizer = idist.auto_optim(self.optimizer)

        self.loss = self.losses['cee'].cuda()

    def train_step(self, engine: Engine, batch: Iterable):

        hr_img, hr_label = batch
        hr_img = hr_img.cuda().float()
        seg_img = hr_label.cuda().long()

        netSeg: Module = self.netSeg.train()
        netSeg.zero_grad()

        pred_label = netSeg(hr_img)
        loss = self.loss(pred_label, hr_label)
        loss.backward()

        return pred_label, seg_img

    def validate_step(self, engine: Engine, batch: Iterable):
        hr_img, hr_label = batch

        netSeg: Module = self.netSeg.eval()

        hr_img = hr_img.float().cuda()
        pred_label = netSeg(hr_img)

        return pred_label, hr_label

    def run_validation(self, engine: Engine, data: Iterable) -> None:
        engine.run(data)
        lossesString = ' '.join(f'{key.upper()}:{value}' for key, value in engine.state.metrics.items()) \
            if engine.state.metrics.values else None
        if lossesString:
            engine.logger.info(lossesString)

    def setup_metrics(self, type: str = 'train') -> List[ignite.ignite.metrics.Metric]:
        if type == 'train':
            metrics = []
            for metric in self.cfg.trainer.metrics.train:
                _instance = instantiate(self.cfg.trainer.metrics.train.get(metric))
                metrics.append(_instance)
        elif type == 'val':
            metrics = {}
            for metric in self.cfg.trainer.metrics.val:
                _instance = instantiate(self.cfg.trainer.metrics.val.get(metric))
                metrics.update({metric:_instance})
        return metrics

    def setup_handlers(self, engine: Engine) -> None:
        saveDict = {'netSeg': self.netSeg}
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

    def fit(self) -> None:
        # torch.autograd.set_detect_anomaly(True)
        train_dataset: Dataset = self.dataloaders['train']
        val_dataset: Dataset = self.dataloaders['val']
        train_loader = idist.auto_dataloader(train_dataset, batch_size=self.cfg.trainer.batch_size)
        val_loader = idist.auto_dataloader(val_dataset, batch_size=self.cfg.trainer.validation.batch_size)

        # TODO: maybe setup a prepare_batch fn
        trainer = create_supervised_trainer(self.netSeg, self.optimizer, self.setup_metrics('train'))
        trainer.logger = setup_logger('trainer')
        evaluator = create_supervised_evaluator(self.netSeg, self.setup_metrics('val'))
        evaluator.logger = setup_logger('validator')

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=self.cfg.trainer.validation.freq) | Events.COMPLETED,
            evaluator.run, val_loader)
        self.setup_handlers(trainer)
        trainer.run(train_loader, max_epochs=self.cfg.trainer.num_epochs)