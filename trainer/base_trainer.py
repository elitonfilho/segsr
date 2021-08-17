import abc
from utils import val_utils
import torch
from torch.nn.parallel import DistributedDataParallel, DataParallel
import logging
from hydra.utils import instantiate
from utils.metrics import get_metrics

logger = logging.getLogger('main')

class BaseTrainer(abc.ABC):
    '''
    Base trainer. Should be used when defining training strategies.
    '''
    def __init__(self, *args, **kwargs) -> None:
        self.cfg, self.models, self.losses, self.dataloaders = args
        # self.initialize_distributed()
        self.optimizers = self.setup_optimizers()
        # self.schedulers = self.setup_schedulers()
        # self.train_metrics, self.val_metrics = self.setup_metrics()
    
    @abc.abstractmethod
    def fit(self):
        pass

    @staticmethod
    def save_state_dict(model, save_path):
        torch.save(model.state_dict(), save_path)

    @staticmethod
    def load_state_dict(model, load_path):
        model.load_state_dict(torch.load(load_path))
    
    def setup_optimizers(self):
        optimizers = {}
        for name, model in self.models.items():
            optimizer = instantiate(self.cfg.trainer.optimizer, model.parameters())
            optimizers.update({name: optimizer})
            if not optimizer:
                logger.critical(f'Optimizer {self.cfg.trainer.optimizer} not found.')
        return optimizers

    def setup_schedulers(self):
        schedulers = {}
        for name, optimizer in self.optimizers.items():
            scheduler = instantiate(self.cfg.trainer.scheduler, optimizer)
            if not scheduler:
                logger.critical(f'Scheduler {self.cfg.trainer.scheduler} not found.')
            schedulers.update({name: scheduler})
        return schedulers

    def setup_metrics(self):
        train_metrics = []
        val_metrics = []
        for metric in self.cfg.trainer.metrics.train:
            train_metrics.append(get_metrics(metric))
        for metric in self.cfg.trainer.metrics.val:
            val_metrics.append(get_metrics(metric))
        return train_metrics, val_metrics

    def initialize_distributed(self):
        if len(self.cfg.gpus) > 1:
            if self.cfg.dist_type == 'ddp' :
                torch.cuda.set_device(self.cfg.rank)
                self.model = DistributedDataParallel(
                    self.model,
                    self.cfg.gpus,
                    torch.device('gpu:0')
                )
            elif self.cfg.dist_type == 'dp':
                self.model = DataParallel(
                    self.model,
                    self.cfg.gpus,
                    torch.device('gpu:0')
                )
