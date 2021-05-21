import abc
import torch
from torch.nn.parallel import DistributedDataParallel, DataParallel

class BaseTrainer(abc.ABCMeta):
    '''
    Base trainer. Should be used when defining training strategies.
    '''
    def __init__(self, *args, **kwargs) -> None:
        self.cfg, self.model, self.loss, self.dataloader = args
        self.initialize_distributed()
    
    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def save_state_dict(self):
        pass

    @abc.abstractmethod
    def load_state_dict(self):
        pass
    
    @abc.abstractmethod
    def setup_optimizers(self):
        pass

    @abc.abstractmethod
    def setup_schedulers(self):
        pass

    def initialize_distributed(self):
        if self.cfg.dist_type == 'ddp':
            self.model = DistributedDataParallel(
                self.model,

            )
        elif self.cfg.dist_type == 'dp':
            self.model = DataParallel(
                self.model,
                self.cfg.gpus,
                torch.device('gpu:0')
            )
