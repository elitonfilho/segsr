import abc
import torch
import ignite

class BaseTester(abc.ABC):

    def __init__(self, *args, **kwargs) -> None:
        self.cfg, self.model, self.dataset = args

    @abc.abstractmethod
    def load_state_dict(self, path):
        pass

    @abc.abstractmethod
    def run(self):
        pass