from scripts.train import train
from .base_trainer import BaseTrainer
from hydra.utils import instantiate
import logging

logger = logging.getLogger('main')

class DefaultTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def setup_optimizers(self):
        return super().setup_optimizers()

    def setup_schedulers(self):
        return super().setup_schedulers()

    def fit(self):
        pass