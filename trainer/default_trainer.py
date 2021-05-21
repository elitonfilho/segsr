from .base_trainer import BaseTrainer

class DefaultTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super(DefaultTrainer, self).__init__()