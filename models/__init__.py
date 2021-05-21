import importlib
import logging
from hydra.utils import instantiate
from omegaconf import DictConfig
from typing import Dict

logger = logging.getLogger(__name__)

# TODO: support dynamic import with importlib
def import_modelsv1(cfg):
    models = []
    for model in cfg.archs:
        module = importlib.import_module(f'models.model_{model.type}')
        try:
            cls = getattr(module, model.type)
        except ValueError:
            logger.critical('Did not find the specified module: ', model, model.type)
        return cls(model)

def get_models(cfg: DictConfig) -> Dict:
    models = {}
    for model in cfg.archs:
        print(model)
        models.update({model: instantiate(cfg.archs.get(model))})
    return models