import importlib
from pathlib import Path
from .dataset_cgeo import CGEODataset
from .dataset_landcover import LandCoverDataset
from .dataset_landcoverai import LandCoverAIDataset

# TODO: Refactor archs to accept cfg 
# data_folder = Path(__file__).resolve().parent
# filenames = (x for x in data_folder.iterdir() if 'dataset' in x.stem)
# datasets = [importlib.import_module(f'data.{x}') for x in filenames]

# def get_dataset(cfg):
#     chosen_dataset = cfg.DATASET.type
#     for dataset in datasets:
#         data_cls = getattr(dataset, chosen_dataset, None)
#         if data_cls is not None:
#             break
#     if data_cls is None:
#         raise ValueError(f'Implementation of {chosen_dataset} was not found')

#     dataset = data_cls(cfg)
    
#     return dataset
