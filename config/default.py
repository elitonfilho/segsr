from yacs.config import CfgNode as CN

_C = CN()

_C.DIR = 'results/experiment1'

_C.DATASET = CN()
_C.DATASET.train_dir = 'data/train'
_C.DATASET.val_dir = 'data/val'
_C.DATASET.seg_dir = 'data/annotation'
_C.DATASET.n_classes = 5

_C.TRAIN = CN()
_C.TRAIN.use_seg = True
_C.TRAIN.arch_enc = 'hrnet'
_C.TRAIN.begin_seg = 0.7
_C.TRAIN.num_epochs = 100
_C.TRAIN.batch_size = 10
_C.TRAIN.crop_size = 256
_C.TRAIN.visualize = False
_C.TRAIN.upscale_factor = 4
_C.TRAIN.model_save_path = 'epochs/'
_C.TRAIN.model_name = 'model'

_C.VAL = CN()
_C.VAL.batch_size = 5
_C.VAL.n_rows = 5

_C.TEST = CN()
_C.TEST.batch_size = 1
_C.TEST.path_encoder = 'epochs/encoder.pth'
_C.TEST.path_decoder = 'epochs/decoder.pth'
