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
_C.TRAIN.use_pretrained_seg = True
_C.TRAIN.path_pretrained_seg = ''
_C.TRAIN.use_pretrained_sr = False
_C.TRAIN.path_pretrained_sr = ''
_C.TRAIN.begin_seg = 0.7
_C.TRAIN.num_epochs = 100
_C.TRAIN.batch_size = 10
_C.TRAIN.crop_size = 256
_C.TRAIN.visualize = False
_C.TRAIN.upscale_factor = 4
_C.TRAIN.lr = 1e-3
_C.TRAIN.model_save_path = 'epochs/'
_C.TRAIN.model_name = 'model'
_C.TRAIN.use_aug = None
_C.TRAIN.scheduler_milestones = None
_C.TRAIN.scheduler_gamma = None

_C.TRAIN.loss_factor = CN()
_C.TRAIN.loss_factor.il = 1.0
_C.TRAIN.loss_factor.adv = 0.001
_C.TRAIN.loss_factor.per = 0.006
_C.TRAIN.loss_factor.tv = 2e-8
_C.TRAIN.loss_factor.seg = 1e-3

_C.VAL = CN()
_C.VAL.batch_size = 5
_C.VAL.n_rows = 5

_C.TEST = CN()
_C.TEST.batch_size = 1
_C.TEST.path_encoder = 'epochs/encoder.pth'
_C.TEST.path_decoder = 'epochs/decoder.pth'
