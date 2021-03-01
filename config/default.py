from yacs.config import CfgNode as CN

_C = CN()

_C.DIR = 'results/experiment1'

_C.DATASET = CN()
_C.DATASET.type = 'cgeo'
_C.DATASET.train_dir = 'data/train'
_C.DATASET.val_dir = 'data/val'
_C.DATASET.seg_dir = 'data/annotation'
_C.DATASET.n_classes = 5
_C.DATASET.weight_classes = None

_C.ARCHS = CN()

_C.ARCHS.netG = CN()
_C.ARCHS.netG.type = 'RRDBNet'
_C.ARCHS.netG.num_in_ch = 3
_C.ARCHS.netG.num_out_ch = 3
_C.ARCHS.netG.num_feat = 64
_C.ARCHS.netG.num_block = 23

_C.ARCHS.netD = CN()
_C.ARCHS.netD.type = 'VGG128'
_C.ARCHS.netD.num_in_ch = 3
_C.ARCHS.netD.num_feat = 64

_C.ARCHS.netSeg = CN()
_C.ARCHS.netSeg.type = 'UNet'

_C.TRAIN = CN()
_C.TRAIN.use_seg = True
_C.TRAIN.arch_enc = 'hrnet'
_C.TRAIN.use_pretrained_seg = True
_C.TRAIN.path_pretrained_seg = ''
_C.TRAIN.use_pretrained_sr = False
_C.TRAIN.path_pretrained_g = ''
_C.TRAIN.path_pretrained_d = ''
_C.TRAIN.begin_seg = 0.7
_C.TRAIN.num_epochs = 100
_C.TRAIN.batch_size = 10
_C.TRAIN.crop_size = 256
_C.TRAIN.upscale_factor = 4
_C.TRAIN.lr = 1e-3
_C.TRAIN.model_save_path = 'epochs/'
_C.TRAIN.model_name = 'model'
_C.TRAIN.save_best = ''
_C.TRAIN.use_aug = None
_C.TRAIN.scheduler_milestones = None
_C.TRAIN.scheduler_gamma = None

_C.TRAIN.losses = CN()
_C.TRAIN.losses.il = 1.0
_C.TRAIN.losses.adv = 0.001
_C.TRAIN.losses.per = 0.006
_C.TRAIN.losses.tv = 2e-8
_C.TRAIN.losses.seg = 1e-3

_C.hrnet = CN()
_C.hrnet.FINAL_CONV_KERNEL = 1
_C.hrnet.STAGE1 = CN()
_C.hrnet.STAGE1.NUM_MODULES = 1
_C.hrnet.STAGE1.NUM_RANCHES = 1
_C.hrnet.STAGE1.BLOCK = 'BOTTLENECK'
_C.hrnet.STAGE1.NUM_BLOCKS = [4]
_C.hrnet.STAGE1.NUM_CHANNELS = [64]
_C.hrnet.STAGE1.FUSE_METHOD = 'SUM'
_C.hrnet.STAGE2 = CN()
_C.hrnet.STAGE2.NUM_MODULES = 1
_C.hrnet.STAGE2.NUM_BRANCHES = 2
_C.hrnet.STAGE2.BLOCK = 'BASIC'
_C.hrnet.STAGE2.NUM_BLOCKS = [4, 4]
_C.hrnet.STAGE2.NUM_CHANNELS = [48, 96]
_C.hrnet.STAGE2.FUSE_METHOD = 'SUM'
_C.hrnet.STAGE3 = CN()
_C.hrnet.STAGE3.NUM_MODULES = 4
_C.hrnet.STAGE3.NUM_BRANCHES = 3
_C.hrnet.STAGE3.BLOCK = 'BASIC'
_C.hrnet.STAGE3.NUM_BLOCKS = [4, 4, 4]
_C.hrnet.STAGE3.NUM_CHANNELS = [48, 96, 192]
_C.hrnet.STAGE3.FUSE_METHOD = 'SUM'
_C.hrnet.STAGE4 = CN()
_C.hrnet.STAGE4.NUM_MODULES = 3
_C.hrnet.STAGE4.NUM_BRANCHES = 4
_C.hrnet.STAGE4.BLOCK = 'BASIC'
_C.hrnet.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
_C.hrnet.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
_C.hrnet.STAGE4.FUSE_METHOD = 'SUM'

_C.VAL = CN()
_C.VAL.visualize = False
_C.VAL.path_save = 'results/'
_C.VAL.batch_size = 5
_C.VAL.n_chunks = 5
_C.VAL.freq = 10

_C.TEST = CN()
_C.TEST.batch_size = 1
_C.TEST.path_encoder = 'epochs/encoder.pth'
_C.TEST.upscale_factor = 4
_C.TEST.path_image = 'data/test'
_C.TEST.path_save = 'results/'
_C.TEST.prefix_save = ''
