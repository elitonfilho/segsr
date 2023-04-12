# Segsr
[![Torch](https://img.shields.io/badge/-PyTorch-red?logo=pytorch&labelColor=gray)](https://pytorch.org/get-started/locally/)
[![Hydra](https://img.shields.io/badge/conf-hydra-blue)](https://hydra.cc/)
[![Ignite](https://img.shields.io/badge/-Ignite-orange?logo=pytorch&labelColor=gray)](https://pytorch-ignite.ai)

This repository contains the official implementation of the paper "Joint-task learning to improve perceptually-aware super-resolution of aerial images" from the IJRS Vol.44. The article is avalaible [here](https://doi.org/10.1080/01431161.2023.2190469).

SegSR focus on improving image super-resolution by using a segmentation module capable of verifying the performance of the generator module. The extra criterion provided by the seg module captures well the reconstruction quality of synthesized images, especially when considering perceptual metrics, such as LPIPS (Learned Perceptual Image Patch Similarity) or PI (Perceptual Index).

If you find our work useful in your search, don't forget to cite it! You can use the BibTeX entry in the end of this Readme.

## Dependencies
Dependencies requirements are met at `requirements.txt` file.
Just run `pip install -r requirements.txt` to install them!

## Quick start
- Clone the repository: `git clone https://github.com/elitonfilho/segsr`
- Install dependencies using `pip install -r requirements.txt`
- (Optional) Create your own model / dataloader / trainer / configuration files!
    - Dataloaders are defined by a conventional `torch.utils.data.dataset.Dataset` class. You can also create your own dataloader with augmentation policies, such as our example in [dataloaders/dataloader_cgeo.py](dataloaders/dataloader_cgeo.py) which uses `albumentations` as augmentation library
    - Models (and loss functions) are defined inside the folder `models`. You can use any pre-existent models / loss functions or implement a new one!
    - Trainers (or testers) describes the behavior of the training (or testing) procedure. They are responsible in organizing the train/validation steps, saving/loading checkpoints, logging or any other desired functionality! We employ [Pytorch Ignite](https://pytorch.org/ignite/index.html) as high-level framework on top of Pytorch to help us with distributed training, checkpoint management, event-based training procedure and much more!.
    Examples are found at [trainer/ignite_trainer.py](trainer/ignite_trainer.py) or [trainer/ignite_tester.py](trainer/ignite_tester.py)
    - We use [Hydra](https://hydra.cc) as a configuration framework to define our experiments parameters. Default configuration, parameter overriding, multirun and parameter sweeping are just a few of its interesting tools! Check the [docs](https://hydra.cc/docs/intro/) and some examples at [config/](config/)
- Run the code! Some examples:
```python
# Train on multiple GPUs (default)
python3 main.py archs=segsr-unet trainer=segsr-trainer trainer.num_epochs=100 trainer.path_pretrained_seg='pretrained-segnet-here.pt' trainer.path_pretrained_sr='pretrained-srnet-here.pt' trainer.batch_size=32 trainer.validation.batch_size=32

# Train on specific GPUs (GPU:0 and GPU:1 for example)
CUDA_VISIBLE_DEVICES=0,1 python3 main.py gpus=[0,1] name=segsr-sagan-hrnet-lcai dataloader=sr-lcai-server trainer=segsr-sagan-trainer archs=segsr-sagan-hrnet trainer.losses.seg.loss_weight=0.1 trainer.path_pretrained_seg='/mnt/data/eliton/results/zoo/pretrained-hrnet-lcai.pt' trainer.path_pretrained_sr='/mnt/data/eliton/results/zoo/pretrained-sr-lcai.pt' trainer.num_epochs=50

# Train with parameter sweep on loss_weight
python3 main.py -m trainer.losses.seg.loss_weight=5e-4,1e-3,5e-3,1e-2 trainer=segsr-trainer dataloader=sr-lcai-server trainer.num_epochs=50 archs=segsr-hrnet trainer.path_pretrained_seg='zoo/new-pretrained-seg-hrnet-lcai.pt' trainer.path_pretrained_sr='zoo/new-pretrained-sr-lcai.pt' trainer.batch_size=14 trainer.validation.batch_size=14

# Inference on a single GPU
CUDA_VISIBLE_DEVICES=0 python3 main.py mode=test gpus=[0] tester=sr-tester archs=sr-sagan tester.path_pretrained="trained-model-here" tester.save_path="/mnt/data/inference_path" tester.savefig_mode=sronly

```

## Supported architectures
- SR:
    - [SRGAN](https://arxiv.org/abs/1609.04802)
    - [ESRGAN](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf)
    - [ABPN](https://arxiv.org/abs/1910.04476)
    - [CSNLN](https://arxiv.org/abs/2006.01424)
    - [DBPN](https://openaccess.thecvf.com/content_cvpr_2018/papers/Haris_Deep_Back-Projection_Networks_CVPR_2018_paper.pdf)
    - [DRLN](https://arxiv.org/abs/1906.12021)
    - [EDSR](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf)
    - [PAEDSR](https://arxiv.org/abs/2004.13824)
    - [RCAN](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.pdf)
    - [RDN](https://arxiv.org/abs/1802.08797)
    - [SAGAN](http://proceedings.mlr.press/v97/zhang19d/zhang19d.pdf)
- Segmentation:
    - [UNet](https://arxiv.org/pdf/1505.04597.pdf)
    - [HRNet](https://arxiv.org/abs/1908.07919)


## Citation
```latex
@article{doi:10.1080/01431161.2023.2190469,
author = {J. E. Albuquerque F. and C. R. Jung},
title = {Joint-task learning to improve perceptually-aware super-resolution of aerial images},
journal = {International Journal of Remote Sensing},
volume = {44},
number = {6},
pages = {1820-1841},
year  = {2023},
publisher = {Taylor & Francis},
doi = {10.1080/01431161.2023.2190469},
URL = {https://doi.org/10.1080/01431161.2023.2190469},
eprint = {https://doi.org/10.1080/01431161.2023.2190469}}

```

## Acknowledgements
We are grateful for using network implementations from multiple repositories, either mentioned in the model file of here:

https://github.com/leftthomas/SRGAN

https://github.com/xinntao/BasicSR

https://github.com/stefanopini/simple-HRNet