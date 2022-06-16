# Segsr
[![Torch](https://img.shields.io/badge/-PyTorch-red?logo=pytorch&labelColor=gray)](https://pytorch.org/get-started/locally/)
[![Hydra](https://img.shields.io/badge/conf-hydra-blue)](https://hydra.cc/)
[![Ignite](https://img.shields.io/badge/-Ignite-orange?logo=pytorch&labelColor=gray)](https://pytorch-ignite.ai)
GAN-based Image Super Resolution improved by semantic segmentation

SegSR focus on improving image super-resolution by using a segmentation module capable of verifying the performance of the generator module. The extra criterion provided by the seg module captures well the reconstruction quality of synthesized images, especially when considering perceptual metrics, such as LPIPS (Learned Perceptual Image Patch Similarity) or PI (Perceptual Index).

## Dependencies
Dependencies requirements are met at `requirements.txt` file.
Just run `pip install -r requirements.txt` to install them!

## Quick start
- Clone the repository: `git clone https://github.com/elitonfilho/segsr`
- Install dependencies using `pip install -r requirements.txt`
- (Optional) Create your own model / dataloader / trainer / configuration files!
    - Dataloaders are defined by a conventional `torch.utils.data.dataset.Dataset` class. Example: [dataloaders/dataloader_cgeo.py](dataloaders/dataloader_cgeo.py)
    - Models (and loss functions) are defined inside the folder `models`. You can use any of them of define your own!
    - Trainers (or testers) describes the behavior of the training (or testing) procedure. In there you will organize the train/validation steps, saving/loading checkpoints, logging or any other desired functionality! We employ [Pytorch Ignite](https://pytorch.org/ignite/index.html) as high-level framework on top of Pytorch to help us with distributed training, checkpoint management, event-based training procedure and much more!.
    Examples are found at [trainer/ignite_trainer.py](trainer/ignite_trainer.py) or [trainer/ignite_tester.py](trainer/ignite_tester.py)
    - We use [Hydra](https://hydra.cc) as a configuration framework to define our experiments parameters. Default configuration, parameter overriding, multirun and parameter sweeping are just a few of its interesting tools! Check the [docs](https://hydra.cc/docs/intro/) and some examples at [config/](config/)
- Run the code! Example:
```
python3 main.py archs=segsr-unet trainer=segsr-trainer trainer.num_epochs=100 trainer.path_pretrained_seg='/mnt/data/eliton/zoo/pretrained-unet.pt' trainer.path_pretrained_sr='/mnt/data/eliton/zoo/pretrained-sr.pt' trainer.batch_size=32 trainer.validation.batch_size=32
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

## Acknowledgements
We are grateful for using network implementations from multiple repositories, either mentioned in the model file of here:

https://github.com/leftthomas/SRGAN
https://github.com/xinntao/BasicSR
https://github.com/stefanopini/simple-HRNet