from pathlib import Path

import albumentations as alb
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (CenterCrop, Compose, Normalize, RandomCrop,
                                    Resize, ToPILImage, ToTensor)
from omegaconf import DictConfig
from hydra.utils import instantiate

aug_train = alb.Compose([
    alb.VerticalFlip(p=0.5),
    alb.HorizontalFlip(p=0.5),
    alb.Transpose(p=0.5),
    alb.RandomRotate90(p=0.5),
    # ToTensor(num_classes=4, sigmoid=True, normalize={
    #     'mean': [0.1593, 0.2112, 0.1966],
    #     'std': [0.1065, 0.0829, 0.0726]
    # })
    # alb.Normalize(mean=(40.6193, 53.8484, 50.1273), std=(27.1632, 21.1268, 18.5095)),
    ToTensorV2()
],
    additional_targets={
    'image_lr': 'image'
})

def buildAug(cfg) -> alb.Compose:
    _augList = []
    for aug in cfg.dataloader.augmentation:
        _augList.append(instantiate(aug))
    return alb.Compose(_augList)


class CGEODataset(Dataset):
    def __init__(self, path_lr, path_hr, path_seg=None, aug=None):
        super(CGEODataset, self).__init__()
        path_lr = Path(path_lr)
        path_hr = Path(path_hr)
        path_seg = Path(path_seg)
        filenames = [x.name for x in Path(path_hr).iterdir() if x.suffix in ('.png', '.jpeg')]
        self.lr_images = [path_lr / x for x in filenames]
        self.hr_images = [path_hr / x for x in filenames]
        self.seg_images = [path_seg / x for x in filenames]
        print(aug)

    def __getitem__(self, index):
        lr_image = np.array(Image.open(self.lr_images[index]), dtype=np.uint8)
        hr_image = np.array(Image.open(self.hr_images[index]), dtype=np.uint8)
        seg_image = np.array(Image.open(self.seg_images[index]), dtype=np.int32)
        if self.aug:
            transformed = self.aug(image=hr_image/255., image_lr=lr_image/255., mask=seg_image)
            lr_image = transformed['image_lr']
            hr_image = transformed['image']
            seg_image = transformed['mask']
            return lr_image, hr_image, seg_image
        elif not self.aug:
            # TODO: normalize
            return ToTensor()(lr_image), ToTensor()(hr_image), torch.tensor(seg_image, dtype=torch.int32)

    def __len__(self):
        return len(self.hr_images)


def debug():
    train_set = CGEODataset(
        'D:\datasets\cgeo\lr',
        'D:\datasets\cgeo\hr',
        r'D:\datasets\cgeo\annotation'
        )

    lr, hr, mask = train_set[0]
    print(type(lr), lr.dtype, lr.shape)
    print(type(hr), hr.dtype, hr.shape)
    print(type(mask), mask.dtype, mask.shape)

    fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(hr.astype(np.uint8))
    # ax[1].imshow(lr)
    ax[0].imshow(lr.permute(1, 2, 0))
    ax[1].imshow(hr.permute(1, 2, 0))
    # ax[3].imshow(hr_restore.permute(1, 2, 0))
    ax[2].imshow(mask)
    plt.show()


if __name__ == "__main__":
    debug()
