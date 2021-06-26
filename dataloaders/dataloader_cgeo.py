from pathlib import Path

import albumentations as alb
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from albumentations.pytorch import ToTensorV2
from hydra.utils import instantiate
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (CenterCrop, Compose, Normalize, RandomCrop,
                                    Resize, ToPILImage, ToTensor)

aug_train = alb.Compose([
    alb.VerticalFlip(p=0.5),
    alb.HorizontalFlip(p=0.5),
    alb.Transpose(p=0.5),
    alb.RandomRotate90(p=0.5),
    ToTensorV2()
],
    additional_targets={
    'image_lr': 'image'
})

def buildAug(cfg) -> alb.Compose:
    _augList = []
    for aug in cfg:
        _augList.append(instantiate(aug))
    return alb.Compose(
        _augList, 
        additional_targets={
            'image_lr':'image'
    })

def buildAugForSegTask(augCfg) -> alb.Compose:
    _augList = []
    for aug in augCfg:
        _augList.append(instantiate(aug))
    return alb.Compose(
        _augList, 
        additional_targets={
            'image_lr':'image'
    })


class CGEODataset(Dataset):
    def __init__(self, path_lr, path_hr, path_seg, augCfg=None):
        super(CGEODataset, self).__init__()
        path_hr = Path(path_hr)
        path_seg = Path(path_seg)
        filenames = [x.name for x in path_hr.iterdir() if x.suffix in ('.png', '.jpeg')]
        self.hr_images = [path_hr / x for x in filenames]
        self.seg_images = [path_seg / x for x in filenames]
        self.aug = augCfg or buildAug(augCfg)

    def __getitem__(self, index):
        hr_image = np.array(Image.open(self.hr_images[index]), dtype=np.uint8)
        seg_image = np.array(Image.open(self.seg_images[index]), dtype=np.int32)
        if self.aug:
            transformed = self.aug(image=hr_image/255., mask=seg_image)
            lr_image = cv2.resize(hr_image, tuple(reversed(hr_image.shape[:2]/2)), cv2.INTER_CUBIC)
            hr_image = transformed['image']
            seg_image = transformed['mask']
            return lr_image, hr_image, seg_image
        elif not self.aug:
            # TODO: normalize
            return ToTensor()(lr_image), ToTensor()(hr_image), torch.tensor(seg_image, dtype=torch.int32)

    def __len__(self):
        return len(self.hr_images)

class CGEODatasetForSegTask(Dataset):
    def __init__(self, path_hr, path_seg, augCfg=None):
        super(CGEODatasetForSegTask, self).__init__()
        path_hr = Path(path_hr)
        path_seg = Path(path_seg)
        filenames = [x.name for x in path_hr.iterdir() if x.suffix in ('.png', '.jpeg')]
        self.hr_images = [path_hr / x for x in filenames]
        self.seg_images = [path_seg / x for x in filenames]
        print(augCfg)
        self.aug = augCfg or buildAugForSegTask(augCfg)

    def __getitem__(self, index):
        hr_image = np.array(Image.open(self.hr_images[index]), dtype=np.uint8)
        seg_image = np.array(Image.open(self.seg_images[index]), dtype=np.int32)
        if self.aug:
            transformed = self.aug(image=hr_image/255., mask=seg_image)
            hr_image = transformed['image']
            seg_image = transformed['mask']
            return hr_image, seg_image
        elif not self.aug:
            # TODO: normalize
            return ToTensor()(hr_image), torch.tensor(seg_image, dtype=torch.int32)

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
