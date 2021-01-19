from os import listdir
from os.path import join
from pathlib import Path

import albumentations as alb
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (CenterCrop, Compose, Normalize, RandomCrop,
                                    Resize, ToPILImage, ToTensor, functional)


class LandCoverDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor, use_aug=None):
        super(LandCoverDataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x)
                                for x in listdir(dataset_dir)]
        self.resize_lr = (crop_size//upscale_factor, crop_size//upscale_factor)
        self.aug = aug_train if use_aug else None

    def __getitem__(self, index):
        with np.load(self.image_filenames[index]) as x:
            load_img = x['arr_0'].squeeze()
        hr_image = (load_img[0:3]).transpose(1, 2, 0).astype(np.uint8)
        lr_image = cv2.resize(hr_image, dsize=self.resize_lr, interpolation=cv2.INTER_CUBIC)
        lr_image = lr_image.astype(np.uint8)
        seg_image = load_img[8]
        if self.aug:
            transformed = self.aug(image=hr_image/255., image_lr=lr_image/255., mask=seg_image)
            lr_image = transformed['image_lr']
            hr_image = transformed['image']
            seg_image = transformed['mask']
        elif not self.aug:
            # TODO: normalize
            hr_image = ToTensor()(hr_image)
            lr_image = ToTensor()(lr_image)
            seg_image = torch.tensor(seg_image)
            return lr_image, hr_image, seg_image

    def __len__(self):
        return len(self.image_filenames)


def debug():
    train_set = LandCoverDataset(
        'D:\\de_1m_2013_extended-val_patches',
        crop_size=256,
        upscale_factor=4,
        use_aug=False)
    
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
