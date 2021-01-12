from os import listdir
from os.path import join
from pathlib import Path

import albumentations as alb
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (CenterCrop, Compose, Normalize, RandomCrop,
                                    Resize, ToPILImage, ToTensor, functional)

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


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor, use_aug=None):
        super(TrainDatasetFromFolder, self).__init__()
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


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor, crop_size):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x)
                                for x in listdir(dataset_dir)]
        self.resize_lr = (crop_size//upscale_factor, crop_size//upscale_factor)

    def __getitem__(self, index):
        with np.load(self.image_filenames[index]) as x:
            load_img = x['arr_0'].squeeze()
        hr_image = (load_img[0:3]).transpose(1, 2, 0)
        size_hr = hr_image.shape[:2]
        lr_image = cv2.resize(hr_image, dsize=self.resize_lr, interpolation=cv2.INTER_CUBIC)
        seg_img = load_img[8]
        hr_restore_img = cv2.resize(lr_image, dsize=size_hr, interpolation=cv2.INTER_CUBIC)
        lr_image = torch.tensor((lr_image/255.).transpose(2, 0, 1), dtype=torch.float32)
        hr_restore_img = torch.tensor((hr_restore_img/255.).transpose(2, 0, 1), dtype=torch.float32)
        hr_image = torch.tensor((hr_image/255.).transpose(2, 0, 1), dtype=torch.float32)
        seg_image = torch.tensor(seg_img, dtype=torch.long)
        return lr_image, hr_restore_img, hr_image, seg_img

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x)
                             for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x)
                             for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w),
                          interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)


def debug():
    train_set = TrainDatasetFromFolder('D:\\de_1m_2013_extended-val_patches', crop_size=256,
                                       upscale_factor=4, use_aug=False)
    # dev_set = ValDatasetFromFolder('D:\\de_1m_2013_extended-val_patches', crop_size=256, upscale_factor=4)
    lr, hr, mask = train_set[0]
    # lr, hr_restore, hr, mask = dev_set[0]
    print(type(lr), lr.dtype, lr.shape)
    print(type(hr), hr.dtype, hr.shape)
    # print(type(hr_restore), hr_restore.shape)
    print(type(mask), mask.shape)

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
