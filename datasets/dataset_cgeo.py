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
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (CenterCrop, Compose, Normalize, RandomCrop,
                                    Resize, ToPILImage, ToTensor)

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


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

# TODO: Reorganize get_seg_img


def get_seg_img(pathImageHR):
    name = Path(pathImageHR).stem
    pathSeg = Path(pathImageHR, '../../ann-0-3', f'{name}.png').resolve()
    return Image.open(pathSeg)


def get_lr_img(pathImageHR):
    name = Path(pathImageHR).stem
    pathSeg = Path(pathImageHR, '../../lr', f'{name}.png').resolve()
    return Image.open(pathSeg)


class CGEODataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor, use_aug=None):
        super(CGEODataset, self).__init__()
        self.image_filenames = [join(dataset_dir, x)
                                for x in listdir(dataset_dir) if is_image_file(x)]
        self.resize_lr = (crop_size//upscale_factor, crop_size//upscale_factor)
        self.aug = aug_train if use_aug else None

    def __getitem__(self, index):
        hr_image = np.array(Image.open(self.image_filenames[index]), dtype=np.uint8)
        # lr_image = np.array(get_lr_img(self.image_filenames[index]), dtype=np.uint8)
        lr_image = cv2.resize(hr_image, dsize=self.resize_lr, interpolation=cv2.INTER_CUBIC)
        seg_image = np.array(get_seg_img(self.image_filenames[index]), dtype=np.int32)
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
        return len(self.image_filenames)


def debug():
    train_set = CGEODataset(
        'C:\\Users\\eliton\\Documents\\ml\\datasets\\train',
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
