from os import listdir
from os.path import join
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import albumentations as alb
from albumentations.pytorch import ToTensorV2


aug_train = alb.Compose([
    alb.VerticalFlip(p=0.5),
    alb.HorizontalFlip(p=0.5),
    alb.Transpose(p=0.5),
    alb.Transpose(p=0.5),
    alb.RandomRotate90(p=0.5),
    ToTensorV2()
],
    additional_targets={
    'image_lr': 'image'
})


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def get_seg_img(pathImageHR, val=False):
    name = Path(pathImageHR).stem
    pathSeg = Path(pathImageHR, '../../annotation', f'{name}.png').resolve()
    if val:
        return Image.open(pathImageHR)
    else:
        return Compose([
            CenterCrop(256)
        ])(Image.open(pathSeg))


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
                                for x in listdir(dataset_dir) if is_image_file(x)]
        self.resize_lr = (crop_size//upscale_factor, crop_size//upscale_factor)
        self.aug = aug_train if use_aug else None

    def __getitem__(self, index):
        hr_image = np.asarray(Image.open(self.image_filenames[index]), dtype=np.uint8)
        lr_image = cv2.resize(hr_image, dsize=self.resize_lr, interpolation=cv2.INTER_CUBIC)
        seg_image = np.asarray(get_seg_img(self.image_filenames[index]), dtype=np.long)
        if self.aug:
            transformed = self.aug(image=hr_image, image_lr=lr_image, mask=seg_image)
            lr_image = transformed['image_lr']
            hr_image = transformed['image']
            seg_image = transformed['mask']
        return lr_image, hr_image, seg_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x)
                                for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        seg_img = get_seg_img(self.image_filenames[index], val=True)
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image), ToTensor()(seg_img)

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

        return len(self.lr_filenames)


def debug():
    train_set = TrainDatasetFromFolder('data/train', crop_size=256,
                                       upscale_factor=4, use_aug=True)
    hr, lr, mask = train_set[0]
    print(type(hr), hr.shape)
    print(type(lr), lr.shape)
    print(type(mask), mask.shape)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(hr.permute(1,2,0))
    ax[1].imshow(lr.permute(1,2,0))
    ax[2].imshow(mask)
    plt.show()


if __name__ == "__main__":
    debug()
