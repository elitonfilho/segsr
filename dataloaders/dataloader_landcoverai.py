from pathlib import Path

import albumentations as alb
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from hydra.utils import instantiate
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor


def buildAug(augCfg) -> alb.Compose:
    _augList = []
    if augCfg:
        for aug in augCfg:
            _augList.append(instantiate(augCfg.get(aug)))
        return alb.Compose(
            _augList,
            additional_targets={
                'image_lr': 'image'
            })

def buildAugForSegTask(augCfg) -> alb.Compose:
    _augList = []
    if augCfg:
        for aug in augCfg:
            _augList.append(instantiate(augCfg.get(aug)))


class LandCoverAIDataset(Dataset):

    def __init__(self, path_hr, path_lr, path_seg,  augCfg=None):
        super(LandCoverAIDataset, self).__init__()
        path_hr = Path(path_hr)
        path_lr = Path(path_lr)
        path_seg = Path(path_seg)
        filenames = [x.name for x in path_hr.iterdir() if x.suffix in ('.png', '.jpeg')]
        self.hr_images = [path_hr / x for x in filenames]
        self.lr_images = [path_lr / x for x in filenames]
        self.seg_images = [path_seg / x for x in filenames]
        self.aug = buildAug(augCfg)

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, index):
        hr_image = np.array(Image.open(self.hr_images[index]), dtype=np.uint8)
        lr_image = np.array(Image.open(self.lr_images[index]), dtype=np.uint8)
        seg_image = np.array(Image.open(self.seg_images[index]), dtype=np.int32)
        if self.aug:
            transformed = self.aug(image=hr_image/255., image_lr=lr_image/255., mask=seg_image)
            hr_image = transformed['image']
            lr_image = transformed['image_lr']
            seg_image = transformed['mask'].long()
        else:
            lr_image = ToTensor()(lr_image)
            hr_image = ToTensor()(hr_image)
            seg_image = torch.tensor(np.array(seg_image, dtype=np.int32))
        return lr_image, hr_image, seg_image, self.hr_images[index].stem


class LandCoverAIDatasetForSegTask(Dataset):

    def __init__(self, path_hr, path_seg, augCfg=None):
        super(LandCoverAIDatasetForSegTask, self).__init__()
        path_hr = Path(path_hr)
        path_seg = Path(path_seg)
        filenames = [x.name for x in path_hr.iterdir() if x.suffix in ('.png', '.jpeg')]
        self.hr_images = [path_hr / x for x in filenames]
        self.seg_images = [path_seg / x for x in filenames]
        self.aug = buildAugForSegTask(augCfg)

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, index):
        hr_image = np.array(Image.open(self.hr_images[index]), dtype=np.uint8)
        seg_image = np.array(Image.open(self.seg_images[index]), dtype=np.int32)
        if self.aug:
            transformed = self.aug(image=hr_image/255., mask=seg_image)
            hr_image = transformed['image']
            seg_image = transformed['mask'].long()
        else:
            hr_image = ToTensor()(hr_image)
            seg_image = torch.tensor(np.array(seg_image, dtype=np.int32))
        return hr_image, seg_image, self.hr_images[index].stem
