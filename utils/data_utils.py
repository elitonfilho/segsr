import albumentations as alb
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import (CenterCrop, Compose,
                                    Resize, ToPILImage, ToTensor)

aug = alb.Compose([
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
