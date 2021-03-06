import numpy as np
import torch
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import (CenterCrop, Compose,
                                    Resize, ToPILImage, ToTensor,ToPILImage, Lambda)

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

def tensor2img(tensor, min_max=(0,1), out_type = np.uint8, to_pil = False):
    """Converts torch Tensor(s) to numpy arrays
    
    Clamps the input tensor or a list of tensors to (min,max) and returns a numpy image.

    Args:
        tensor (Tensor or list[Tensor]). Acceptable shape: (C,H,W)

    Returns:
        Tensor or list[Tensor]: Consists of ndarray(s) of shape (H,W,C)
    """
    if not (torch.is_tensor(tensor) or 
            (isinstance(torch, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'Expected a Tensor of list[Tensors], got {type(tensor)}')

    result = []
    if torch.is_tensor(tensor):
        tensor = [tensor]
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        img = _tensor.numpy().transpose(1,2,0)
        if out_type == np.uint8:
            img = (img * 255.0).round()
        img = img.astype(out_type)
        if to_pil:
            img = ToPILImage()(img)
        result.append(img)
    if len(result) == 1:
        result = result[0]
    return result

def compose_val(min_max = (0,1)):
    return Lambda(lambda img: img.squeeze(0).clamp_(*min_max).cpu())
