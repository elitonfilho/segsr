import numpy as np
import torch

def tensor2img(tensor, min_max=(0,1), out_type = np.uint8):
    """Converts torch Tensor(s) to numpy arrays
    
    Clamps the input tensor or a list of tensors to (min,max) and returns a numpy image.

    Args:
        tensor (Tensor or list[Tensor]). Acceptable shape: (C,H,W)

    Returns:
        Tensor or list[Tensor]: Consists of ndarray(s) of shape (H,W,C)
    """
    result = []
    if torch.is_tensor(tensor):
        tensor = [tensor]
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        img_np = _tensor.numpy().transpose(1,2,0)
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result
