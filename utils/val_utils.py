from math import log10

import numpy as np
import torch

def validate_seg(mask, seg, cfg):
    val_stats = {

    }
    intersection = (mask & seg).float().sum((1,2))
    union = (mask | seg).float().sum((1,2))
    acc = torch.eq(mask, seg).sum()/len(mask)
    iou = (intersection + 1e-6) / (union + 1e-6)

    raise NotImplementedError