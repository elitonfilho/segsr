import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt

def validate_seg(mask, seg, val_stats, bs, cfg):
    seg = torch.argmax(seg, dim=1)
    intersection = (mask & seg).sum()
    union = (mask | seg).sum()
    acc = torch.eq(mask, seg).sum().float() / len(mask)
    iou = (intersection + 1e-6) / (union + 1e-6)
    val_stats['accs'] +=  acc * bs
    val_stats['ious'] += iou * bs
    val_stats['acc'] = (val_stats['accs'] / val_stats['batch_sizes']).item()
    val_stats['iou'] = (val_stats['ious'] / val_stats['batch_sizes']).item()
    plt.imshow(seg.cpu().squeeze())
    plt.show()
    val_stats['cmatrix'] += confusion_matrix(
        mask.cpu().reshape(-1),
        seg.detach().cpu().reshape(-1),
        labels=list(range(cfg.DATASET.n_classes)))