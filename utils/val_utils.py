import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
from PIL import Image

def validate_seg(mask, seg, val_stats, bs):
    intersection = (mask & seg).float().sum()
    union = (mask | seg).float().sum()
    acc = torch.eq(mask, seg).sum().float() / len(mask)
    iou = (intersection + 1e-6) / (union + 1e-6)
    val_stats['accs'] +=  acc * bs
    val_stats['ious'] += iou * bs
    val_stats['acc'] = val_stats['accs'] / val_stats['batch_sizes']
    val_stats['iou'] = val_stats['ious'] / val_stats['batch_sizes']
    val_stats['cmatrix'] += confusion_matrix(mask, seg)

if __name__ == '__main__':
    i1 = Image.open(r'C:\Users\Eliton\Documents\master\segsr\data\annotation\2953-3-SO_0_HR.png')
    i2 = Image.open(r'C:\Users\Eliton\Documents\master\segsr\data\annotation\2953-3-SO_1_HR.png')
    i1 = torch.as_tensor(np.array(i1)).unsqueeze(dim=0).flatten()
    i2 = torch.as_tensor(np.array(i2)).unsqueeze(dim=0).flatten()
    validate_seg(i1, i2, None)