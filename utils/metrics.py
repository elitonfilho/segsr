import math
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_seg_stats(mask, seg, metrics_seg):
    acc = (mask == seg).sum()
    labels = np.unique((*np.unique(mask),*np.unique(seg)))
    for i in labels:
        metrics_seg[i]['intersection'] += np.sum(np.logical_and(mask==i, seg==i), keepdims=False)
        metrics_seg[i]['union'] += np.sum(np.logical_or(mask==i, seg==i), keepdims=False)
        metrics_seg[i]['accumulator'] += len(mask[mask==i])

if __name__ == "__main__":

    path_HR = sorted([x.resolve() for x in Path(r'D:\datasets\landcover.ai\val_hr').iterdir() if x.suffix == '.png'])
    path_mask = map(lambda x: Path(r'D:\datasets\landcover.ai\mval_hr', x.name), path_HR)
    path_SR = map(lambda x: Path(r'C:\Users\eliton\Documents\ml\BasicSR\results\EDSR_Lx4_f256b32_DIV2K_official\visualization\lcai', x.name), path_HR)
    path_seg = map(lambda x: Path(r'D:\ml\mrlcai-hrnet-noseg-ow', x.name), path_HR)
    lpips = Path(r'D:\ml\redsr-lcai.txt').open(encoding='utf-8').readlines()
    results = map(lambda x: float(x.replace('\n', '').strip().split(':')[-1]), lpips)
    reduced = sum(results)/len(lpips)

    metrics = {
        'psnr': 0,
        'ssim': 0,
        'lpips': reduced,
        'count': 0
    }

    # metrics_seg = {
    #     i: {
    #         'intersection': 0,
    #         'union': 0,
    #         'accumulator': 0
    #     } for i in range(0,4)
    # }

    # Metrics
    for hr, sr in zip(path_HR, path_SR):
        img_hr = np.array(Image.open(hr))
        img_sr = np.array(Image.open(sr))
        metrics['psnr'] += calculate_psnr(img_sr, img_hr)
        metrics['ssim'] += calculate_ssim(img_sr, img_hr)
        metrics['count'] += 1

    # Metrics seg
    # for mask, seg in zip(path_mask, path_seg):
    #     img_mask = np.array(Image.open(mask))
    #     img_seg = np.array(Image.open(seg))
    #     calculate_seg_stats(img_mask, img_seg, metrics_seg)

    # for values in metrics_seg.values():
    #     values['acc'] = values['intersection'] / values['accumulator']
    #     values['iou'] = values['intersection'] / values['union']


    print('PSNR: {} \nSSIM: {}\nLPIPS: {}'.format(
        metrics['psnr']/metrics['count'],
        metrics['ssim']/metrics['count'],
        metrics['lpips']))
    # print(metrics_seg)
    