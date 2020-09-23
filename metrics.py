import math
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
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
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

if __name__ == "__main__":
    # img1 = Image.open('data/HR/2953-3-SO_0_HR.png').resize((256,256), resample=Image.BICUBIC)
    # img1 = np.array(img1)
    # img2 = Image.open('1e-2.png').resize((256,256), resample=Image.BICUBIC)
    img2 = Image.open('lr.png')
    img2 = np.array(img2)
    imgHR = Image.open('data/HR/2953-3-SO_0_HR.png')
    # w, h = imgHR.size
    # imgHR = imgHR.resize((4*w, 4*h), resample=Image.BICUBIC)
    imgHR = np.array(imgHR)
    # 1: Com seg, 2: Sem Seg
    # print('1 e 2', calculate_psnr(img1, img2))
    print('1 e HR', calculate_psnr(img2, imgHR))
    # print('2 e HR', calculate_psnr(img2, imgHR))
    # print('1 e 2', calculate_ssim(img1, img2))
    print('1 e HR', calculate_ssim(img2, imgHR))
    # print('2 e HR', calculate_ssim(img2, imgHR))