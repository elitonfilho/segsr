from typing import Sequence
from enum import Enum

import numpy as np
import torch
from ignite.metrics import PSNR, SSIM
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
from scipy.ndimage import uniform_filter
from scipy import signal
from torch.functional import Tensor

class CustomPSNR(PSNR):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        if len(output) == 3:
            output = (output[0].detach(), output[1].detach())
        return super().update(output)

class CustomSSIM(SSIM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        if len(output) == 3:
            output = (output[0].detach(), output[1].detach())
        return super().update(output)

class UQI(Metric):

    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self._num_correct = None
        self._num_examples = None
        super(UQI, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._value = torch.tensor(0, device=self._device, dtype=torch.float)
        self._num_examples = 0
        super(UQI, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]):
        y_pred, y = output[0].detach(), output[1].detach()
        if any((y_pred.dim() > 3, y.dim() > 3)):
            for _y_pred, _y in zip(y_pred[:,...], y[:,...]):
                self.update((_y_pred, _y))
        else:
            y_pred = y_pred.numpy().squeeze().transpose((1,2,0))
            y = y.numpy().squeeze().transpose((1,2,0))
            value = torch.tensor(uqi(y, y_pred), requires_grad=False)
            self._value += torch.sum(value).to(self._device)
            self._num_examples += 1

    @sync_all_reduce("_num_examples", "_value:SUM")
    def compute(self):
        if self._num_examples == 0:
            raise ZeroDivisionError('UQI must have at least one example before it can be computed.')
        return self._value.item() / self._num_examples


# Helper functions from https://github.com/andrewekhalel/sewar

def mse(gt, im):
    return np.mean((gt.astype(np.float64)-im.astype(np.float64))**2)

def rmse(gt, im):
    return np.sqrt(mse(gt, im))

def _rmse_sw_single(gt, im, ws):
    errors = (gt-im)**2
    errors = uniform_filter(errors.astype(np.float64), ws)
    rmse_map = np.sqrt(errors)
    s = int(np.round((ws/2)))
    return np.mean(rmse_map[s:-s, s:-s]), rmse_map

def rmse_sw(gt, im, ws=8):
    rmse_map = np.zeros(gt.shape)
    vals = np.zeros(gt.shape[2])
    for i in range(gt.shape[2]):
        vals[i], rmse_map[:, :, i] = _rmse_sw_single(gt[:, :, i], im[:, :, i], ws)

    return np.mean(vals), rmse_map

def psnr(gt, im, MAX=None):
    if MAX is None:
        MAX = np.iinfo(gt.dtype).max
    mse_value = mse(gt, im)
    if mse_value == 0.:
        return np.inf
    return 10 * np.log10(MAX**2 / mse_value)

def _uqi_single(gt, im, ws):
    N = ws**2
    gt_sq = gt*gt
    P_sq = im*im
    gt_P = gt*im

    gt_sum = uniform_filter(gt, ws)
    P_sum = uniform_filter(im, ws)
    gt_sq_sum = uniform_filter(gt_sq, ws)
    P_sq_sum = uniform_filter(P_sq, ws)
    gt_P_sum = uniform_filter(gt_P, ws)

    gt_P_sum_mul = gt_sum*P_sum
    gt_P_sum_sq_sum_mul = gt_sum*gt_sum + P_sum*P_sum
    numerator = 4*(N*gt_P_sum - gt_P_sum_mul)*gt_P_sum_mul
    denominator1 = N*(gt_sq_sum + P_sq_sum) - gt_P_sum_sq_sum_mul
    denominator = denominator1*gt_P_sum_sq_sum_mul

    q_map = np.ones(denominator.shape)
    index = np.logical_and((denominator1 == 0), (gt_P_sum_sq_sum_mul != 0))
    q_map[index] = 2*gt_P_sum_mul[index]/gt_P_sum_sq_sum_mul[index]
    index = (denominator != 0)
    q_map[index] = numerator[index]/denominator[index]

    s = int(np.round(ws/2))
    return np.mean(q_map[s:-s, s:-s])

def uqi(gt, im, ws=8):
    return np.mean([_uqi_single(gt[:, :, i], im[:, :, i], ws) for i in range(gt.shape[2])])

def _ssim_single(gt, P, ws, C1, C2, fltr_specs, mode):
    win = fspecial(**fltr_specs)

    gt_sum_sq, P_sum_sq, gt_P_sum_mul = _get_sums(gt, P, win, mode)
    sigmagt_sq, sigmaP_sq, sigmagt_P = _get_sigmas(
        gt, P, win, mode, sums=(gt_sum_sq, P_sum_sq, gt_P_sum_mul))

    assert C1 > 0
    assert C2 > 0

    ssim_map = ((2*gt_P_sum_mul + C1)*(2*sigmagt_P + C2)) / \
        ((gt_sum_sq + P_sum_sq + C1)*(sigmagt_sq + sigmaP_sq + C2))
    cs_map = (2*sigmagt_P + C2)/(sigmagt_sq + sigmaP_sq + C2)

    return np.mean(ssim_map), np.mean(cs_map)

def ssim(gt, P, ws=11, K1=0.01, K2=0.03, MAX=None, fltr_specs=None, mode='valid'):
    if MAX is None:
        MAX = np.iinfo(gt.dtype).max
    if fltr_specs is None:
        fltr_specs = dict(fltr=Filter.UNIFORM, ws=ws)

    C1 = (K1*MAX)**2
    C2 = (K2*MAX)**2

    ssims = []
    css = []
    for i in range(gt.shape[2]):
        ssim, cs = _ssim_single(gt[:, :, i], P[:, :, i], ws, C1, C2, fltr_specs, mode)
        ssims.append(ssim)
        css.append(cs)
    return np.mean(ssims), np.mean(css)

def ergas(gt, im, r=4, ws=8):

    rmse_map = None
    nb = 1

    _, rmse_map = rmse_sw(gt, im, ws)

    means_map = uniform_filter(gt, ws)/ws**2

    # Avoid division by zero
    idx = means_map == 0
    means_map[idx] = 1
    rmse_map[idx] = 0

    ergasroot = np.sqrt(np.sum(((rmse_map**2)/(means_map**2)), axis=2)/nb)
    ergas_map = 100*r*ergasroot

    s = int(np.round(ws/2))
    return np.mean(ergas_map[s:-s, s:-s])

class Filter(Enum):
	UNIFORM = 0
	GAUSSIAN = 1

def fspecial(fltr,ws,**kwargs):
	if fltr == Filter.UNIFORM:
		return np.ones((ws,ws))/ ws**2
	elif fltr == Filter.GAUSSIAN:
		x, y = np.mgrid[-ws//2 + 1:ws//2 + 1, -ws//2 + 1:ws//2 + 1]
		g = np.exp(-((x**2 + y**2)/(2.0*kwargs['sigma']**2)))
		g[ g < np.finfo(g.dtype).eps*g.max() ] = 0
		assert g.shape == (ws,ws)
		den = g.sum()
		if den !=0:
			g/=den
		return g
	return None

def _get_sums(GT,P,win,mode='same'):
	mu1,mu2 = (filter2(GT,win,mode),filter2(P,win,mode))
	return mu1*mu1, mu2*mu2, mu1*mu2

def _get_sigmas(GT,P,win,mode='same',**kwargs):
	if 'sums' in kwargs:
		GT_sum_sq,P_sum_sq,GT_P_sum_mul = kwargs['sums']
	else:
		GT_sum_sq,P_sum_sq,GT_P_sum_mul = _get_sums(GT,P,win,mode)

	return filter2(GT*GT,win,mode)  - GT_sum_sq,\
			filter2(P*P,win,mode)  - P_sum_sq, \
			filter2(GT*P,win,mode) - GT_P_sum_mul

def filter2(img,fltr,mode='same'):
	return signal.convolve2d(img, np.rot90(fltr,2), mode=mode)