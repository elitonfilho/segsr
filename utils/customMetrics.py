from ignite.metrics import PSNR, SSIM
from ignite.metrics.metric import reinit__is_reduced
from typing import Sequence
import torch

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