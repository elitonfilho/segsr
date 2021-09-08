from pathlib import Path
from typing import List, Sequence
import numpy as np

import ignite.distributed as idist
import torch
from hydra.utils import instantiate
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import Checkpoint
from ignite.utils import setup_logger
from torch.functional import Tensor
from torch.nn import Module
from torch.profiler import ProfilerActivity, profile, record_function
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from .base_tester import BaseTester


class IgniteTester(BaseTester):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.netG: Module = self.model['netG'].cuda().eval()
        self.netG = idist.auto_model(self.netG)
        self.plot = None

    def load_state_dict(self, path):
        loadDict = {
            'netG': self.netG
            }
        ckpt = torch.load(path, map_location=f'cuda:{idist.get_rank()}')
        Checkpoint.load_objects(to_load=loadDict, checkpoint=ckpt)

    def setup_handlers(self, engine: Engine):
        engine.add_event_handler(Events.STARTED, self.load_state_dict, self.cfg.tester.path_pretrained)
        pbar = ProgressBar()
        pbar.attach(engine)

    def generate_plot(self, name: str, *images: Sequence[torch.tensor]):
        if not self.plot:
            self.plot = plt.subplots(1,len(images))
        fig, axes = self.plot
        img_hr, result, scaled_lr, label_hr = images
        with torch.no_grad():
            im1 = img_hr.squeeze().numpy().transpose((1,2,0))
            im2 = result.cpu().squeeze().numpy().transpose((1,2,0))
            im3 = scaled_lr.squeeze().numpy().transpose((1,2,0))
            im4 = label_hr.squeeze().numpy()
            images = (im1, im2, im3, im4)
        for ax, img in zip(axes, images):
            ax.axis('off')
            ax.imshow(img)
        fig.subplots_adjust(wspace=0)
        plt.savefig(f'{self.cfg.tester.save_path}/{name[0]}', bbox_inches='tight', pad_inches=0)

    def generate_grid(self, name, *images):
        img_hr, result, scaled_lr, label_hr = images
        stack = torch.cat((scaled_lr,img_hr, result.cpu())).squeeze()
        grid = make_grid(stack, nrow=3)
        p = Path(self.cfg.tester.save_path)
        save_image(grid,p / f'{name[0]}.png', format='png')

    def run_test(self, engine: Engine, batch: List[Tensor]):
        img_lr, img_hr, label_hr, name = batch
        result = self.netG(img_lr.float().cuda())
        scaled_lr = torch.nn.functional.interpolate(img_lr, (256,256), mode='bicubic')
        if self.cfg.tester.get('savefig_mode', None) == 'matplotlib':
            self.generate_plot(name, img_hr, result, scaled_lr, label_hr)
        elif self.cfg.tester.get('savefig_mode', None) == 'torchvision':
            self.generate_grid(name, img_hr, result, scaled_lr, label_hr)
        return result.float().cpu(), img_hr.float().cpu()

    def transformFunctionType1(self, *values):
        return values[1], values[0]

    def setupMetrics(self, engine: Engine):
        for metric in self.cfg.tester.metrics:
            _instance = instantiate(self.cfg.tester.metrics.get(metric))
            _instance.attach(engine, metric)
        
    def run(self):
        dataloader = idist.auto_dataloader(self.dataset['test'], batch_size=self.cfg.tester.batch_size)
        tester = Engine(self.run_test)
        self.setup_handlers(tester)
        self.setupMetrics(tester)
        tester.logger = setup_logger('Tester')
        Path(self.cfg.tester.save_path).mkdir(exist_ok=True)
        results = tester.run(dataloader)
        tester.logger.info(results.metrics.get('uqi'))
        tester.logger.info(results.metrics.get('psnr'))
        tester.logger.info(results.metrics.get('ssim'))
