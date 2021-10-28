from pathlib import Path
from typing import List

import cv2
import ignite.distributed as idist
import numpy as np
import torch
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import Checkpoint
from ignite.utils import setup_logger
from torch.functional import Tensor
from torch.nn import Module
from torchvision.utils import make_grid, save_image

from .base_tester import BaseTester


class IgniteTesterSeg(BaseTester):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.netSeg: Module = self.model['netSeg'].cuda().eval()
        self.netSeg = idist.auto_model(self.netSeg)

    def load_state_dict(self, path):
        loadDict = {
            'netSeg': self.netSeg
            }
        ckpt = torch.load(path, map_location=f'cuda:{idist.get_rank()}')
        Checkpoint.load_objects(to_load=loadDict, checkpoint=ckpt)

    def setup_handlers(self, engine: Engine):
        engine.add_event_handler(Events.STARTED, self.load_state_dict, self.cfg.tester.path_pretrained)
        pbar = ProgressBar()
        pbar.attach(engine)

    def generate_grid(self, name, *images):
        label_hr, label_seg = images
        stack = torch.stack((label_hr.float(), label_seg.float()))
        grid = make_grid(stack, nrow=2)
        p = Path(self.cfg.tester.save_path)
        save_image(grid, p / f'{name[0]}.png', format='png')

    def save_figure(self, name: list, image: Tensor):
        image = image.squeeze()
        if image.ndim == 3:
            for idx, img in enumerate(image[:,...]):
                self.save_figure([name[idx]], img)
        else:
            p = Path(self.cfg.tester.save_path) / f'{name[0]}.png'
            cv2.imwrite(str(p), np.array(image, np.long))

    def run_test(self, engine: Engine, batch: List[Tensor]):
        img_hr, label_hr, name = batch
        result = self.netSeg(img_hr.float().cuda())
        result = torch.argmax(result, 1).cpu()
        if self.cfg.tester.get('savefig_mode', None) == 'grid':
            self.generate_grid(name, label_hr, result)      
        elif self.cfg.tester.get('savefig_mode', None) == 'segonly':
            self.save_figure(name, result)
    
    def run(self):
        dataloader = idist.auto_dataloader(self.dataset['test'], batch_size=self.cfg.tester.batch_size)
        tester = Engine(self.run_test)
        self.setup_handlers(tester)
        tester.logger = setup_logger('Tester')
        Path(self.cfg.tester.save_path).mkdir(exist_ok=True)
        tester.run(dataloader)
