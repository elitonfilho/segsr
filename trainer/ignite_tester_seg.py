from typing import List
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.utils import setup_logger
import torch
from pathlib import Path
from torch.functional import Tensor
from torch.nn import Module
import ignite.distributed as idist
from ignite.handlers import Checkpoint
from .base_tester import BaseTester
from torchvision.utils import make_grid, save_image


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

    def run_test(self, engine: Engine, batch: List[Tensor]):
        img_hr, label_hr = batch
        result = self.netSeg(img_hr.float().cuda())
        result = torch.argmax(result, 1).cpu()
        stack = torch.stack((label_hr.float(), result.float()))
        grid = make_grid(stack, nrow=2)
        p = Path(self.cfg.tester.save_path)
        save_image(grid,p / f'{engine.state.iteration}_{idist.get_rank()}.png', format='png')
        
    
    def run(self):
        dataloader = idist.auto_dataloader(self.dataset['test'], batch_size=self.cfg.tester.batch_size)
        tester = Engine(self.run_test)
        self.setup_handlers(tester)
        tester.logger = setup_logger('Tester')
        Path(self.cfg.tester.save_path).mkdir(exist_ok=True)
        print(self.cfg.tester.path_pretrained)
        tester.run(dataloader)
