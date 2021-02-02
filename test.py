import argparse
import time
from pathlib import Path
from matplotlib import pyplot as plt

import torch
from torch.tensor import Tensor
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm

from config import cfg
from models.rrdb_arch import RRDBNet
from models.vgg_arch import VGG128
from models.model_sr import Generator
from utils.img_utils import tensor2img

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Tesr Super Resolution Models')

    parser.add_argument(
        "--cfg",
        default="config/exp1.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # model = Generator(cfg.TEST.upscale_factor).eval()
    model = RRDBNet(3, 3).eval()
    save_dir = Path(cfg.TEST.path_save).resolve()
    save_dir.mkdir(exist_ok=True)

    if torch.cuda.is_available():
        model.cuda()
        model.load_state_dict(torch.load(cfg.TEST.path_encoder))
    else:
        model.load_state_dict(torch.load(cfg.TEST.path_encoder, map_location=lambda storage, loc: storage))

    path_img = Path(cfg.TEST.path_image).resolve()
    if path_img.is_file():
        p_imgs = [path_img]
    else:
        p_imgs = [path_img / x for x in path_img.iterdir() if x.suffix in ['.png', '.jpg', '.jpeg']]

    model.eval()

    p_imgs_bar = tqdm(p_imgs)

    for p_img in p_imgs_bar:
        i = 0
        _img = Image.open(p_img)
        _img = ToTensor()(_img).unsqueeze(0)
        if torch.cuda.is_available():
            _img = _img.cuda()
        output = model(_img)
        output = tensor2img(output, to_pil=True)
        output.save(save_dir / f'{cfg.TEST.prefix_save}_{p_img.name}')
        i += 1
        p_imgs_bar.set_description(f'[{i}/{len(p_imgs_bar)}]Eval of image {p_img.stem} done.')
