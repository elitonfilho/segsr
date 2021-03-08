import torch
from torchvision.transforms import ToTensor
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class LandCoverAIDataset(torch.utils.data.Dataset):

    def __init__(self, img_dir):
        super().__init__()
        self.filenames_hr = sorted([Path(x) for x in Path(img_dir).iterdir() if not '_m' in x.stem])
        self.filenames_an = list(map(lambda x: x.parent / f'{x.stem}_m{x.suffix}', self.filenames_hr))

    def __len__(self):
        return len(self.filenames_hr)

    def __getitem__(self, index):
        hr_img = Image.open(self.filenames_hr[index])
        label_img = Image.open(self.filenames_an[index])
        hr_img = ToTensor()(hr_img)
        label_img = torch.tensor(np.array(label_img, dtype=np.int32))
        return hr_img, label_img.long()


if __name__ == "__main__":
    data = LandCoverAIDataset('D:\\datasets\\landcover.ai\\output256')
    # print(*zip(map(lambda x:x.name,data.filenames_hr), map(lambda x:x.name,data.filenames_an)))
    bs = 10
    train_loader = torch.utils.data.DataLoader(data, num_workers=1, batch_size=bs, shuffle=True)
    for idx, batch in enumerate(train_loader):
        hr, an = batch
        fix, axes = plt.subplots(bs, 2)
        for i in range(bs):
            axes[i, 0].imshow(hr[i].permute(1,2,0).numpy())
            axes[i, 1].imshow(an[i].numpy())
        plt.show()
