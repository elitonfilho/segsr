# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
# Implementation from EDSR-PyTorch

import torch
import torch.nn as nn

class RDBConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(RDBConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=(kernel_size-1)//2, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, in_ch1, in_ch2, n_layers, kSize=3):
        super(RDB, self).__init__()
        
        self.convs = nn.Sequential(*[   
            RDBConv(in_ch1 + i*in_ch2, in_ch2) for i in range(n_layers)   
        ])
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(in_ch1 + n_layers*in_ch2, in_ch1, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDN(nn.Module):
    def __init__(self, in_ch, num_feat, num_blocks, num_conv, num_channels, kernel_size=3):
        super(RDN, self).__init__()

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(in_ch, num_feat, kernel_size, padding=(kernel_size-1)//2)
        self.SFENet2 = nn.Conv2d(num_feat, num_feat, kernel_size, padding=(kernel_size-1)//2)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for _ in range(num_blocks):
            self.RDBs.append(
                RDB(num_feat, num_channels, num_conv)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(num_blocks * num_feat, num_feat, 1, padding=0),
            nn.Conv2d(num_feat, num_feat, kernel_size, padding=(kernel_size-1)//2)
        ])

        # Up-sampling net
        self.UPNet = nn.Sequential(*[
            nn.Conv2d(num_feat, num_channels * 4, kernel_size, padding=(kernel_size-1)//2),
            nn.PixelShuffle(2),
            nn.Conv2d(num_channels, num_channels * 4, kernel_size, padding=(kernel_size-1)//2),
            nn.PixelShuffle(2),
            nn.Conv2d(num_channels, in_ch, kernel_size, padding=(kernel_size-1)//2)
        ])

    def forward(self, x):
        _x = self.SFENet1(x)
        x  = self.SFENet2(_x)

        RDBs_out = []
        for module in self.RDBs:
            x = module(x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += _x

        return self.UPNet(x)

if __name__ == '__main__':
    import torch
    import torch.nn as nn
    model = RDN(3,64,16,8,64)
    print(model(torch.ones(1,3,128,128)).shape)