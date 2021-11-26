import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.functional import interpolate
from torch.nn.modules.activation import ReLU, Tanh
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Linear
from torch.nn.modules.module import Module
from torch.nn.modules.pooling import AvgPool2d
from torch.nn.modules.sparse import Embedding
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import spectral_norm

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)

def spectral_conv(in_ch, out_ch, kernel_size, stride=1, padding=0):
    return spectral_norm(Conv2d(in_ch,out_ch,kernel_size,stride,padding))


class SelfAttention(nn.Module):
    def __init__(self, in_ch, k=8):
        super().__init__()
        self.in_ch = in_ch
        self.k = k
        self.convf = nn.Conv2d(in_ch, in_ch//k, 1)
        self.convg = nn.Conv2d(in_ch, in_ch//k, 1)
        self.convh = nn.Conv2d(in_ch, in_ch, 1)
        self.convv = nn.Conv2d(in_ch, in_ch, 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        bs, ch, w, h = x.shape
        f = self.convf(x)
        f = f.view(bs, -1, h*w).transpose(1,2)
        g = self.convg(x)
        g = g.view(bs, -1, h*w)
        att = torch.bmm(f,g)
        att = self.softmax(att).transpose(1,2)
        _h = self.convh(x)
        _h = _h.view(bs, -1, h*w)
        att_h = torch.bmm(_h, att)
        att_h = att_h.view((bs, ch, w, h))
        att_h = self.convv(att_h)
        return x + self.gamma*att_h
        
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, n_feat, n_class):
        super().__init__()
        self.n_feat = n_feat
        self.bn = nn.BatchNorm2d(n_feat, 0.001, False)
        self.embed = nn.Embedding(n_class, n_feat*2 )
        self.embed.weight.data[:, :n_feat].fill_(1.)
        self.embed.weight.data[:, n_feat:].zero_()

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).permute(0,3,1,2).chunk(2, 1) # CHECK
        out = gamma * out + beta
        return out

class GenBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_class, interpolate=True):
        super().__init__()
        self.cbn1 = ConditionalBatchNorm2d(in_ch, n_class)
        self.interpolate = interpolate
        # self.cbn1 = BatchNorm2d(in_ch)
        self.snconv1 = spectral_conv(in_ch,out_ch,3,1,1)
        self.relu = ReLU()
        self.cbn2 = ConditionalBatchNorm2d(out_ch, n_class)
        # self.cbn2 = BatchNorm2d(out_ch)
        self.snconv2 = spectral_conv(out_ch,out_ch,3,1,1)
        self.snconv3 = spectral_conv(in_ch,out_ch,1,1)

    def forward(self, x, label):
        # Maybe copy??
        _x = x
        x = self.cbn1(x, label)
        # x = self.cbn1(x)
        x = self.relu(x)
        if self.interpolate:
            x = interpolate(x, scale_factor=2)
        x = self.snconv1(x)
        if self.interpolate:
            x = self.cbn2(x,interpolate(label.unsqueeze(1).float(), scale_factor=2).squeeze(1).int())
        # x = self.cbn2(x)
        x = self.relu(x)
        x = self.snconv2(x)

        if self.interpolate:
            _x = interpolate(_x, scale_factor=2)
        _x = self.snconv3(_x)

        return x + _x

class Generator(nn.Module):
    def __init__(self, g_dim, n_class):
        super().__init__()
        self.g_dim = g_dim
        sconv1 = spectral_conv(3,g_dim*16, 1, 1, 0)
        self.v1 = Sequential(sconv1, BatchNorm2d(g_dim*16), ReLU())
        self.block1 = GenBlock(g_dim*16, g_dim*16, n_class, interpolate=False)
        self.block2 = GenBlock(g_dim*16, g_dim*8, n_class)
        self.block3 = GenBlock(g_dim*8, g_dim*4, n_class, interpolate=False)
        self.att = SelfAttention(g_dim*4)
        self.block4 = GenBlock(g_dim*4, g_dim*2, n_class)
        self.block5 = GenBlock(g_dim*2, g_dim, n_class, interpolate=False)
        self.bn = BatchNorm2d(g_dim, momentum=1e-4)
        self.relu = ReLU()
        self.snconv = spectral_conv(g_dim, 3, 3, 1, 1)
        self.tanh = Tanh()
        # self.apply(init_weights)

    def forward(self, x, label):
        x = self.v1(x)
        x = self.block1(x,interpolate(label.unsqueeze(1).float(), scale_factor=.25).squeeze(1).int())
        x = self.block2(x,interpolate(label.unsqueeze(1).float(), scale_factor=.25).squeeze(1).int())
        x = self.block3(x,interpolate(label.unsqueeze(1).float(), scale_factor=.5).squeeze(1).int())
        x = self.att(x)
        x = self.block4(x,interpolate(label.unsqueeze(1).float(), scale_factor=.5).squeeze(1).int())
        x = self.block5(x,label)
        x = self.bn(x)
        x = self.relu(x)
        x = self.snconv(x)
        x = self.tanh(x)
        return x

class DiscOptBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.snconv1 = spectral_norm(Conv2d(in_ch,out_ch,3,1,1))
        self.relu = ReLU()
        self.snconv2 = spectral_norm(Conv2d(out_ch,out_ch,3,1,1))
        self.downsample = AvgPool2d(2)
        self.snconv3 = spectral_norm(Conv2d(in_ch,out_ch,1,1,0))

    def forward(self, x):
        _x = x
        
        x = self.snconv1(x)
        x = self.relu(x)
        x = self.snconv2(x)
        x = self.downsample(x)
        
        _x = self.downsample(_x)
        _x = self.snconv3(_x)

        return x + _x

class DiscBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.relu = ReLU()
        self.snconv1 = spectral_norm(Conv2d(in_ch,out_ch,3,1,1))
        self.snconv2 = spectral_norm(Conv2d(out_ch,out_ch,3,1,1))
        self.downsample = AvgPool2d(2)
        self.ch_mismatch = False if in_ch == out_ch else True
        self.snconv3 = spectral_norm(Conv2d(in_ch,out_ch,1,1,0))

    def forward(self, x, downsample=True):
        _x = x

        x = self.relu(x)
        x = self.snconv1(x)
        x = self.relu(x)
        x = self.snconv2(x)
        if downsample:
            x = self.downsample(x)

        if downsample or self.ch_mismatch:
            _x = self.snconv3(_x)
            if downsample:
                _x = self.downsample(_x)

        out = x + _x
        return out


class Discriminator(nn.Module):
    def __init__(self, d_dim, n_class):
        super().__init__()
        self.d_dim = d_dim
        self.opt_block1 = DiscOptBlock(3, d_dim)
        self.block1 = DiscBlock(d_dim, d_dim*2)
        self.att = SelfAttention(d_dim*2)
        self.block2 = DiscBlock(d_dim*2, d_dim*4)
        self.block3 = DiscBlock(d_dim*4, d_dim*8)
        self.block4 = DiscBlock(d_dim*8, d_dim*16)
        self.block5 = DiscBlock(d_dim*16, d_dim*16)
        self.relu = nn.ReLU(inplace=True)
        self.snlinear = spectral_norm(Linear(d_dim*16,1))
        # parametrize.register_parametrization(self.snlinear, 'weight', spectral_norm(self.snlinear))
        self.snembed = spectral_norm(Embedding(n_class,d_dim*16))
        # parametrize.register_parametrization(self.snembed, 'weight',SS spectral_norm(self.snembed))

        # self.apply(init_weights)
        # xavier_uniform_(self.snembed.weight)

    def forward(self, x, label):
        x = self.opt_block1(x)
        x = self.block1(x)
        x = self.att(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x, downsample=False)
        x = self.relu(x)
        x = torch.sum(x, dim=[2,3])
        output1 = torch.squeeze(self.snlinear(x))
        # Projection
        h_labels = self.snembed(label)
        x = x.unsqueeze(1).unsqueeze(1) #Mod
        proj = torch.mul(x, h_labels)
        # output2 = torch.sum(proj, dim=[1,2,3]) # Mod, was dim[1]
        output2 = torch.mean(proj, dim=(1,2,3))
        output = output1 + output2
        return output

if __name__ == '__main__':
    # g = Generator(8,2).to('cuda:0')
    d = Discriminator(4,2).to('cuda:0')
    import torch
    a = torch.ones(2,3,256,256, device='cuda:0')
    b = torch.ones(2,256,256, device='cuda:0', dtype=torch.int)
    # print(a.shape)
    print(d(a,b).shape)
    # print(g(a,b).shape)
