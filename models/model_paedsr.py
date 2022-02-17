# Pyramid Attention Networks for Image Restoration
# https://arxiv.org/abs/2004.13824
# https://github.com/SHI-Labs/Pyramid-Attention-Networks

from .utils import Upsample, MeanShift, ResidualBlockBN, BasicBlock, extract_image_patches, reduce_sum, same_padding
import torch.nn as nn
import torch
import torch.nn.functional as F


class PAEDSR(nn.Module):
    def __init__(self, in_ch, n_feats, n_resblock, res_scale, kernel_size, scale, rgb_range, rgb_mean, rgb_std):
        super(PAEDSR, self).__init__()
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        self.msa = PyramidAttention(channel=256, reduction=8, res_scale=res_scale)       

        # define head module
        m_head = [nn.Conv2d(in_ch, n_feats, kernel_size)]

        # define body module
        m_body = [ResidualBlockBN(n_feats) for _ in range(n_resblock//2)]
        m_body.append(self.msa)
        for _ in range(n_resblock//2):
            m_body.append(ResidualBlockBN(n_feats))
        m_body.append(nn.Conv2d(n_feats, n_feats, 3))

        # define tail module
        m_tail = [
            Upsample(scale, n_feats),
            nn.Conv2d(
                n_feats, in_ch, kernel_size,
                padding=(kernel_size//2)
            )
        ]

        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 


class PyramidAttention(nn.Module):
    def __init__(self, level=5, res_scale=1, channel=64, reduction=2, ksize=3, stride=1, softmax_scale=10, average=True):
        super(PyramidAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.res_scale = res_scale
        self.softmax_scale = softmax_scale
        self.scale = [1-i/10 for i in range(level)]
        self.average = average
        escape_NaN = torch.FloatTensor([1e-4])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_L_base = BasicBlock(channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match = BasicBlock(channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = BasicBlock(channel, channel, 1, bn=False, act=nn.PReLU())

    def forward(self, input):
        res = input
        #theta
        match_base = self.conv_match_L_base(input)
        shape_base = list(res.size())
        input_groups = torch.split(match_base,1,dim=0)
        # patch size for matching 
        kernel = self.ksize
        # raw_w is for reconstruction
        raw_w = []
        # w is for matching
        w = []
        #build feature pyramid
        for i in range(len(self.scale)):    
            ref = input
            if self.scale[i]!=1:
                ref  = F.interpolate(input, scale_factor=self.scale[i], mode='bicubic')
            #feature transformation function f
            base = self.conv_assembly(ref)
            shape_input = base.shape
            #sampling
            raw_w_i = extract_image_patches(base, ksizes=[kernel, kernel],
                                      strides=[self.stride,self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
            raw_w_i = raw_w_i.view(shape_input[0], shape_input[1], kernel, kernel, -1)
            raw_w_i = raw_w_i.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
            raw_w_i_groups = torch.split(raw_w_i, 1, dim=0)
            raw_w.append(raw_w_i_groups)

            #feature transformation function g
            ref_i = self.conv_match(ref)
            shape_ref = ref_i.shape
            #sampling
            w_i = extract_image_patches(ref_i, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
            w_i = w_i.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
            w_i = w_i.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
            w_i_groups = torch.split(w_i, 1, dim=0)
            w.append(w_i_groups)

        y = []
        for idx, xi in enumerate(input_groups):
            #group in a filter
            wi = torch.cat([w[i][idx][0] for i in range(len(self.scale))],dim=0)  # [L, C, k, k]
            #normalize
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
                                                     axis=[1, 2, 3],
                                                     keepdim=True)),
                               self.escape_NaN)
            wi_normed = wi/ max_wi
            #matching
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)   # [1, L, H, W] L = shape_ref[2]*shape_ref[3]
            yi = yi.view(1,wi.shape[0], shape_base[2], shape_base[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax matching score
            yi = F.softmax(yi*self.softmax_scale, dim=1)
            
            if self.average == False:
                yi = (yi == yi.max(dim=1,keepdim=True)[0]).float()
            
            # deconv for patch pasting
            raw_wi = torch.cat([raw_w[i][idx][0] for i in range(len(self.scale))],dim=0)
            yi = F.conv_transpose2d(yi, raw_wi, stride=self.stride,padding=1)/4.
            y.append(yi)
      
        y = torch.cat(y, dim=0)+res*self.res_scale  # back to the mini-batch
        return y