# Deep Back-Projection Networks For Super-Resolution
# https://openaccess.thecvf.com/content_cvpr_2018/papers/Haris_Deep_Back-Projection_Networks_CVPR_2018_paper.pdf
# https://github.com/soapisnotfat/super-resolution

import torch
import torch.nn as nn

class DBPN(nn.Module):
    def __init__(self, num_channels, base_channels, feat_channels, num_stages, scale_factor):
        super(DBPN, self).__init__()

        if scale_factor == 2:
            kernel_size = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel_size = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel_size = 12
            stride = 8
            padding = 2
        else:
            kernel_size = None
            stride = None
            padding = None
            Warning("please choose the scale factor from 2, 4, 8")
            exit()

        # Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat_channels, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(feat_channels, base_channels, 1, 1, 0, activation='prelu', norm=None)
        # Back-projection stages
        self.up1 = UpBlock(base_channels, kernel_size, stride, padding)
        self.down1 = DownBlock(base_channels, kernel_size, stride, padding)
        self.up2 = UpBlock(base_channels, kernel_size, stride, padding)
        self.down2 = D_DownBlock(base_channels, kernel_size, stride, padding, 2)
        self.up3 = D_UpBlock(base_channels, kernel_size, stride, padding, 2)
        self.down3 = D_DownBlock(base_channels, kernel_size, stride, padding, 3)
        self.up4 = D_UpBlock(base_channels, kernel_size, stride, padding, 3)
        self.down4 = D_DownBlock(base_channels, kernel_size, stride, padding, 4)
        self.up5 = D_UpBlock(base_channels, kernel_size, stride, padding, 4)
        self.down5 = D_DownBlock(base_channels, kernel_size, stride, padding, 5)
        self.up6 = D_UpBlock(base_channels, kernel_size, stride, padding, 5)
        self.down6 = D_DownBlock(base_channels, kernel_size, stride, padding, 6)
        self.up7 = D_UpBlock(base_channels, kernel_size, stride, padding, 6)
        # Reconstruction
        self.output_conv = ConvBlock(num_stages * base_channels, num_channels, 3, 1, 1, activation=None, norm=None)

    def weight_init(self):
        for m in self._modules:
            class_name = m.__class__.__name__
            if class_name.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif class_name.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.feat0(x)
        x = self.feat1(x)

        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)

        concat_h = torch.cat((h2, h1), 1)
        l = self.down2(concat_h)

        concat_l = torch.cat((l, l1), 1)
        h = self.up3(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down3(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up4(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down4(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up5(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down5(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up6(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down6(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up7(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        x = self.output_conv(concat_h)

        return x


class DBPNS(nn.Module):
    def __init__(self, num_channels, base_channels, feat_channels, num_stages, scale_factor):
        super(DBPNS, self).__init__()

        if scale_factor == 2:
            kernel_size = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel_size = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel_size = 12
            stride = 8
            padding = 2
        else:
            kernel_size = None
            stride = None
            padding = None
            Warning("please choose the scale factor from 2, 4, 8")
            exit()

        # Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat_channels, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(feat_channels, base_channels, 1, 1, 0, activation='prelu', norm=None)
        # Back-projection stages
        self.up1 = UpBlock(base_channels, kernel_size, stride, padding)
        self.down1 = DownBlock(base_channels, kernel_size, stride, padding)
        self.up2 = UpBlock(base_channels, kernel_size, stride, padding)
        # Reconstruction
        self.output_conv = ConvBlock(num_stages * base_channels, num_channels, 3, 1, 1, activation=None, norm=None)

    def weight_init(self):
        for m in self._modules:
            class_name = m.__class__.__name__
            if class_name.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif class_name.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.feat0(x)
        x = self.feat1(x)

        h1 = self.up1(x)
        h2 = self.up2(self.down1(h1))

        x = self.output_conv(torch.cat((h2, h1), 1))

        return x


class DBPNLL(nn.Module):
    def __init__(self, num_channels, base_channels, feat_channels, num_stages, scale_factor):
        super(DBPNLL, self).__init__()

        if scale_factor == 2:
            kernel_size = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel_size = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel_size = 12
            stride = 8
            padding = 2
        else:
            kernel_size = None
            stride = None
            padding = None
            Warning("please choose the scale factor from 2, 4, 8")
            exit()

        # Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat_channels, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(feat_channels, base_channels, 1, 1, 0, activation='prelu', norm=None)
        # Back-projection stages
        self.up1 = UpBlock(base_channels, kernel_size, stride, padding)
        self.down1 = DownBlock(base_channels, kernel_size, stride, padding)
        self.up2 = UpBlock(base_channels, kernel_size, stride, padding)
        self.down2 = D_DownBlock(base_channels, kernel_size, stride, padding, 2)
        self.up3 = D_UpBlock(base_channels, kernel_size, stride, padding, 2)
        self.down3 = D_DownBlock(base_channels, kernel_size, stride, padding, 3)
        self.up4 = D_UpBlock(base_channels, kernel_size, stride, padding, 3)
        self.down4 = D_DownBlock(base_channels, kernel_size, stride, padding, 4)
        self.up5 = D_UpBlock(base_channels, kernel_size, stride, padding, 4)
        self.down5 = D_DownBlock(base_channels, kernel_size, stride, padding, 5)
        self.up6 = D_UpBlock(base_channels, kernel_size, stride, padding, 5)
        self.down6 = D_DownBlock(base_channels, kernel_size, stride, padding, 6)
        self.up7 = D_UpBlock(base_channels, kernel_size, stride, padding, 6)
        self.down7 = D_DownBlock(base_channels, kernel_size, stride, padding, 7)
        self.up8 = D_UpBlock(base_channels, kernel_size, stride, padding, 7)
        self.down8 = D_DownBlock(base_channels, kernel_size, stride, padding, 8)
        self.up9 = D_UpBlock(base_channels, kernel_size, stride, padding, 8)
        self.down9 = D_DownBlock(base_channels, kernel_size, stride, padding, 9)
        self.up10 = D_UpBlock(base_channels, kernel_size, stride, padding, 9)
        # Reconstruction
        self.output_conv = ConvBlock(num_stages * base_channels, num_channels, 3, 1, 1, activation=None, norm=None)

    def weight_init(self):
        for m in self._modules:
            class_name = m.__class__.__name__
            if class_name.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif class_name.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.feat0(x)
        x = self.feat1(x)

        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)

        concat_h = torch.cat((h2, h1), 1)
        l = self.down2(concat_h)

        concat_l = torch.cat((l, l1), 1)
        h = self.up3(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down3(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up4(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down4(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up5(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down5(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up6(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down6(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up7(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down7(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up8(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down8(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up9(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down9(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up10(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        x = self.output_conv(concat_h)

        return x


############################################################################################
# Base models
############################################################################################

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu',
                 norm=None):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation=activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu',
                 norm=None):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation=activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0