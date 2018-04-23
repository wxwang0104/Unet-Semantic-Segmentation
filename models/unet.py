import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import Model


class SingleConv(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm, dropout_rate):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        if batch_norm:
            self.conv.add_module('BN', nn.BatchNorm2d(out_channels))
        self.conv.add_module('Relu', nn.ReLU(inplace=True))
        if dropout_rate:
            self.conv.add_module('Dropout', nn.Dropout(dropout_rate))

    def forward(self, x):
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):

    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, batch_norm, dropout_rate):
        """
        Initialize a double convolution layer.
        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param batch_norm: use batch normalization after each convolution operation.
        :param dropout_rate: dropout rate after each convolution operation.
        """
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            SingleConv(in_channels, out_channels, batch_norm, dropout_rate),
            SingleConv(out_channels, out_channels, batch_norm, dropout_rate)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class FirstConv(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm, dropout_rate):
        super(FirstConv, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels, batch_norm, dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        return x


class ContractingPathConv(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm, dropout_rate):
        super(ContractingPathConv, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, batch_norm, dropout_rate)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class ExpansivePathConv(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm, dropout_rate):
        super(ExpansivePathConv, self).__init__()
        self.upscale = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, output_padding=0)
        self.conv = DoubleConv(in_channels, out_channels, batch_norm, dropout_rate)

    def forward(self, x1, x2):
        x1 = self.upscale(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffY // 2, diffY - diffY // 2,
                        diffX // 2, diffX - diffX // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class FinalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(Model):
    def __init__(self, n_channels, n_classes, first_conv_channels=32, batch_norm=False, dropout_rate=None):
        super(Model, self).__init__()
        print('unet dropout rate: ', dropout_rate)
        self.inc = FirstConv(n_channels, first_conv_channels, batch_norm, dropout_rate)
        self.conv1 = ContractingPathConv(first_conv_channels, first_conv_channels * 2, batch_norm, dropout_rate)
        self.conv2 = ContractingPathConv(first_conv_channels * 2, first_conv_channels * 4, batch_norm, dropout_rate)
        self.conv3 = ContractingPathConv(first_conv_channels * 4, first_conv_channels * 8, batch_norm, dropout_rate)
        self.conv4 = ContractingPathConv(first_conv_channels * 8, first_conv_channels * 16, batch_norm, dropout_rate)
        self.up1 = ExpansivePathConv(first_conv_channels * 16, first_conv_channels * 8, batch_norm, dropout_rate)
        self.up2 = ExpansivePathConv(first_conv_channels * 8, first_conv_channels * 4, batch_norm, dropout_rate)
        self.up3 = ExpansivePathConv(first_conv_channels * 4, first_conv_channels * 2, batch_norm, dropout_rate)
        self.up4 = ExpansivePathConv(first_conv_channels * 2, first_conv_channels, batch_norm, dropout_rate)
        self.outc = FinalConv(first_conv_channels, n_classes)

    def forward(self, x):
        h, w = x.size()[2:]
        dh = ((h - 1) // 16 + 1) * 16 - h
        dw = ((w - 1) // 16 + 1) * 16 - w
        pl = dw // 2
        pr = dw - pl
        pt = dh // 2
        pd = dh - pt
        x = F.pad(x, (pl, pr, pt, pd))
        x1 = self.inc(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = F.pad(x, (-pl, -pr, -pt, -pd))
        # x = F.upsample(x, (h + dh, w + dw), mode='bilinear')
        return x
