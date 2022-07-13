import torch
import torch.nn as nn
import logging
import sys
from torch.nn import functional as F
from thop import profile
from torchsummary import summary
from ptflops import get_model_complexity_info


class BA_Block(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(BA_Block, self).__init__()
        midplanes = int(outplanes//2)

        self.pool_1_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_1_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_1_h = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv_1_w = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)

        self.pool_3_h = nn.AdaptiveAvgPool2d((None, 3))
        self.pool_3_w = nn.AdaptiveAvgPool2d((3, None))
        self.conv_3 = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)

        self.pool_5_h = nn.AdaptiveAvgPool2d((None, 5))
        self.pool_5_w = nn.AdaptiveAvgPool2d((5, None))
        self.conv_5 = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)

        self.pool_7_h = nn.AdaptiveAvgPool2d((None, 7))
        self.pool_7_w = nn.AdaptiveAvgPool2d((7, None))
        self.conv_7 = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)

        self.fuse_conv = nn.Conv2d(midplanes * 4, midplanes, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.conv_final = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)

        self.mask_conv_1 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)
        self.mask_relu = nn.ReLU(inplace=False)
        self.mask_conv_2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)


    def forward(self, x):
        _, _, h, w = x.size()
        x_1_h = self.pool_1_h(x)
        x_1_h = self.conv_1_h(x_1_h)
        x_1_h = x_1_h.expand(-1, -1, h, w)
        #x1 = F.interpolate(x1, (h, w))

        x_1_w = self.pool_1_w(x)
        x_1_w = self.conv_1_w(x_1_w)
        x_1_w = x_1_w.expand(-1, -1, h, w)
        #x2 = F.interpolate(x2, (h, w))

        x_3_h = self.pool_3_h(x)
        x_3_h = self.conv_3(x_3_h)
        x_3_h = F.interpolate(x_3_h, (h, w))

        x_3_w = self.pool_3_w(x)
        x_3_w = self.conv_3(x_3_w)
        x_3_w = F.interpolate(x_3_w, (h, w))

        x_5_h = self.pool_5_h(x)
        x_5_h = self.conv_5(x_5_h)
        x_5_h = F.interpolate(x_5_h, (h, w))

        x_5_w = self.pool_5_w(x)
        x_5_w = self.conv_5(x_5_w)
        x_5_w = F.interpolate(x_5_w, (h, w))

        x_7_h = self.pool_7_h(x)
        x_7_h = self.conv_7(x_7_h)
        x_7_h = F.interpolate(x_7_h, (h, w))

        x_7_w = self.pool_7_w(x)
        x_7_w = self.conv_7(x_7_w)
        x_7_w = F.interpolate(x_7_w, (h, w))

        hx = self.relu(self.fuse_conv(torch.cat((x_1_h + x_1_w, x_3_h + x_3_w, x_5_h + x_5_w, x_7_h + x_7_w),dim=1)))
        multi_scale_out = hx
        mask_1 = self.conv_final(hx).sigmoid()
        out1 = x * mask_1

        hx = self.mask_relu(self.mask_conv_1(out1))
        mask_2 = self.mask_conv_2(hx).sigmoid()
        hx = out1 * mask_2

        return hx, multi_scale_out

class BAM(nn.Module):
    def __init__(self, input_channel, channel_number, shortcut=False):
        super(BAM, self).__init__()
        self.shortcut = shortcut
        dilated_channel = channel_number // 2


        # Shortcut for residual
        if input_channel != channel_number:
            self.shortcut = True
            self.shortcut_conv = nn.Conv2d(input_channel, channel_number, 1, 1, bias=False)
        # Input conv
        self.input_conv = nn.Conv2d(channel_number, channel_number, kernel_size=3, padding=1)
        self.input_relu = nn.LeakyReLU(0.2, True)

        # PDC_1
        self.content_conv_d1 = nn.Conv2d(channel_number, dilated_channel, kernel_size=3, padding=1, dilation=1)
        self.content_conv_d3 = nn.Conv2d(channel_number, dilated_channel, kernel_size=3, padding=3, dilation=3)
        self.content_conv_d5 = nn.Conv2d(channel_number, dilated_channel, kernel_size=3, padding=5, dilation=5)
        # PDC_2
        self.content_conv_d1_2 = nn.Conv2d(channel_number, dilated_channel, kernel_size=3, padding=1, dilation=1)
        self.content_conv_d3_2= nn.Conv2d(channel_number, dilated_channel, kernel_size=3, padding=3, dilation=3)
        self.content_conv_d5_2 = nn.Conv2d(channel_number, dilated_channel, kernel_size=3, padding=5, dilation=5)
        # Bridge of CPDC
        self.fuse_conv_1 = nn.Conv2d(3*dilated_channel, channel_number, kernel_size=3, padding=1)
        self.fuse_relu = nn.LeakyReLU(0.2, True)
        # Fuse BA and CPDC
        self.final_conv = nn.Conv2d(5*dilated_channel, channel_number, kernel_size=3, padding=1)
        self.fina_relu = nn.LeakyReLU(0.2, True)

        # Blur-aware Attention
        self.BA = BA_Block(channel_number, channel_number)

    def forward(self, x):

        if self.shortcut:
            x = self.shortcut_conv(x)

        in_feature = self.input_relu((self.input_conv(x)))
        # BA
        BA_out, attn = self.BA(in_feature)

        # CPDC
        content_d1 = (self.content_conv_d1(in_feature))
        content_d3 = (self.content_conv_d3(in_feature))
        content_d5 = (self.content_conv_d5(in_feature))
        dilation_fusion_1 = self.fuse_relu(self.fuse_conv_1(torch.cat((content_d1, content_d3, content_d5), dim=1)))
        content_d1_2 = (self.content_conv_d1_2(dilation_fusion_1))
        content_d3_2 = (self.content_conv_d3_2(dilation_fusion_1))
        content_d5_2 = (self.content_conv_d5_2(dilation_fusion_1))
        CPDC_out = torch.cat((content_d1_2, content_d3_2, content_d5_2), dim=1)

        # Concatenate and Fuse
        BAM_out = self.final_conv(torch.cat((CPDC_out, BA_out), dim=1))
        BAM_out = BAM_out + x
        BAM_out = self.fina_relu(BAM_out)

        return BAM_out, attn

class BANet_model(nn.Module):
    def __init__(self):
        super(BANet_model, self).__init__()

        dim_1 = 64
        dim_2 = 128

        self.en_layer1 = nn.Sequential(
            nn.Conv2d(3, dim_1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        self.en_layer2 = nn.Sequential(
            nn.Conv2d(dim_1, dim_2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.BAM_1 = BAM(dim_2, dim_2)
        self.BAM_2 = BAM(dim_2, dim_2)
        self.BAM_3 = BAM(dim_2, dim_2)
        self.BAM_4 = BAM(dim_2, dim_2)
        self.BAM_5 = BAM(dim_2, dim_2)
        self.BAM_6 = BAM(dim_2, dim_2)
        self.BAM_7 = BAM(dim_2, dim_2)
        self.BAM_8 = BAM(dim_2, dim_2)
        self.BAM_9 = BAM(dim_2, dim_2)
        self.BAM_10 = BAM(dim_2, dim_2)

        self.de_layer1 = nn.Sequential(
            nn.ConvTranspose2d(dim_2, dim_1, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        self.de_layer2 = nn.Sequential(
            nn.Conv2d(dim_1 + dim_1, dim_1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        self.de_layer3 = nn.Sequential(
            nn.Conv2d(dim_1, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):

        hx = self.en_layer1(x)

        residual = hx
        hx = self.en_layer2(hx)

        hx, attn_1 = self.BAM_1(hx)
        hx, attn_2 = self.BAM_2(hx)
        hx, attn_3 = self.BAM_3(hx)
        hx, attn_4 = self.BAM_4(hx)
        hx, attn_5 = self.BAM_5(hx)
        hx, attn_6 = self.BAM_6(hx)
        hx, attn_7 = self.BAM_7(hx)
        hx, attn_8 = self.BAM_8(hx)
        hx, attn_9 = self.BAM_9(hx)
        hx, attn_10 = self.BAM_10(hx)

        hx = self.de_layer1(hx)
        hx = self.de_layer2(torch.cat((hx, residual), dim=1))
        hx = self.de_layer3(hx)

        return hx + x
