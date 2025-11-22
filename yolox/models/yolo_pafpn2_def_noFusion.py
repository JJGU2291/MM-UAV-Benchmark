#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv

###########
def save_with_sequence(tensor, base_filename, directory="."):
    """
    将张量保存为.npy文件，如果文件已存在，则在文件名后添加序号。

    参数:
        tensor (torch.Tensor): 要保存的张量。
        base_filename (str): 基础文件名（不包含扩展名）。
        directory (str): 保存文件的目录，默认为当前目录。

    返回:
        str: 最终保存的文件路径。
    """
    # 确保目录存在
    os.makedirs(directory, exist_ok=True)

    # 构造完整的基础文件路径
    base_path = os.path.join(directory, base_filename)

    # 初始化序号
    seq = 0

    # 如果张量在GPU上，将其移动到CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # 将PyTorch张量转换为NumPy数组
    numpy_array = tensor.numpy()

    # 循环检查文件是否存在
    while True:
        # 构造带序号的文件名
        if seq == 0:
            file_path = f"{base_path}.npy"
        else:
            file_path = f"{base_path}_{seq}.npy"

        # 如果文件不存在，保存并退出循环
        if not os.path.exists(file_path):
            np.save(file_path, numpy_array)
            print(f"文件已保存为: {file_path}")
            return file_path

        # 文件存在，增加序号
        seq += 1
# 示例：保存一个随机张量
# tensor = np.random.rand(3, 4)
# save_with_sequence(tensor, "my_tensor", "saved_data")
###########


class ASPPModule(nn.Module):
    """空洞空间金字塔池化"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=3, dilation=3)
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6)
        self.conv4 = nn.Conv2d(in_ch, out_ch, 3, padding=9, dilation=9)
        self.fuse = nn.Conv2d(4*out_ch, out_ch, 1)

    def forward(self, x):
        return self.fuse(torch.cat([
            self.conv1(x),
            self.conv2(x),
            self.conv3(x),
            self.conv4(x)
        ], dim=1))

class Offset_Module(nn.Module):
    def __init__(self,c1, k=1):
        super().__init__()

        self.conv1 = nn.Conv2d(c1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.leakyReLU = nn.LeakyReLU(0.2)

        self.aspp1 = ASPPModule(8, 8)

        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.leakyReLU2 = nn.LeakyReLU(0.2)

        # 输出 offset 和缩放因子
        # self.conv_offset = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        # self.conv_offset1 = nn.Conv2d(4, 2 * k * k, kernel_size=1)

        self.conv_offset = nn.Conv2d(8, 2 * k * k, kernel_size=1)

        # self.conv2 = nn.Conv2d(c1, 8, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(8)
        # self.leakyReLU2 = nn.LeakyReLU(0.2)
        #
        # self.conv_scale = nn.Conv2d(8, 2 * k * k, kernel_size=1)  # 每一个卷积位置预测出一个缩放因子


    def forward(self, feature):
        # save_with_sequence(x, "feature_for_offset", "saved_npy")
        ##with BN
        x = self.leakyReLU(self.bn1(self.conv1(feature)))
        x = self.aspp1(x)
        x = self.leakyReLU2(self.bn2(self.conv2(x)))
        # offset = self.conv_offset1(self.conv_offset(x))
        offset = self.conv_offset(x)

        ##no BN
        # x = self.leakyReLU(self.conv1(feature))
        # x = self.aspp1(x)
        # x = self.leakyReLU2(self.conv2(x))
        # offset = self.conv_offset(x)

        # x = self.leakyReLU2(self.bn2(self.conv2(feature)))


        # save_with_sequence(offset, "offset","saved_npy")

        # scale = torch.sigmoid(self.conv_scale(x)) * 5  # 缩放因子范围 [0, 10]

        # save_with_sequence(scale, "scale","saved_npy")

        # offset = offset * 5  # 动态调整范围

        # save_with_sequence(offset[:,:1,:,:], "offset_x","saved_npy")
        # save_with_sequence(offset[:,1:,:,:],"offset_y","saved_npy")

        return offset

class Align_DefConv(nn.Module): ##基本的offset模块

    def __init__(self, c1, k=1):
        super().__init__()

        self.offset_conv = Offset_Module(16, k=k)

        self.weight = torch.ones(c1, 1, k, k)


    def forward(self, x, feature_for_offset):

        offset = self.offset_conv(feature_for_offset)

        if self.weight.device != x.device or self.weight.dtype != x.dtype:
            weight = self.weight.to(device=x.device, dtype=x.dtype)
        else:
            weight = self.weight

        x_output = torchvision.ops.deform_conv2d(input=x, offset=offset, weight=weight)

        return x_output #偏移后的特征

class AdaptiveWeightFusion(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(2 * channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, 2 * channels),
            nn.Sigmoid()
        )

    def forward(self, x_rgb, x_ir):
        batch_size = x_rgb.size(0)
        # 计算每个模态的全局特征
        avg_rgb = self.avg_pool(x_rgb).view(batch_size, -1)
        avg_ir = self.avg_pool(x_ir).view(batch_size, -1)
        # 拼接并生成通道权重
        combined = torch.cat([avg_rgb, avg_ir], dim=1)
        weights = self.fc(combined).view(batch_size, -1, 1, 1)

        # print("weight=",weights)
        # save_with_sequence(weights, "weights", "saved_npy")
        # 分割权重并加权融合
        w_rgb, w_ir = torch.split(weights, x_rgb.size(1), dim=1)
        return x_rgb * w_rgb + x_ir * w_ir

class Fusion(nn.Module): # ASPP 加权融合
    def __init__(self, c1):
        super().__init__()

        # self.CBAM = CBAM(c1//2)

        self.conv1 = nn.Conv2d(c1, 8, kernel_size=3, stride=1, padding=1)

        self.align_for_rgb = Align_DefConv(c1)
        self.align_for_ir = Align_DefConv(c1)

        # self.AdaptiveFusion_for_rgb = AdaptiveWeightFusion(c1)
        # self.AdaptiveFusion_for_ir = AdaptiveWeightFusion(c1)


    def forward(self, x_rgb, x_ir):

        # save_with_sequence(x_rgb, "x_rgb", "saved_npy")
        # save_with_sequence(x_ir, "x_ir", "saved_npy")


        # 捕获共同特征
        x_rgb_feature_for_offset = self.conv1(x_rgb)
        x_ir_feature_for_offset = self.conv1(x_ir)


        #得到两个模态偏移后的各自的特征
        x_rgb_offset = self.align_for_rgb(x_rgb, torch.cat([x_rgb_feature_for_offset, x_ir_feature_for_offset], dim=1))
        x_ir_offset = self.align_for_ir(x_ir, torch.cat([x_rgb_feature_for_offset, x_ir_feature_for_offset], dim=1))


        # 自适应加权融合
        x_rgb_output = x_ir_offset + x_rgb
        x_ir_output = x_rgb_offset + x_ir

        # save_with_sequence(x_rgb_output, "x_rgb_output", "saved_npy")
        # save_with_sequence(x_ir_output, "x_ir_output", "saved_npy")

        return x_rgb_output, x_ir_output



# class Fusion0(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16, act="silu"):
#         super().__init__()
#         self.in_channels = in_channels
#
#     def forward(self, x0_0, x0_1):
#
#         return x0_0, x0_1
#
# class Fusion1(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16, act="silu"):
#         super().__init__()
#         self.in_channels = in_channels
#
#     def forward(self, x1_0, x1_1):
#
#         return x1_0, x1_1
#
# class Fusion2(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16, act="silu"):
#         super().__init__()
#         self.in_channels = in_channels
#
#     def forward(self, x2_0, x2_1):
#
#         return x2_0, x2_1


class YOLOPAFPN2(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone_rgb = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.backbone_ir = CSPDarknet(depth, width, depthwise=depthwise, act=act)

        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")



        self.lateral_conv0_rgb = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4_rgb = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1_rgb = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3_rgb = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2_rgb = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3_rgb = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1_rgb = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4_rgb = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )


        ####ir
        self.lateral_conv0_ir = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4_ir = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1_ir = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3_ir = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2_ir = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3_ir = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1_ir = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4_ir = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.Fusion0 = Fusion(int(in_channels[2] * width))
        self.Fusion1 = Fusion(int(in_channels[1] * width))
        self.Fusion2 = Fusion(int(in_channels[0] * width))




    def forward(self, input1, input2):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features1 = self.backbone_rgb(input1)

        out_features2 = self.backbone_ir(input2)

        features1 = [out_features1[f] for f in self.in_features]
        features2 = [out_features2[f] for f in self.in_features]

        [x2_rgb, x1_rgb, x0_rgb] = features1

        [x2_ir, x1_ir, x0_ir] = features2

        x0_rgb, x0_ir = self.Fusion0(x0_rgb, x0_ir)
        x1_rgb, x1_ir = self.Fusion1(x1_rgb, x1_ir)
        x2_rgb, x2_ir = self.Fusion2(x2_rgb, x2_ir)


        # x0.shape torch.Size([32, 512, 20, 20])
        # x1.shape torch.Size([32, 256, 40, 40])
        # x2.shape torch.Size([32, 128, 80, 80])

        fpn_out0_rgb = self.lateral_conv0_rgb(x0_rgb)  # 1024->512/32
        f_out0_rgb = self.upsample(fpn_out0_rgb)  # 512/16
        f_out0_rgb = torch.cat([f_out0_rgb, x1_rgb], 1)  # 512->1024/16
        f_out0_rgb = self.C3_p4_rgb(f_out0_rgb)  # 1024->512/16

        fpn_out1_rgb = self.reduce_conv1_rgb(f_out0_rgb)  # 512->256/16
        f_out1_rgb = self.upsample(fpn_out1_rgb)  # 256/8
        f_out1_rgb = torch.cat([f_out1_rgb, x2_rgb], 1)  # 256->512/8
        pan_out2_rgb = self.C3_p3_rgb(f_out1_rgb)  # 512->256/8

        p_out1_rgb = self.bu_conv2_rgb(pan_out2_rgb)  # 256->256/16
        p_out1_rgb = torch.cat([p_out1_rgb, fpn_out1_rgb], 1)  # 256->512/16
        pan_out1_rgb = self.C3_n3_rgb(p_out1_rgb)  # 512->512/16

        p_out0_rgb = self.bu_conv1_rgb(pan_out1_rgb)  # 512->512/32
        p_out0_rgb = torch.cat([p_out0_rgb, fpn_out0_rgb], 1)  # 512->1024/32
        pan_out0_rgb = self.C3_n4_rgb(p_out0_rgb)  # 1024->1024/32


        ####ir

        fpn_out0_ir = self.lateral_conv0_ir(x0_ir)  # 1024->512/32
        f_out0_ir = self.upsample(fpn_out0_ir)  # 512/16
        f_out0_ir = torch.cat([f_out0_ir, x1_ir], 1)  # 512->1024/16
        f_out0_ir = self.C3_p4_ir(f_out0_ir)  # 1024->512/16

        fpn_out1_ir = self.reduce_conv1_ir(f_out0_ir)  # 512->256/16
        f_out1_ir = self.upsample(fpn_out1_ir)  # 256/8
        f_out1_ir = torch.cat([f_out1_ir, x2_ir], 1)  # 256->512/8
        pan_out2_ir = self.C3_p3_ir(f_out1_ir)  # 512->256/8

        p_out1_ir = self.bu_conv2_ir(pan_out2_ir)  # 256->256/16
        p_out1_ir = torch.cat([p_out1_ir, fpn_out1_ir], 1)  # 256->512/16
        pan_out1_ir = self.C3_n3_ir(p_out1_ir)  # 512->512/16

        p_out0_ir = self.bu_conv1_ir(pan_out1_ir)  # 512->512/32
        p_out0_ir = torch.cat([p_out0_ir, fpn_out0_ir], 1)  # 512->1024/32
        pan_out0_ir = self.C3_n4_ir(p_out0_ir)  # 1024->1024/32


        outputs1 = (pan_out2_rgb, pan_out1_rgb, pan_out0_rgb)

        outputs2 = (pan_out2_ir, pan_out1_ir, pan_out0_ir)

        return outputs1, outputs2
