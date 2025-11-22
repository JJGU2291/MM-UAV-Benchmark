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

class Fusion0(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, act="silu"):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x0_0, x0_1):

        return x0_0, x0_1

class Fusion1(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, act="silu"):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x1_0, x1_1):

        return x1_0, x1_1

class Fusion2(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, act="silu"):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x2_0, x2_1):

        return x2_0, x2_1


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

        self.Fusion0 = Fusion0(int(in_channels[2] * width))
        self.Fusion1 = Fusion1(int(in_channels[1] * width))
        self.Fusion2 = Fusion2(int(in_channels[0] * width))




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
