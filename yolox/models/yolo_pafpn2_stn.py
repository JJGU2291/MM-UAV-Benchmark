#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv

Vis = False


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

        w_rgb, w_ir = torch.split(weights, x_rgb.size(1), dim=1)
        return x_rgb * w_rgb + x_ir * w_ir

class Fusion2(nn.Module):
    def __init__(self, c1):
        super().__init__()

        self.conv1 = nn.Conv2d(c1, 8, kernel_size=3, stride=1, padding=1)

        # 初始化默认仿射变换矩阵，并确保它们在正确的设备上
        self.default_ir_to_rgb_theta = nn.Parameter(torch.tensor([
            [ 1.2161, -0.0351, -0.0825],
            [ 0.0351,  1.2161,  0.2926]],
            dtype=torch.float), requires_grad=False)

        self.default_rgb_to_ir_theta = nn.Parameter(torch.tensor([
            [0.8271, 0.0119, 0.0615],
            [-0.0119, 0.8271, -0.2396]
        ], dtype=torch.float), requires_grad=False)

        self.ir_to_rgb_stn = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0),

            nn.Flatten(),
            nn.Linear(32, 6),
            nn.Tanh()
        )

        # 设置偏移量 Δθ 的初始值
        self.ir_to_rgb_stn[-2].weight.data.zero_()
        self.ir_to_rgb_stn[-2].bias.data.zero_()

        # STN for aligning RGB to IR
        self.rgb_to_ir_stn = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0),

            nn.Flatten(),
            nn.Linear(32, 6),
            nn.Tanh()
        )

        # 设置偏移量 Δθ 的初始值
        self.rgb_to_ir_stn[-2].weight.data.zero_()
        self.rgb_to_ir_stn[-2].bias.data.zero_()

        self.conv_for_rgb = nn.Conv2d(c1, c1, kernel_size=3, padding=1)
        self.conv_for_ir = nn.Conv2d(c1, c1, kernel_size=3, padding=1)


        self.fusion_for_rgb = AdaptiveWeightFusion(c1)
        self.fusion_for_ir = AdaptiveWeightFusion(c1)


    def forward(self, x_rgb, x_ir):
        # Get batch size, channels, height, width
        N, C, H, W = x_rgb.shape

        if Vis:
            save_with_sequence(x_rgb, "x_rgb", "saved_npy")
            save_with_sequence(x_ir, "x_ir", "saved_npy")

        # 捕获共同特征
        x_rgb_feature_for_offset = self.conv1(x_rgb)
        x_ir_feature_for_offset = self.conv1(x_ir)

        # 计算偏移量 Δθ
        delta_theta_ir_to_rgb = self.ir_to_rgb_stn(
            torch.cat([x_ir_feature_for_offset, x_rgb_feature_for_offset], dim=1)).view(-1, 2, 3)
        delta_theta_rgb_to_ir = self.rgb_to_ir_stn(
            torch.cat([x_rgb_feature_for_offset, x_ir_feature_for_offset], dim=1)).view(-1, 2, 3)

        # 计算最终的变换矩阵（默认矩阵 + Δθ*scale）
        theta_ir_to_rgb = self.default_ir_to_rgb_theta.to(x_rgb.device).repeat(N, 1, 1) + (delta_theta_ir_to_rgb * 0.001)
        theta_rgb_to_ir = self.default_rgb_to_ir_theta.to(x_rgb.device).repeat(N, 1, 1) + (delta_theta_rgb_to_ir * 0.001)

        # Generate sampling grids
        grid_ir_to_rgb = F.affine_grid(theta_ir_to_rgb, [N, C, H, W], align_corners=True)
        grid_rgb_to_ir = F.affine_grid(theta_rgb_to_ir, [N, C, H, W], align_corners=True)

        # Align features
        x_ir_aligned_to_rgb = F.grid_sample(x_ir, grid_ir_to_rgb, mode='bilinear', padding_mode='zeros',
                                            align_corners=True)
        x_rgb_aligned_to_ir = F.grid_sample(x_rgb, grid_rgb_to_ir, mode='bilinear', padding_mode='zeros',
                                            align_corners=True)
        if Vis:
            save_with_sequence(x_ir_aligned_to_rgb, "x_ir_aligned_to_rgb", "saved_npy")
            save_with_sequence(x_rgb_aligned_to_ir, "x_rgb_aligned_to_ir", "saved_npy")

        x_ir_aligned_to_rgb = self.conv_for_rgb(x_ir_aligned_to_rgb)
        x_rgb_aligned_to_ir = self.conv_for_ir(x_rgb_aligned_to_ir)

        if Vis:
            save_with_sequence(x_ir_aligned_to_rgb, "x_ir_aligned_to_rgb_after_conv", "saved_npy")
            save_with_sequence(x_rgb_aligned_to_ir, "x_rgb_aligned_to_ir_after_conv", "saved_npy")

        x_rgb_fused = self.fusion_for_rgb(x_rgb, x_ir_aligned_to_rgb) + x_rgb
        x_ir_fused = self.fusion_for_ir(x_ir, x_rgb_aligned_to_ir) + x_ir

        if Vis:
            save_with_sequence(x_rgb_fused, "x_rgb_fused", "saved_npy")
            save_with_sequence(x_ir_fused, "x_ir_fused", "saved_npy")

        return x_rgb_fused, x_ir_fused
class Fusion1(nn.Module):
    def __init__(self, c1):
        super().__init__()

        self.conv1 = nn.Conv2d(c1, 8, kernel_size=3, stride=1, padding=1)

        # 初始化默认仿射变换矩阵，并确保它们在正确的设备上
        self.default_ir_to_rgb_theta = nn.Parameter(torch.tensor([
            [ 1.2161, -0.0351, -0.0825],
            [ 0.0351,  1.2161,  0.2926]],
            dtype=torch.float), requires_grad=False)

        self.default_rgb_to_ir_theta = nn.Parameter(torch.tensor([
            [0.8271, 0.0119, 0.0615],
            [-0.0119, 0.8271, -0.2396]
        ], dtype=torch.float), requires_grad=False)

        self.ir_to_rgb_stn = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1), #40
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, kernel_size=7, stride=2, padding=3), #20
            nn.MaxPool2d(2, stride=2), #10
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), #5
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0), #1

            nn.Flatten(),
            nn.Linear(32, 6),
            nn.Tanh()
        )

        # 设置偏移量 Δθ 的初始值
        self.ir_to_rgb_stn[-2].weight.data.zero_()
        self.ir_to_rgb_stn[-2].bias.data.zero_()

        # STN for aligning RGB to IR
        self.rgb_to_ir_stn = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1), #40
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, kernel_size=7, stride=2, padding=3), #20
            nn.MaxPool2d(2, stride=2), #10
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), #5
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0), #1

            nn.Flatten(),
            nn.Linear(32, 6),
            nn.Tanh()
        )

        # 设置偏移量 Δθ 的初始值
        self.rgb_to_ir_stn[-2].weight.data.zero_()
        self.rgb_to_ir_stn[-2].bias.data.zero_()

        self.conv_for_rgb = nn.Conv2d(c1, c1, kernel_size=3, padding=1)
        self.conv_for_ir = nn.Conv2d(c1, c1, kernel_size=3, padding=1)


        self.fusion_for_rgb = AdaptiveWeightFusion(c1)
        self.fusion_for_ir = AdaptiveWeightFusion(c1)


    def forward(self, x_rgb, x_ir):
        # Get batch size, channels, height, width
        N, C, H, W = x_rgb.shape

        if Vis:
            save_with_sequence(x_rgb, "x_rgb", "saved_npy")
            save_with_sequence(x_ir, "x_ir", "saved_npy")
        # 捕获共同特征
        x_rgb_feature_for_offset = self.conv1(x_rgb)
        x_ir_feature_for_offset = self.conv1(x_ir)

        # 计算偏移量 Δθ
        delta_theta_ir_to_rgb = self.ir_to_rgb_stn(
            torch.cat([x_ir_feature_for_offset, x_rgb_feature_for_offset], dim=1)).view(-1, 2, 3)
        delta_theta_rgb_to_ir = self.rgb_to_ir_stn(
            torch.cat([x_rgb_feature_for_offset, x_ir_feature_for_offset], dim=1)).view(-1, 2, 3)

        # 计算最终的变换矩阵（默认矩阵 + Δθ*scale）
        # 确保默认矩阵在正确的设备上
        theta_ir_to_rgb = self.default_ir_to_rgb_theta.to(x_rgb.device).repeat(N, 1, 1) + (delta_theta_ir_to_rgb * 0.001)
        theta_rgb_to_ir = self.default_rgb_to_ir_theta.to(x_rgb.device).repeat(N, 1, 1) + (delta_theta_rgb_to_ir * 0.001)

        # print(delta_theta_ir_to_rgb/100)
        # print(delta_theta_rgb_to_ir/100)
        # print("ori:",self.default_ir_to_rgb_theta)
        # print("oir:",self.default_rgb_to_ir_theta)
        # print(theta_ir_to_rgb)
        # print(theta_rgb_to_ir)

        # theta_ir_to_rgb = self.default_ir_to_rgb_theta.to(x.device).repeat(N, 1, 1)
        # theta_rgb_to_ir = self.default_rgb_to_ir_theta.to(x.device).repeat(N, 1, 1)

        # Generate sampling grids
        grid_ir_to_rgb = F.affine_grid(theta_ir_to_rgb, [N, C, H, W], align_corners=True)
        grid_rgb_to_ir = F.affine_grid(theta_rgb_to_ir, [N, C, H, W], align_corners=True)

        # print(grid_ir_to_rgb)
        # print(grid_rgb_to_ir)

        # Align features
        x_ir_aligned_to_rgb = F.grid_sample(x_ir, grid_ir_to_rgb, mode='bilinear', padding_mode='zeros',
                                            align_corners=True)
        x_rgb_aligned_to_ir = F.grid_sample(x_rgb, grid_rgb_to_ir, mode='bilinear', padding_mode='zeros',
                                            align_corners=True)

        if Vis:
            save_with_sequence(x_ir_aligned_to_rgb, "x_ir_aligned_to_rgb", "saved_npy")
            save_with_sequence(x_rgb_aligned_to_ir, "x_rgb_aligned_to_ir", "saved_npy")

        # 对齐通道特征
        x_ir_aligned_to_rgb = self.conv_for_rgb(x_ir_aligned_to_rgb)
        x_rgb_aligned_to_ir = self.conv_for_ir(x_rgb_aligned_to_ir)

        if Vis:
            save_with_sequence(x_ir_aligned_to_rgb, "x_ir_aligned_to_rgb_after_conv", "saved_npy")
            save_with_sequence(x_rgb_aligned_to_ir, "x_rgb_aligned_to_ir_after_conv", "saved_npy")


        x_rgb_fused = self.fusion_for_rgb(x_rgb, x_ir_aligned_to_rgb) + x_rgb
        x_ir_fused = self.fusion_for_ir(x_ir, x_rgb_aligned_to_ir) + x_ir

        if Vis:
            save_with_sequence(x_rgb_fused, "x_rgb_fused", "saved_npy")
            save_with_sequence(x_ir_fused, "x_ir_fused", "saved_npy")

        return x_rgb_fused, x_ir_fused

class Fusion0(nn.Module):
    def __init__(self, c1):
        super().__init__()

        self.conv1 = nn.Conv2d(c1, 8, kernel_size=3, stride=1, padding=1)

        # 初始化默认仿射变换矩阵，并确保它们在正确的设备上
        self.default_ir_to_rgb_theta = nn.Parameter(torch.tensor([
            [ 1.2161, -0.0351, -0.0825],
            [ 0.0351,  1.2161,  0.2926]],
            dtype=torch.float), requires_grad=False)

        self.default_rgb_to_ir_theta = nn.Parameter(torch.tensor([
            [0.8271, 0.0119, 0.0615],
            [-0.0119, 0.8271, -0.2396]
        ], dtype=torch.float), requires_grad=False)

        self.ir_to_rgb_stn = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1), #20
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=7, stride=2, padding=3), #10
            nn.MaxPool2d(2, stride=2), #5
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0), #1

            nn.Flatten(),
            nn.Linear(32, 6),
            nn.Tanh()
        )

        # 设置偏移量 Δθ 的初始值
        self.ir_to_rgb_stn[-2].weight.data.zero_()
        self.ir_to_rgb_stn[-2].bias.data.zero_()

        # STN for aligning RGB to IR
        self.rgb_to_ir_stn = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1), #20
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=7, stride=2, padding=3),  #10
            nn.MaxPool2d(2, stride=2), #5
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0), #1

            nn.Flatten(),
            nn.Linear(32, 6),
            nn.Tanh()
        )

        # 设置偏移量 Δθ 的初始值
        self.rgb_to_ir_stn[-2].weight.data.zero_()
        self.rgb_to_ir_stn[-2].bias.data.zero_()

        self.conv_for_rgb = nn.Conv2d(c1, c1, kernel_size=3, padding=1)
        self.conv_for_ir = nn.Conv2d(c1, c1, kernel_size=3, padding=1)


        self.fusion_for_rgb = AdaptiveWeightFusion(c1)
        self.fusion_for_ir = AdaptiveWeightFusion(c1)


    def forward(self, x_rgb, x_ir):
        # Get batch size, channels, height, width
        N, C, H, W = x_rgb.shape

        if Vis:
            save_with_sequence(x_rgb, "x_rgb", "saved_npy")
            save_with_sequence(x_ir, "x_ir", "saved_npy")

        # 捕获共同特征
        x_rgb_feature_for_offset = self.conv1(x_rgb)
        x_ir_feature_for_offset = self.conv1(x_ir)


        # 计算偏移量 Δθ
        delta_theta_ir_to_rgb = self.ir_to_rgb_stn(
            torch.cat([x_ir_feature_for_offset, x_rgb_feature_for_offset], dim=1)).view(-1, 2, 3)
        delta_theta_rgb_to_ir = self.rgb_to_ir_stn(
            torch.cat([x_rgb_feature_for_offset, x_ir_feature_for_offset], dim=1)).view(-1, 2, 3)

        # 计算最终的变换矩阵（默认矩阵 + Δθ*scale）
        # 确保默认矩阵在正确的设备上
        theta_ir_to_rgb = self.default_ir_to_rgb_theta.to(x_rgb.device).repeat(N, 1, 1) + (delta_theta_ir_to_rgb * 0.001)
        theta_rgb_to_ir = self.default_rgb_to_ir_theta.to(x_rgb.device).repeat(N, 1, 1) + (delta_theta_rgb_to_ir * 0.001)

        # print(delta_theta_ir_to_rgb/100)
        # print(delta_theta_rgb_to_ir/100)
        # print("ori:",self.default_ir_to_rgb_theta)
        # print("oir:",self.default_rgb_to_ir_theta)
        # print(theta_ir_to_rgb)
        # print(theta_rgb_to_ir)

        # theta_ir_to_rgb = self.default_ir_to_rgb_theta.to(x.device).repeat(N, 1, 1)
        # theta_rgb_to_ir = self.default_rgb_to_ir_theta.to(x.device).repeat(N, 1, 1)

        # Generate sampling grids
        grid_ir_to_rgb = F.affine_grid(theta_ir_to_rgb, [N, C, H, W], align_corners=True)
        grid_rgb_to_ir = F.affine_grid(theta_rgb_to_ir, [N, C, H, W], align_corners=True)

        # print(grid_ir_to_rgb)
        # print(grid_rgb_to_ir)

        # Align features
        x_ir_aligned_to_rgb = F.grid_sample(x_ir, grid_ir_to_rgb, mode='bilinear', padding_mode='zeros',
                                            align_corners=True)
        x_rgb_aligned_to_ir = F.grid_sample(x_rgb, grid_rgb_to_ir, mode='bilinear', padding_mode='zeros',
                                            align_corners=True)

        if Vis:
            save_with_sequence(x_ir_aligned_to_rgb, "x_ir_aligned_to_rgb", "saved_npy")
            save_with_sequence(x_rgb_aligned_to_ir, "x_rgb_aligned_to_ir", "saved_npy")

        # 对齐通道特征
        x_ir_aligned_to_rgb = self.conv_for_rgb(x_ir_aligned_to_rgb)
        x_rgb_aligned_to_ir = self.conv_for_ir(x_rgb_aligned_to_ir)

        if Vis:
            save_with_sequence(x_ir_aligned_to_rgb, "x_ir_aligned_to_rgb_after_conv", "saved_npy")
            save_with_sequence(x_rgb_aligned_to_ir, "x_rgb_aligned_to_ir_after_conv", "saved_npy")

        x_rgb_fused = self.fusion_for_rgb(x_rgb, x_ir_aligned_to_rgb) + x_rgb
        x_ir_fused = self.fusion_for_ir(x_ir, x_rgb_aligned_to_ir) + x_ir

        if Vis:
            save_with_sequence(x_rgb_fused, "x_rgb_fused", "saved_npy")
            save_with_sequence(x_ir_fused, "x_ir_fused", "saved_npy")

        return x_rgb_fused, x_ir_fused

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
