#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.distributed as dist

from yolox.utils import synchronize

import random


class DataPrefetcher2:
    """
    Dual-modal DataPrefetcher that returns targets for both modalities.
    """

    def __init__(self, loader_rgb, loader_ir):
        self.loader_rgb = iter(loader_rgb)
        self.loader_ir = iter(loader_ir)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            # RGB数据: (images, targets, info, ids)
            self.next_input_rgb, self.next_target_rgb, self.next_info_rgb, self.next_id_rgb = next(self.loader_rgb)
            # IR数据: (images, targets, info, ids)
            self.next_input_ir, self.next_target_ir, self.next_info_ir, self.next_id_ir = next(self.loader_ir)

        except StopIteration:
            self.next_input_rgb = None
            self.next_target_rgb = None
            self.next_info_rgb = None
            self.next_id_rgb = None
            self.next_input_ir = None
            self.next_target_ir = None
            self.next_info_ir = None
            self.next_id_ir = None
            return

        # 使用CUDA流预取数据
        with torch.cuda.stream(self.stream):
            # 转换RGB数据到GPU
            self.next_input_rgb = self.next_input_rgb.cuda(non_blocking=True)
            if isinstance(self.next_target_rgb, (list, tuple)):
                self.next_target_rgb = [t.cuda(non_blocking=True) for t in self.next_target_rgb]
            else:
                self.next_target_rgb = self.next_target_rgb.cuda(non_blocking=True)

            # 转换IR数据到GPU
            self.next_input_ir = self.next_input_ir.cuda(non_blocking=True)
            if isinstance(self.next_target_ir, (list, tuple)):
                self.next_target_ir = [t.cuda(non_blocking=True) for t in self.next_target_ir]
            else:
                self.next_target_ir = self.next_target_ir.cuda(non_blocking=True)

    def next(self):
        # 等待当前流完成
        torch.cuda.current_stream().wait_stream(self.stream)

        # 获取当前批次数据
        input_rgb = self.next_input_rgb
        input_ir = self.next_input_ir

        # 获取两个模态的目标
        target_rgb = self.next_target_rgb
        target_ir = self.next_target_ir

        info_rgb = self.next_info_rgb
        info_ir = self.next_info_ir
        ids = self.next_id_rgb  # 假设ID匹配

        # 记录流以确保正确执行顺序
        for tensor in [input_rgb, input_ir]:
            if tensor is not None:
                tensor.record_stream(torch.cuda.current_stream())

        # 处理目标张量
        def record_targets(target):
            if target is not None:
                if isinstance(target, (list, tuple)):
                    for t in target:
                        if t is not None:
                            t.record_stream(torch.cuda.current_stream())
                else:
                    target.record_stream(torch.cuda.current_stream())

        record_targets(target_rgb)
        record_targets(target_ir)

        # 预取下一批数据
        self.preload()

        # 返回双模态数据和两个目标
        return input_rgb, input_ir, target_rgb, target_ir, info_rgb, info_ir, ids

    def __len__(self):
        return min(len(self.loader_rgb), len(self.loader_ir))


def random_resize(rgb_loader, ir_loader, exp, epoch, rank, is_distributed):
    tensor = torch.LongTensor(1).cuda()
    if is_distributed:
        synchronize()

    if rank == 0:
        if epoch > exp.max_epoch - 10:
            size = exp.input_size
        else:
            size = random.randint(*exp.random_size)
            size = int(32 * size)
        tensor.fill_(size)

    if is_distributed:
        synchronize()
        dist.broadcast(tensor, 0)

    # 同时调整两个加载器的输入尺寸
    input_size_rgb = rgb_loader.change_input_dim(multiple=tensor.item(), random_range=None)
    input_size_ir = ir_loader.change_input_dim(multiple=tensor.item(), random_range=None)

    return input_size_rgb, input_size_ir
