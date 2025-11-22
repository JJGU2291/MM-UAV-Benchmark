#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

import torch
import torch.distributed as dist
import torch.nn as nn

import os
import random
from yolox.exp import Exp2 as MyExp2


class Exp(MyExp2):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "/mnt/sda/Disk_D/MMMUAV/"

        self.train_ann1 = "train-rgb.json"
        self.val_ann1 = "val-rgb.json"

        self.train_ann2 = "train-ir.json"
        self.val_ann2 = "val-ir.json"

        self.num_classes = 1

        self.max_epoch = 50
        self.data_num_workers = 4
        self.eval_interval = 1

        self.no_aug_epochs = 50
        self.enable_mixup = False
        self.random_size=None

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            # 使用 AdamW 优化器
            optimizer = torch.optim.AdamW(
                [
                    {"params": pg0, "weight_decay": 0.0},  # 不对 BatchNorm 和 bias 应用权重衰减
                    {"params": pg1, "weight_decay": self.weight_decay},  # 应用权重衰减
                    {"params": pg2, "weight_decay": 0.0},  # 不对 bias 应用权重衰减
                ],
                lr=lr,
                betas=(0.9, 0.999),  # AdamW 的默认 beta 值
            )
            self.optimizer = optimizer

        return self.optimizer

    def get_model(self):
        from yolox.models import YOLOX2, YOLOXHead
        from yolox.models.yolo_pafpn2_def import YOLOPAFPN2

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN2(self.depth, self.width, in_channels=in_channels)

            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels)
            head2 = YOLOXHead(self.num_classes, self.width, in_channels=in_channels)

            self.model = YOLOX2(backbone, head, head2)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        # 加载权重
        weight_path = "YOLOX_outputs/yolox_s_2_stream/best_ckpt.pth.tar"

        if not os.path.exists(weight_path):
            raise RuntimeError("No pretrained 1 stage weights found at {}".format(weight_path))

        checkpoint = torch.load(weight_path)

        # 获取预训练模型
        pretrained_state_dict = checkpoint['model']  # YOLO官方格式
        # print(pretrained_state_dict)

        # 获取目标模型的state_dict
        model_state_dict = self.model.state_dict()

        # 筛选可用的预训练权重并记录匹配的层
        matched_keys = []
        for k, v in pretrained_state_dict.items():
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                model_state_dict[k] = v  # 更新权重
                matched_keys.append(k)  # 记录匹配的键

        # 加载筛选后的权重到目标模型
        self.model.load_state_dict(model_state_dict, strict=False)


        # 冻结除 Fusion 模块外的其他模块的权重
        for name, param in self.model.named_parameters():
            if 'Fusion0' not in name and 'Fusion1' not in name and 'Fusion2' not in name:
                if 'head' not in name:
                    param.requires_grad = False

        print("Will not train these layers:")
        for k, v in self.model.named_parameters():
            if not v.requires_grad:
                print(f"❄️ : {k} (参数数量: {v.numel()})")

        print("\nWill train these layers:")
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                print(f"🔥  {k} (参数数量: {v.numel()})")


        return self.model



