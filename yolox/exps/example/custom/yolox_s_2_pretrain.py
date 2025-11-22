#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

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

        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 1