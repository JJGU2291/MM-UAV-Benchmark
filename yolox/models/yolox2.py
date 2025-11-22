#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn2 import YOLOPAFPN2


class YOLOX2(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None, head2=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN2()
        if head is None:
            head = YOLOXHead(1)

        if head2 is None:
            head2 = YOLOXHead(1)

        self.backbone = backbone
        self.head = head
        self.head2 = head2

    def forward(self, x1, x2, targets1=None, targets2=None):
        # fpn output content features of [dark3, dark4, dark5]

        fpn_outs1, fpn_outs2 = self.backbone(x1, x2)

        if self.training: #训练时返回损失
            assert targets1 is not None
            assert targets2 is not None

            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs1, targets1, x1
            )

            loss2, iou_loss2, conf_loss2, cls_loss2, l1_loss2, num_fg2 = self.head2(
                fpn_outs2, targets2, x2
            )

            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,

                "total_loss2": loss2,
                "iou_loss2": iou_loss2,
                "l1_loss2": l1_loss2,
                "conf_loss2": conf_loss2,
                "cls_loss2": cls_loss2,
                "num_fg2": num_fg,
            }


        else: #推理时返回结果

            outputs1 = self.head(fpn_outs1)
            outputs2 = self.head2(fpn_outs2)

            return outputs1, outputs2

        return outputs
