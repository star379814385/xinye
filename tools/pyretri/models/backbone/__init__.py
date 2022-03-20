# -*- coding: utf-8 -*-

from .backbone_impl.resnet import ResNet
from .backbone_impl.vgg import VGG
from .backbone_impl.reid_baseline import ft_net, ft_net_50, ft_net_18
from .backbone_base import BackboneBase
from senet import se_resnext50

__all__ = [
    'BackboneBase',
    'ResNet', 'VGG',
    'ft_net',
    'ft_net_50',
    'ft_net_18',
    "se_resnext50",
]
