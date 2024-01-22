# Copyright (c) OpenMMLab. All rights reserved.
from .fpn import Identity
from .gap_neck import GlobalAveragePooling


__all__ = ['GlobalAveragePooling', 'Identity',]
