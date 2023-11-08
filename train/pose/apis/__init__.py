# Copyright (c) OpenMMLab. All rights reserved.
from .test import multi_gpu_test, single_gpu_test
from .train import init_random_seed, train_model

__all__ = [
    'train_model',
    'multi_gpu_test',
    'single_gpu_test',

]
