# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry

HOOKS = Registry('hook')


class Hook:
    stages = ('before_run','before_iter', 'after_iter',
              'after_run')

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

