# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import os
import numpy as np

from ..pipelines import Compose



class BaseDataset():

    def __init__(self,
                 data_root,
                 pipeline,):
        super(BaseDataset, self).__init__()
        self.pipeline = Compose(pipeline)
        self.data_root=data_root
        self.data_infos = self.load_annotations()


    def load_annotations(self,):
        pass
    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

