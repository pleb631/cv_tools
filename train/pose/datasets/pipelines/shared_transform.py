# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.utils import build_from_cfg
from numpy import random
from torchvision.transforms import functional as F

from ..builder import PIPELINES

try:
    import albumentations
except ImportError:
    albumentations = None
    
    
def random_negative(value, random_negative_prob):
    """Randomly negate value based on random_negative_prob."""
    return -value if np.random.rand() < random_negative_prob else value

@PIPELINES.register_module()
class ToTensor:
    """Transform image to Tensor.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        results (dict): contain all information about training.
    """

    def __init__(self, device='cpu'):
        self.device = device

    def _to_tensor(self, x):
        return torch.from_numpy(x.astype('float32')).permute(2, 0, 1).to(
            self.device).div_(255.0)

    def __call__(self, results):
        if isinstance(results['img'], (list, tuple)):
            results['img'] = [self._to_tensor(img) for img in results['img']]
        else:
            results['img'] = self._to_tensor(results['img'])

        return results


@PIPELINES.register_module()
class NormalizeTensor:
    """Normalize the Tensor image (CxHxW), with mean and std.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        mean (list[float]): Mean values of 3 channels.
        std (list[float]): Std values of 3 channels.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        if isinstance(results['img'], (list, tuple)):
            results['img'] = [
                F.normalize(img, mean=self.mean, std=self.std, inplace=True)
                for img in results['img']
            ]
        else:
            results['img'] = F.normalize(
                results['img'], mean=self.mean, std=self.std, inplace=True)

        return results


@PIPELINES.register_module()
class Compose:
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]): Either config
          dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
            dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        """Compute the string representation."""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string


@PIPELINES.register_module()
class Collect:
    """Collect data from the loader relevant to the specific task.

    This keeps the items in `keys` as it is, and collect items in `meta_keys`
    into a meta item called `meta_name`.This is usually the last stage of the
    data loader pipeline.
    For example, when keys='imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_metas', the results will be a dict with
    keys 'imgs' and 'img_metas', where 'img_metas' is a DataContainer of
    another dict with keys 'filename', 'label', 'original_shape'.

    Args:
        keys (Sequence[str|tuple]): Required keys to be collected. If a tuple
          (key, key_new) is given as an element, the item retrieved by key will
          be renamed as key_new in collected data.
        meta_name (str): The name of the key that contains meta information.
          This key is always populated. Default: "img_metas".
        meta_keys (Sequence[str|tuple]): Keys that are collected under
          meta_name. The contents of the `meta_name` dictionary depends
          on `meta_keys`.
    """

    def __init__(self, keys, meta_keys, meta_name='img_metas'):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name

    def __call__(self, results):
        """Performs the Collect formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
              to the next transform in pipeline.
        """
        if 'ann_info' in results:
            results.update(results['ann_info'])

        data = {}
        for key in self.keys:
            if isinstance(key, tuple):
                assert len(key) == 2
                key_src, key_tgt = key[:2]
            else:
                key_src = key_tgt = key
            data[key_tgt] = results[key_src]

        meta = {}
        if len(self.meta_keys) != 0:
            for key in self.meta_keys:
                if isinstance(key, tuple):
                    assert len(key) == 2
                    key_src, key_tgt = key[:2]
                else:
                    key_src = key_tgt = key
                meta[key_tgt] = results[key_src]
        data[self.meta_name] = DC(meta, cpu_only=True)

        return data

    def __repr__(self):
        """Compute the string representation."""
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys})')


@PIPELINES.register_module()
class Albumentation:
    """Albumentation augmentation (pixel-level transforms only). Adds custom
    pixel-level transformations from Albumentations library. Please visit
    `https://albumentations.readthedocs.io` to get more information.

    Note: we only support pixel-level transforms.
    Please visit `https://github.com/albumentations-team/`
    `albumentations#pixel-level-transforms`
    to get more information about pixel-level transforms.

    An example of ``transforms`` is as followed:

    .. code-block:: python

        [
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]

    Args:
        transforms (list[dict]): A list of Albumentation transformations
        keymap (dict): Contains {'input key':'albumentation-style key'},
            e.g., {'img': 'image'}.
    """

    def __init__(self, transforms, keymap=None):
        if albumentations is None:
            raise RuntimeError('albumentations is not installed')

        self.transforms = transforms
        self.filter_lost_elements = False

        self.aug = albumentations.Compose(
            [self.albu_builder(t) for t in self.transforms])

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.

        It resembles some of :func:`build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            if not hasattr(albumentations.augmentations.transforms, obj_type):
                warnings.warn('{obj_type} is not pixel-level transformations. '
                              'Please use with caution.')
            obj_cls = getattr(albumentations, obj_type)
        else:
            raise TypeError(f'type must be a str, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper.

        Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}

        Returns:
            dict: new dict.
        """

        updated_dict = {keymap.get(k, k): v for k, v in d.items()}
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        results = self.aug(**results)
        # back to the original format
        results = self.mapper(results, self.keymap_back)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


@PIPELINES.register_module()
class MultiItemProcess:
    """Process each item and merge multi-item results to lists.

    Args:
        pipeline (dict): Dictionary to construct pipeline for a single item.
    """

    def __init__(self, pipeline):
        self.pipeline = Compose(pipeline)

    def __call__(self, results):
        results_ = {}
        for idx, result in results.items():
            single_result = self.pipeline(result)
            for k, v in single_result.items():
                if k in results_:
                    results_[k].append(v)
                else:
                    results_[k] = [v]

        return results_



@PIPELINES.register_module()
class RenameKeys:
    """Rename the keys.

    Args:
        key_pairs (Sequence[tuple]): Required keys to be renamed.
            If a tuple (key_src, key_tgt) is given as an element,
            the item retrieved by key_src will be renamed as key_tgt.
    """

    def __init__(self, key_pairs):
        self.key_pairs = key_pairs

    def __call__(self, results):
        """Rename keys."""
        for key_pair in self.key_pairs:
            assert len(key_pair) == 2
            key_src, key_tgt = key_pair
            results[key_tgt] = results.pop(key_src)
        return results


@PIPELINES.register_module()
class Resize(object):
    def __init__(self,
                 size,
                 interpolation='bilinear'):

        self.adaptive_resize = False
        if isinstance(size, int):
            assert size > 0
            size = (size, size)

        assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                     'lanczos')

        self.size = size
        self.interpolation = interpolation

    def _resize_img(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img, w_scale, h_scale = mmcv.imresize(
                    img,
                    size=self.size,
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend='cv2')
            results[key] = img
            results['joints_3d'] = results['joints_3d']*[w_scale, h_scale,1]
            results['img_shape'] = img.shape

    def __call__(self, results):
        self._resize_img(results)
        return results
