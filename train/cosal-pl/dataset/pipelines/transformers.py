# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import cv2
import torch
import warnings
import random
from ..builder import PIPELINES

try:
    import albumentations
except ImportError:
    albumentations = None


@PIPELINES.register_module()
class Resize(object):
    """Resize images.

    Args:
        size (int | tuple): Images scales for resizing (h, w).
            When size is int, the default behavior is to resize an image
            to (size, size). When size is tuple and the second value is -1,
            the image will be resized according to adaptive_side. For example,
            when size is 224, the image is resized to 224x224. When size is
            (224, -1) and adaptive_size is "short", the short side is resized
            to 224 and the other side is computed based on the short side,
            maintaining the aspect ratio.
        interpolation (str): Interpolation method. For "cv2" backend, accepted
            values are "nearest", "bilinear", "bicubic", "area", "lanczos". For
            "pillow" backend, accepted values are "nearest", "bilinear",
            "bicubic", "box", "lanczos", "hamming".
            More details can be found in `mmcv.image.geometric`.
        adaptive_side(str): Adaptive resize policy, accepted values are
            "short", "long", "height", "width". Default to "short".
        backend (str): The image resize backend type, accepted values are
            `cv2` and `pillow`. Default: `cv2`.
    """

    def __init__(
        self, size, interpolation="bilinear", adaptive_side="short", backend="cv2"
    ):
        assert isinstance(size, int) or (isinstance(size, tuple) and len(size) == 2)
        assert adaptive_side in {"short", "long", "height", "width"}

        self.adaptive_side = adaptive_side
        self.adaptive_resize = False
        if isinstance(size, int):
            assert size > 0
            size = (size, size)
        else:
            assert size[0] > 0 and (size[1] > 0 or size[1] == -1)
            if size[1] == -1:
                self.adaptive_resize = True
        if backend not in ["cv2", "pillow"]:
            raise ValueError(
                f"backend: {backend} is not supported for resize."
                'Supported backends are "cv2", "pillow"'
            )
        if backend == "cv2":
            assert interpolation in (
                "nearest",
                "bilinear",
                "bicubic",
                "area",
                "lanczos",
            )
        else:
            assert interpolation in (
                "nearest",
                "bilinear",
                "bicubic",
                "box",
                "lanczos",
                "hamming",
            )
        self.size = size
        self.interpolation = interpolation
        self.backend = backend

    def _resize_img(self, results):
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            ignore_resize = False
            if self.adaptive_resize:
                h, w = img.shape[:2]
                target_size = self.size[0]

                condition_ignore_resize = {
                    "short": min(h, w) == target_size,
                    "long": max(h, w) == target_size,
                    "height": h == target_size,
                    "width": w == target_size,
                }

                if condition_ignore_resize[self.adaptive_side]:
                    ignore_resize = True
                elif any(
                    [
                        self.adaptive_side == "short" and w < h,
                        self.adaptive_side == "long" and w > h,
                        self.adaptive_side == "width",
                    ]
                ):
                    width = target_size
                    height = int(target_size * h / w)
                else:
                    height = target_size
                    width = int(target_size * w / h)
            else:
                height, width = self.size
            if not ignore_resize:
                img = mmcv.imresize(
                    img,
                    size=(width, height),
                    interpolation=self.interpolation,
                    return_scale=False,
                    backend=self.backend,
                )
                results[key] = img
                results["img_shape"] = img.shape

                if "gt" in results:
                    gt = results["gt"]
                    gt = mmcv.imresize(
                        gt,
                        size=(width, height),
                        interpolation=self.interpolation,
                        return_scale=False,
                        backend=self.backend,
                    )
                    results["gt"] = gt

    def __call__(self, results):
        self._resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"interpolation={self.interpolation})"
        return repr_str


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f"Type {type(data)} cannot be converted to tensor."
            "Supported types are: `numpy.ndarray`, `torch.Tensor`, "
            "`Sequence`, `int` and `float`"
        )


@PIPELINES.register_module()
class ImageToTensor:
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        permute the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and permuted to (C, H, W) order.
        """
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
            if "gt" in results:
                results["gt"] = torch.from_numpy(results["gt"])
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(keys={self.keys})"


@PIPELINES.register_module()
class Normalize:
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get("img_fields", ["img"]):
            results[key] = mmcv.imnormalize(
                results[key], self.mean, self.std, self.to_rgb
            )
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        if "gt" in results:
            results["gt"] = results["gt"] / 255
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


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
            raise RuntimeError("albumentations is not installed")

        self.transforms = transforms
        self.filter_lost_elements = False

        self.aug = albumentations.Compose(
            [self.albu_builder(t) for t in self.transforms]
        )

        if not keymap:
            self.keymap_to_albu = {
                "img": "image",
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

        assert isinstance(cfg, dict) and "type" in cfg
        args = cfg.copy()

        obj_type = args.pop("type")
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError("albumentations is not installed")
            if not hasattr(albumentations.augmentations.transforms, obj_type):
                warnings.warn(
                    "{obj_type} is not pixel-level transformations. "
                    "Please use with caution."
                )
            obj_cls = getattr(albumentations, obj_type)
        else:
            raise TypeError(f"type must be a str, but got {type(obj_type)}")

        if "transforms" in args:
            args["transforms"] = [
                self.albu_builder(transform) for transform in args["transforms"]
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
        repr_str = self.__class__.__name__ + f"(transforms={self.transforms})"
        return repr_str


@PIPELINES.register_module()
class RandomFlip(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, prob=0.5, direction="vertical"):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ["horizontal", "vertical"]

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if "flip" not in results:
            flip = True if np.random.rand() < self.prob else False
            results["flip"] = flip
        if "flip_direction" not in results:
            results["flip_direction"] = self.direction
        if results["flip"]:
            # flip image
            results["img"] = mmcv.imflip(
                results["img"], direction=results["flip_direction"]
            )

            # flip segs
            if "gt" in results:
                # use copy() to make numpy stride positive
                results["gt"] = mmcv.imflip(
                    results["gt"], direction=results["flip_direction"]
                ).copy()
        return results


@PIPELINES.register_module()
class RandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=0.5, ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results["img"]
        crop_bbox = self.get_crop_bbox(img)
        # Repeat 10 times
        for _ in range(3):
            seg_temp = self.crop(results["gt"], crop_bbox)
            if (
                np.sum(seg_temp[seg_temp > 0.5]) / np.sum(results["gt"] > 0.5)
                > self.cat_max_ratio
            ):
                break
            crop_bbox = self.get_crop_bbox(results["img"])

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results["img"] = img
        results["img_shape"] = img_shape

        # crop semantic seg
        results["gt"] = self.crop(results["gt"], crop_bbox)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(crop_size={self.crop_size})"


@PIPELINES.register_module()
class PhotoMetricDistortion:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta: int = 32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta: int = 18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img: np.ndarray, alpha: int = 1, beta: int = 0) -> np.ndarray:
        """Multiple with alpha and add beat with clip.

        Args:
            img (np.ndarray): The input image.
            alpha (int): Image weights, change the contrast/saturation
                of the image. Default: 1
            beta (int): Image bias, change the brightness of the
                image. Default: 0

        Returns:
            np.ndarray: The transformed image.
        """

        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img: np.ndarray) -> np.ndarray:
        """Brightness distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after brightness change.
        """

        if random.randint(0, 1):
            return self.convert(
                img, beta=random.uniform(-self.brightness_delta, self.brightness_delta)
            )
        return img

    def contrast(self, img: np.ndarray) -> np.ndarray:
        """Contrast distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after contrast change.
        """

        if random.randint(0, 1):
            return self.convert(
                img, alpha=random.uniform(self.contrast_lower, self.contrast_upper)
            )
        return img

    def saturation(self, img: np.ndarray) -> np.ndarray:
        """Saturation distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after saturation change.
        """

        if random.randint(0, 1):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower, self.saturation_upper),
            )
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img: np.ndarray) -> np.ndarray:
        """Hue distortion.

        Args:
            img (np.ndarray): The input image.
        Returns:
            np.ndarray: Image after hue change.
        """

        if random.randint(0, 1):
            img = mmcv.bgr2hsv(img)
            img[:, :, 0] = (
                img[:, :, 0].astype(int)
                + random.randint(-self.hue_delta, self.hue_delta)
            ) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, results: dict) -> dict:
        """Transform function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results["img"]
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(0, 1)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        results["img"] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(brightness_delta={self.brightness_delta}, "
            f"contrast_range=({self.contrast_lower}, "
            f"{self.contrast_upper}), "
            f"saturation_range=({self.saturation_lower}, "
            f"{self.saturation_upper}), "
            f"hue_delta={self.hue_delta})"
        )
        return repr_str
