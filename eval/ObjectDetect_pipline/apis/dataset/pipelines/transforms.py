
import numpy as np
from mmcv.utils import build_from_cfg
from collections.abc import Sequence
import mmcv


from ..builder import PIPELINES

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
class LoadImageFromFile:
    """Loading image(s) from file.

    Required key: "image_file".

    Added key: "img".

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): Flags specifying the color type of a loaded image,
          candidates are 'color', 'grayscale' and 'unchanged'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='rgb',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _read_image(self, path):
        img_bytes = self.file_client.get(path)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        if img is None:
            raise ValueError(f'Fail to read {path}')
        if self.to_float32:
            img = img.astype(np.float32)
        return img

    @staticmethod
    def _bgr2rgb(img):
        if img.ndim == 3:
            return mmcv.bgr2rgb(img)
        elif img.ndim == 4:
            return np.concatenate([mmcv.bgr2rgb(img_) for img_ in img], axis=0)
        else:
            raise ValueError('results["img"] has invalid shape '
                             f'{img.shape}')

    def __call__(self, results):
        """Loading image(s) from file."""
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        image_file = results.get('image_file', None)

        if isinstance(image_file, (list, tuple)):
            # Load images from a list of paths
            results['img'] = [self._read_image(path) for path in image_file]
        elif image_file is not None:
            # Load single image from path
            results['img'] = self._read_image(image_file)
        else:
            if 'img' not in results:
                # If `image_file`` is not in results, check the `img` exists
                # and format the image. This for compatibility when the image
                # is manually set outside the pipeline.
                raise KeyError('Either `image_file` or `img` should exist in '
                               'results.')
            if isinstance(results['img'], (list, tuple)):
                assert isinstance(results['img'][0], np.ndarray)
            else:
                assert isinstance(results['img'], np.ndarray)
            if self.color_type == 'color' and self.channel_order == 'rgb':
                # The original results['img'] is assumed to be image(s) in BGR
                # order, so we convert the color according to the arguments.
                if isinstance(results['img'], (list, tuple)):
                    results['img'] = [
                        self._bgr2rgb(img) for img in results['img']
                    ]
                else:
                    results['img'] = self._bgr2rgb(results['img'])
            results['image_file'] = None
        results['ori_shape'] = results['img'].shape
        results['ori_img'] = results['img'].copy()

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


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

    def __init__(self,
                 size,
                 interpolation='bilinear',
                 adaptive_side='short',
                 backend='cv2'):
        assert isinstance(size, int) or (isinstance(size, tuple)
                                         and len(size) == 2)
        assert adaptive_side in {'short', 'long', 'height', 'width'}

        self.adaptive_side = adaptive_side
        self.adaptive_resize = False
        if isinstance(size, int):
            assert size > 0
            size = (size, size)
        else:
            assert size[0] > 0 and (size[1] > 0 or size[1] == -1)
            if size[1] == -1:
                self.adaptive_resize = True
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for resize.'
                             'Supported backends are "cv2", "pillow"')
        if backend == 'cv2':
            assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                     'lanczos')
        else:
            assert interpolation in ('nearest', 'bilinear', 'bicubic', 'box',
                                     'lanczos', 'hamming')
        self.size = size
        self.interpolation = interpolation
        self.backend = backend

    def _resize_img(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            ignore_resize = False
            if self.adaptive_resize:
                h, w = img.shape[:2]
                target_size = self.size[0]

                condition_ignore_resize = {
                    'short': min(h, w) == target_size,
                    'long': max(h, w) == target_size,
                    'height': h == target_size,
                    'width': w == target_size
                }

                if condition_ignore_resize[self.adaptive_side]:
                    ignore_resize = True
                elif any([
                        self.adaptive_side == 'short' and w < h,
                        self.adaptive_side == 'long' and w > h,
                        self.adaptive_side == 'width',
                ]):
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
                    backend=self.backend)
                results[key] = img
                results['img_shape'] = img.shape

    def __call__(self, results):
        self._resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str
    
    

@PIPELINES.register_module()
class Rescale():
    def __init__(self,
             size,
             interpolation='bilinear',
             backend='cv2'):
        
        assert isinstance(size, int) or (isinstance(size, tuple)
                                         and len(size) == 2)
        
        if isinstance(size, int):
            assert size > 0
            size = (size, size)
            
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for resize.'
                             'Supported backends are "cv2", "pillow"')
        if backend == 'cv2':
            assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                     'lanczos')
        else:
            assert interpolation in ('nearest', 'bilinear', 'bicubic', 'box',
                                     'lanczos', 'hamming')
        self.size = size
        self.interpolation = interpolation
        self.backend = backend
    
    
    def _resize_img(self, results):
        
        for key in results.get('img_fields', ['img']):
            img = results[key]
            
            img_h,img_w,_ = img.shape
            
            if img_w / img_h > self.size[1] / self.size[0]:
                scale = self.size[1]/img_w
                width = self.size[1]
                height = int(img_h*scale)
            else:
                scale = self.size[0]/img_h
                height = self.size[0]
                width = int(img_w*scale)
            
            img = mmcv.imresize(
                    img,
                    size=(width, height),
                    interpolation=self.interpolation,
                    return_scale=False,
                    backend=self.backend)
            results[key] = img
            results['img_shape'] = img.shape
            results['Rescale_shape'] = img.shape
    
    def __call__(self, results):
        self._resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str
    
    

@PIPELINES.register_module()
class Pad(object):
    """Pad images.

    Args:
        size (tuple[int] | None): Expected padding size (h, w). Conflicts with
                pad_to_square. Defaults to None.
        pad_to_square (bool): Pad any image to square shape. Defaults to False.
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas when padding_mode is 'constant'. Default to 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default to "constant".
    """

    def __init__(self,
                 size=None,
                 pad_to_square=False,
                 pad_val=0,
                 padding_mode='constant'):
        assert (size is None) ^ (pad_to_square is False), \
            'Only one of [size, pad_to_square] should be given, ' \
            f'but get {(size is not None) + (pad_to_square is not False)}'
        self.size = size
        self.pad_to_square = pad_to_square
        self.pad_val = pad_val
        self.padding_mode = padding_mode

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            if self.pad_to_square:
                target_size = tuple(
                    max(img.shape[0], img.shape[1]) for _ in range(2))
            else:
                target_size = self.size
            img = mmcv.impad(
                img,
                shape=target_size,
                pad_val=self.pad_val,
                padding_mode=self.padding_mode)
            results[key] = img
            results['img_shape'] = img.shape
            results['Pad_shape'] = img.shape
            
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'(pad_val={self.pad_val}, '
        repr_str += f'padding_mode={self.padding_mode})'
        return repr_str


@PIPELINES.register_module()
class Normalize(object):
    """Normalize the image.

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
        for key in results.get('img_fields', ['img']):
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={list(self.mean)}, '
        repr_str += f'std={list(self.std)}, '
        repr_str += f'to_rgb={self.to_rgb})'
        return repr_str