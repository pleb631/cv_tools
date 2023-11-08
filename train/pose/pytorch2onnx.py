# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings
import onnxsim
import numpy as np
import torch
import mmcv
from mmcv.runner import load_checkpoint
from models import build_detector
try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')

try:
    from mmcv.onnx.symbolic import register_extra_symbolics
except ModuleNotFoundError:
    raise NotImplementedError('please update mmcv to version>=1.0.4')



def init_pose_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a pose model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_detector(config.model)
    if checkpoint is not None:
        # load model checkpoint
        load_checkpoint(model, checkpoint, map_location='cpu')
    # save the config in the model for convenience
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def _convert_batchnorm(module):
    """Convert the syncBNs into normal BN3ds."""
    module_output = module
    if isinstance(module, torch.nn.SyncBatchNorm):
        module_output = torch.nn.BatchNorm3d(module.num_features, module.eps,
                                             module.momentum, module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
            # keep requires_grad unchanged
            module_output.weight.requires_grad = module.weight.requires_grad
            module_output.bias.requires_grad = module.bias.requires_grad
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, _convert_batchnorm(child))
    del module
    return module_output


def pytorch2onnx(model,
                 input_shape,
                 opset_version=9,
                 show=False,
                 output_file='output.onnx',
                 verify=False):
    """Convert pytorch model to onnx model.

    Args:
        model (:obj:`nn.Module`): The pytorch model to be exported.
        input_shape (tuple[int]): The input tensor shape of the model.
        opset_version (int): Opset version of onnx used. Default: 11.
        show (bool): Determines whether to print the onnx model architecture.
            Default: False.
        output_file (str): Output onnx model name. Default: 'tmp.onnx'.
        verify (bool): Determines whether to verify the onnx model.
            Default: False.
    """
    model.cpu().eval()

    one_img = torch.randn(input_shape)

    register_extra_symbolics(opset_version)

    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model,
                      one_img,
                      output_file,
                      input_names=input_names,
                      output_names=output_names,
                      verbose=False,
                      opset_version=9)

    # Checks
    onnx_model = onnx.load(output_file)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    print('ONNX export success, saved as %s\n' % output_file)
    
    # Similify onnx
    print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
    onnx_model, check = onnxsim.simplify(
        onnx_model,
        dynamic_input_shape=False,
        input_shapes={'input': list(one_img.shape)})
    assert check, 'assert check failed'
    onnx.save(onnx_model, output_file)

    print(f'Successfully exported ONNX model: {output_file}')
    if verify:
        # check by onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # check the numerical value
        # get pytorch output
        pytorch_results = model(one_img)
        if not isinstance(pytorch_results, (list, tuple)):
            assert isinstance(pytorch_results, torch.Tensor)
            pytorch_results = [pytorch_results]

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]
        net_feed_input = list(set(input_all) - set(input_initializer))
        assert len(net_feed_input) == 1
        sess = rt.InferenceSession(output_file)
        onnx_results = sess.run(None,
                                {net_feed_input[0]: one_img.detach().numpy()})

        # compare results
        assert len(pytorch_results) == len(onnx_results)
        for pt_result, onnx_result in zip(pytorch_results, onnx_results):
            assert np.allclose(
                pt_result.detach().cpu(), onnx_result, atol=1.e-5
            ), 'The outputs are different between Pytorch and ONNX'
        print('The numerical values are same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMPose models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='out.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 3, 128, 128],
        help='input size')
    args = parser.parse_args()
    return args

from torch import nn
class ModelExport(nn.Module):
    def __init__(self,origin_model: nn.Module):
        super(ModelExport, self).__init__()
        self._origin_model = origin_model
       # self._origin_model.backbone.switch_to_deploy()
        self.head = nn.Linear(10, 1)
        nn.init.constant_(self.head.bias, 0)
        nn.init.constant_(self.head.weight, 0)
        
        

    def forward(self, x):
        landm = self._origin_model(x).reshape(-1,10)
        conf = self.head(landm)
        out = torch.cat([landm, conf], 1).reshape(-1, 1, 1, 11)
        return out

if __name__ == '__main__':
    args = parse_args()
    import os
    args.output_file = os.path.splitext(args.checkpoint)[0]+'.onnx'

    #assert args.opset_version == 9, 'MMPose only supports opset 11 now'

    # Following strings of text style are from colorama package
    bright_style, reset_style = '\x1b[1m', '\x1b[0m'
    red_text, blue_text = '\x1b[31m', '\x1b[34m'
    white_background = '\x1b[107m'

    msg = white_background + bright_style + red_text
    msg += 'DeprecationWarning: This tool will be deprecated in future. '
    msg += blue_text + 'Welcome to use the unified model deployment toolbox '
    msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
    msg += reset_style
    warnings.warn(msg)

    model = init_pose_model(args.config, args.checkpoint, device='cpu')
    #model.backbone.switch_to_deploy()
    model = _convert_batchnorm(model)


    # onnx.export does not support kwargs
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'Please implement the forward method for exporting.')
    exportmodel = ModelExport(model)
    # convert model to onnx file
    pytorch2onnx(
        model,
        args.shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify)
