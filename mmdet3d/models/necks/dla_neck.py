import math
import torch
import pdb

import numpy as np
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule
from torch import nn as nn

from mmdet.models.builder import NECKS


def fill_up_weights(up):
    """Simulated bilinear upsampling kernel.
    Args:
        up (nn.Module): ConvTranspose2d module.
    """
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUpsample_V2(BaseModule):
    """Iterative Deep Aggregation (IDA) Upsampling module to upsample features
    of different scales to a similar scale.
    Args:
        out_channels (int): Number of output channels for DeformConv.
        in_channels (List[int]): List of input channels of multi-scale
            feature maps.
        kernel_sizes (List[int]): List of size of the convolving
            kernel of different scales.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        use_dcn (bool, optional): If True, use DCNv2. Default: True.
    """

    def __init__(
        self,
        out_channels,
        in_channels,
        kernel_sizes,
        norm_cfg=None,
        use_dcn=True,
        init_cfg=None,
    ):
        super(IDAUpsample_V2, self).__init__(init_cfg)
        self.use_dcn = use_dcn
        self.projs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.nodes = nn.ModuleList()

        for i in range(1, len(in_channels)):
            in_channel = in_channels[i]
            up_kernel_size = int(kernel_sizes[i])
            proj = ConvModule(
                in_channel,
                out_channels,
                3,
                padding=1,
                bias=True,
                conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
                norm_cfg=norm_cfg)
            node = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                bias=True,
                conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
                norm_cfg=norm_cfg)
            up = build_conv_layer(
                dict(type='deconv'),
                out_channels,
                out_channels,
                up_kernel_size * 2,
                stride=up_kernel_size,
                padding=up_kernel_size // 2,
                output_padding=0,
                groups=out_channels,
                bias=False)

            self.projs.append(proj)
            self.ups.append(up)
            self.nodes.append(node)

    def forward(self, mlvl_features, start_level, end_level):
        """Forward function.
        Args:
            mlvl_features (list[torch.Tensor]): Features from multiple layers.
            start_level (int): Start layer for feature upsampling.
            end_level (int): End layer for feature upsampling.
        """
        for i in range(start_level, end_level - 1):
            upsample = self.ups[i - start_level]
            project = self.projs[i - start_level]
            mlvl_features[i + 1] = upsample(project(mlvl_features[i + 1]))
            node = self.nodes[i - start_level]
            mlvl_features[i + 1] = node(mlvl_features[i + 1] +
                                        mlvl_features[i])

@NECKS.register_module()
class DLAUpsample_V2(BaseModule):
    """Deep Layer Aggregation (DLA) Upsampling module for different scales
    feature extraction, upsampling and fusion, It consists of groups of
    IDAupsample modules.
    Args:
        start_level (int): The start layer.
        channels (List[int]): List of input channels of multi-scale
            feature maps.
        scales(List[int]): List of scale of different layers' feature.
        in_channels (NoneType, optional): List of input channels of
            different scales. Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        use_dcn (bool, optional): Whether to use dcn in IDAup module.
            Default: True.
    """

    def __init__(self,
                 start_level,
                 channels,
                 scales,
                 in_channels=None,
                 out_channels=None,
                 norm_cfg=None,
                 use_dcn=True,
                 init_cfg=None):
        super(DLAUpsample_V2, self).__init__(init_cfg)
        self.start_level = start_level
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        self.out_channels = out_channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self, 'ida_{}'.format(i),
                IDAUpsample_V2(channels[j], in_channels[j:],
                            scales[j:] // scales[j], norm_cfg, use_dcn))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]
        if out_channels is not None:
            for i in range(len(channels)):
                setattr(
                    self, 'proj_{}'.format(i),
                    ConvModule(
                        channels[i],
                        out_channels,
                        3,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='DCNv2') if use_dcn else None,
                        norm_cfg=norm_cfg))
    def forward(self, mlvl_features):
        """Forward function.
        Args:
            mlvl_features(list[torch.Tensor]): Features from multi-scale
                layers.
        Returns:
            tuple[torch.Tensor]: Up-sampled features of different layers.
        """
        if type(mlvl_features)==tuple:
            mlvl_features = [mlvl_features[i] for i in range(len(mlvl_features))]
        outs = [mlvl_features[-1]]
        for i in range(len(mlvl_features) - self.start_level - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(mlvl_features, len(mlvl_features) - i - 2, len(mlvl_features))
            outs.insert(0, mlvl_features[-1])
        if self.out_channels is not None:
            outs = [getattr(self, 'proj_{}'.format(i))(feat) for i,feat in enumerate(outs)]
        return outs


@NECKS.register_module()
class DLANeck_V2(BaseModule):
    """DLA Neck.
    Args:
        in_channels (list[int], optional): List of input channels
            of multi-scale feature map.
        start_level (int, optional): The scale level where upsampling
            starts. Default: 2.
        end_level (int, optional): The scale level where upsampling
            ends. Default: 5.
        norm_cfg (dict, optional): Config dict for normalization
            layer. Default: None.
        use_dcn (bool, optional): Whether to use dcn in IDAup module.
            Default: True.
    """

    def __init__(self,
                 in_channels=[16, 32, 64, 128, 256, 512],
                 start_level=2,
                 end_level=5,
                 norm_cfg=None,
                 use_dcn=True,
                 init_cfg=None):
        super(DLANeck_V2, self).__init__(init_cfg)
        self.start_level = start_level
        self.end_level = end_level
        scales = [2**i for i in range(len(in_channels[self.start_level:]))]
        self.dla_up = DLAUpsample_V2(
            start_level=self.start_level,
            channels=in_channels[self.start_level:],
            scales=scales,
            norm_cfg=norm_cfg,
            use_dcn=use_dcn)
        self.ida_up = IDAUpsample_V2(
            in_channels[self.start_level],
            in_channels[self.start_level:self.end_level],
            [2**i for i in range(self.end_level - self.start_level)], norm_cfg,
            use_dcn)

    def forward(self, x):
        mlvl_features = [x[i] for i in range(len(x))]
        mlvl_features = self.dla_up(mlvl_features)
        outs = []
        for i in range(self.end_level - self.start_level):
            outs.append(mlvl_features[i].clone())
        self.ida_up(outs, 0, len(outs))
        return [outs[-1]]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                fill_up_weights(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

@NECKS.register_module()
class DLANeck_V3(BaseModule):
    """DLA Neck.
    Args:
        in_channels (list[int], optional): List of input channels
            of multi-scale feature map.
        start_level (int, optional): The scale level where upsampling
            starts. Default: 2.
        end_level (int, optional): The scale level where upsampling
            ends. Default: 5.
        norm_cfg (dict, optional): Config dict for normalization
            layer. Default: None.
        use_dcn (bool, optional): Whether to use dcn in IDAup module.
            Default: True.
    """

    def __init__(self,
                 in_channels=[16, 32, 64, 128, 256, 512],
                 start_level=2,
                 end_level=5,
                 norm_cfg=None,
                 use_dcn=True,
                 init_cfg=None):
        super(DLANeck_V3, self).__init__(init_cfg)
        self.start_level = start_level
        self.end_level = end_level
        scales = [2**i for i in range(len(in_channels[self.start_level:]))]
        self.dla_up = DLAUpsample_V2(
            start_level=self.start_level,
            channels=in_channels[self.start_level:],
            scales=scales,
            norm_cfg=norm_cfg,
            use_dcn=use_dcn)
        self.ida_up = IDAUpsample_V2(
            in_channels[self.start_level],
            in_channels[self.start_level:self.end_level],
            [2**i for i in range(self.end_level - self.start_level)], norm_cfg,
            use_dcn)

    def forward(self, x):
        mlvl_features = [x[i] for i in range(len(x))]
        mlvl_features = self.dla_up(mlvl_features)
        outs = []
        for i in range(self.end_level - self.start_level):
            outs.append(mlvl_features[i].clone())
        self.ida_up(outs, 0, len(outs))
        return [torch.cat(outs,1)]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                fill_up_weights(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

class IDAUpsample_V1(BaseModule):
    '''
    input: features map of different layers
    output: up-sampled features
    '''
    def __init__(self, 
        in_channels_list, 
        up_factors_list, 
        out_channels,
        norm_cfg=None,
        use_dcn=True,
        init_cfg=None):
        super(IDAUpsample_V1, self).__init__(init_cfg)
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        for i in range(1, len(in_channels_list)):
            in_channels = in_channels_list[i]
            up_factors = int(up_factors_list[i])
            proj = ConvModule(
                in_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=dict(type='DCNv2') if use_dcn else None,
                norm_cfg=norm_cfg)
            node = ConvModule(
                out_channels*2,
                out_channels,
                3,
                padding=1,
                conv_cfg=dict(type='DCNv2') if use_dcn else None,
                norm_cfg=norm_cfg)
            up = build_conv_layer(
                dict(type='deconv'),
                out_channels,
                out_channels,
                up_factors * 2,
                stride=up_factors,
                padding=up_factors // 2,
                output_padding=0,
                groups=out_channels,
                bias=False)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)


        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, layers):
        assert len(self.in_channels_list) == len(layers), \
            '{} vs {} layers'.format(len(self.in_channels_list), len(layers))
        for i in range(1, len(layers)):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            node = getattr(self, 'node_' + str(i))
            layers[i] = upsample(project(layers[i]))
            layers[i] = node(torch.cat([layers[i-1], layers[i]], 1))
        return layers

@NECKS.register_module()
class DLAUpsample_V1(BaseModule):
    def __init__(self, 
                 channels, 
                 start_level,
                 scales,
                 norm_cfg=None,
                 use_dcn=True,
                 init_cfg=None):
        super(DLAUpsample_V1, self).__init__(init_cfg)
        self.start_level = start_level
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i), IDAUpsample_V1(in_channels_list=channels[j:],
                                                    up_factors_list=scales[j:] // scales[j],
                                                    out_channels=channels[j],
                                                    norm_cfg=norm_cfg,use_dcn=use_dcn))
            scales[j + 1:] = scales[j]
            channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        layers = layers[self.start_level:]
        layers = list(layers)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            layers[-i - 2:] = ida(layers[-i - 2:])
        return layers

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                w = m.weight.data
                f = math.ceil(w.size(2) / 2)
                c = (2 * f - 1 - f % 2) / (2. * f)
                for i in range(w.size(2)):
                    for j in range(w.size(3)):
                        w[0, 0, i, j] = \
                            (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
                for c in range(1, w.size(0)):
                    w[c, 0, :, :] = w[0, 0, :, :]