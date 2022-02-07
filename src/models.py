# parts taken from torchvision.models.resnet

from typing import Type, Callable, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNext(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            layers: List[int],
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            use_pixel_shortcut=False,
            use_s1_block=False,
            num_s1_channels=None,
            block: Type[Bottleneck] = Bottleneck
    ) -> None:
        super(ResNext, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.use_pixel_shortcut = use_pixel_shortcut
        self.use_s1_block = use_s1_block
        self.num_s1_channels = num_s1_channels

        if use_pixel_shortcut:
            self.pixel_shortcut = nn.Sequential(
                conv1x1(in_channels, 128),
                nn.ReLU(inplace=True),
                conv1x1(128, 256)
            )

        if use_s1_block:
            assert num_s1_channels
            if in_channels - num_s1_channels > 0:
                # we have S2 channels
                self.entry_block = nn.Sequential(
                    conv1x1(in_channels - num_s1_channels, self.inplanes // 2),
                    norm_layer(self.inplanes // 2),
                    nn.ReLU(inplace=True)
                )
                self.s1_block = nn.Sequential(
                    nn.Conv2d(num_s1_channels, self.inplanes // 2, kernel_size=5, padding=2),
                    norm_layer(self.inplanes // 2),
                    nn.ReLU(inplace=True)
                )
            else:
                # no s2 channels
                self.s1_block = nn.Sequential(
                    nn.Conv2d(num_s1_channels, self.inplanes, kernel_size=5, padding=2),
                    norm_layer(self.inplanes),
                    nn.ReLU(inplace=True)
                )
        else:
            self.entry_block = nn.Sequential(
                conv1x1(in_channels, self.inplanes),
                norm_layer(self.inplanes),
                nn.ReLU(inplace=True)
            )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])

        in_channels_heads = 512 * block.expansion + 256 if use_pixel_shortcut else 512 * block.expansion
        self.mu_head = nn.Sequential(
            conv1x1(in_channels_heads, 512),
            nn.ReLU(inplace=True),
            conv1x1(512, out_channels)
        )
        self.log_phi_squared_head = nn.Sequential(
            conv1x1(in_channels_heads, 512),
            nn.ReLU(inplace=True),
            conv1x1(512, out_channels)
        )

        # task-dependent homoscedastic log variances (currently unused but should be kept to not break checkpoint
        # loading)
        self.log_eta_squared = nn.Parameter(torch.zeros(out_channels))

    def _make_layer(self, block: Type[Bottleneck], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.use_pixel_shortcut:
            pixel_shortcut = self.pixel_shortcut(x)

        if self.use_s1_block:
            x_s2, x_s1 = x[:, :-self.num_s1_channels], x[:, -self.num_s1_channels:]
            if hasattr(self, 'entry_block'):
                x_s2 = self.entry_block(x_s2)
            x_s1 = self.s1_block(x_s1)
            x = torch.cat([x_s2, x_s1], dim=1)
        else:
            x = self.entry_block(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.use_pixel_shortcut:
            # concatenate in channel dim
            x = torch.cat([x, pixel_shortcut], dim=1)

        mean = self.mu_head(x)
        log_variance = self.log_phi_squared_head(x)

        return mean, log_variance
