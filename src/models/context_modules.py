import torch
import torch.nn as nn
import torch.nn.functional as F

def get_context_module(context_module_name, channels_in, channels_out,
                       input_size, activation, upsampling_mode='bilinear'):
    if context_module_name == 'ppm-1-2-4-8':
        bins = (1, 2, 4, 8)
    else:
        bins = (1, 5)
    context_module = PyramidPoolingModule(
        channels_in, channels_out,
        bins=bins,
        activation=activation,
        upsampling_mode=upsampling_mode)
    channels_after_context_module = channels_out
    return context_module, channels_after_context_module

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, out_dim, bins=(1, 2, 3, 6),
                 activation=nn.ReLU(inplace=True),
                 upsampling_mode='bilinear'):
        reduction_dim = in_dim // len(bins)
        super(PyramidPoolingModule, self).__init__()
        self.features = []
        self.upsampling_mode = upsampling_mode
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                ConvBNAct(in_dim, reduction_dim, kernel_size=1,
                          activation=activation)
            ))
        in_dim_last_conv = in_dim + reduction_dim * len(bins)
        self.features = nn.ModuleList(self.features)

        self.final_conv = ConvBNAct(in_dim_last_conv, out_dim,
                                    kernel_size=1, activation=activation)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            h, w = x_size[2:]
            y = f(x)
            if self.upsampling_mode == 'nearest':
                out.append(F.interpolate(y, (int(h), int(w)), mode='nearest'))
            elif self.upsampling_mode == 'bilinear':
                out.append(F.interpolate(y, (int(h), int(w)),
                                         mode='bilinear',
                                         align_corners=False))
            else:
                raise NotImplementedError(
                    'For the PyramidPoolingModule only nearest and bilinear '
                    'interpolation are supported. '
                    f'Got: {self.upsampling_mode}'
                )
        out = torch.cat(out, 1)
        out = self.final_conv(out)
        return out
    
class ConvBNAct(nn.Sequential):
    def __init__(self, channels_in, channels_out, kernel_size,
                 activation=nn.ReLU(inplace=True), dilation=1, stride=1):
        super(ConvBNAct, self).__init__()
        padding = kernel_size // 2 + dilation - 1
        self.add_module('conv', nn.Conv2d(channels_in, channels_out,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          bias=False,
                                          dilation=dilation,
                                          stride=stride))
        self.add_module('bn', nn.BatchNorm2d(channels_out))
        self.add_module('act', activation)