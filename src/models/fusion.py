import torch.nn as nn
import torch.nn.functional as F

class SEFusion(nn.Module):
    r"""
    fuse rgb image and depth image, these feature maps are 'added' after using SEBlock
    """
    def __init__(self, channels_in):
        super(SEFusion, self).__init__()

        self.se_rgb = SEBlock(channels_in)
        self.se_depth = SEBlock(channels_in)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        return out

class ECAFusion(nn.Module):
    r"""
    fuse rgb image and depth image, these feature maps are 'added' after using ECABlock
    """
    def __init__(self, channels_in):
        super(ECAFusion, self).__init__()
        self.eca_rgb = ECABlock(channels_in)
        self.eca_depth = ECABlock(channels_in)

    def forward(self, rgb, depth):
        rgb = self.eca_rgb(rgb)
        depth = self.eca_depth(depth)
        out = rgb + depth
        return out

class SEBlock(nn.Module):
    r""" Based on Squeeze-and-Excitation Networks
    https://arxiv.org/pdf/2201.03545.pdf
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y
    
class ECABlock(nn.Module):
    r""" Based on ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
    https://arxiv.org/pdf/1910.03151.pdf
    """
    def __init__(self, channel, k_size=3):
        super(ECABlock, self).__init__()
        self.excitation = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=3, padding=(k_size - 1) // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        y = self.excitation(y.unsqueeze(2)).squeeze(2)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)