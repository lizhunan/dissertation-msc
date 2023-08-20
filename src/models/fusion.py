import torch.nn as nn
import torch.nn.functional as F

class SEFusion(nn.Module):
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
    def __init__(self, channels_in):
        super(ECAFusion, self).__init__()
        self.se_rgb = ECABlock(channels_in)
        self.se_depth = ECABlock(channels_in)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        return out

class SEBlock(nn.Module):
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
    def __init__(self, channel, reduction=16):
        super(ECABlock, self).__init__()

    def forward(self, x):
        pass
