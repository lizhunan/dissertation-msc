
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.fusion import SEFusion, ECAFusion
from src.models.context_modules import get_context_module
from src.models.convnext import convnext_tiny, convnext_base

class EISSegNet(nn.Module):

    def __init__(self, 
                 height=480,
                 width=640,
                 dataset='nyuv2',
                 rgb_encoder='convnext_b',
                 depth_encoder='convnext_b',
                 fusion_module='ECA',
                 pretrained=True,
                 upsampling='bilinear'):
        super(EISSegNet, self).__init__()
        r"""EISSegNet.

        Initialize the EISSegNet.

        Args:
            height: height of input image
            width: width of input image
            dataset: The dataset on which to train, 
                     choices: sunrgbd, nyuv2, 
                     default: nyuv2
            rgb_encoder: encoder for rgb image, 
                         choices: convnext_t, convnext_b, 
                         default: convnext_b :see convnext.py
            depth_encoder: encoder for depth image, 
                           choices: convnext_t, convnext_b, 
                           default: convnext_b :see convnext.py
            fusion_module: fusion block in order to fuse the features of rgb and depth images, 
                           choices: SEBlock, ECABlock :see fusion.py
            pretrained: if using pretrained convnext model, 
                        choices: True: using, False: not using, if using pretrained model, the model will be download at pytorch .cache dirtory :see convnet.py
            upsampling: 
        """
        
        # set the number of classes according to the dataset
        if dataset == 'nyuv2':
            self.num_classes = 40
        elif dataset == 'sunrgbd':
            self.num_classes = 37
        else:
            raise NotImplementedError(f'Only nyuv2 or sunrgbd are supported for rgb encoder. Got {dataset}')

        # rgb encoder
        if rgb_encoder == 'convnext_t':
            self.rgb_encoder = convnext_tiny(pretrained=pretrained)
            self.fusion_input = [96, 96, 192, 384, 768]
            self.channels_decoder = [192, 384, 768]
        elif rgb_encoder == 'convnext_b':
            self.rgb_encoder = convnext_base(pretrained=pretrained)
            self.fusion_input = [128, 128, 256, 512, 1024]
            self.channels_decoder = [256, 512, 1024]
        else:
            raise NotImplementedError(f'Only convnext_t or convnext_b are supported for rgb encoder. Got {rgb_encoder}')
        
        # depth encoder
        if depth_encoder == 'convnext_t':
            self.depth_encoder = convnext_tiny(in_chans=1, pretrained=pretrained)
            self.fusion_input = [96, 96, 192, 384, 768]
            self.channels_decoder = [192, 384, 768]
        elif depth_encoder == 'convnext_b':
            self.depth_encoder = convnext_base(in_chans=1, pretrained=pretrained)
            self.fusion_input = [128, 128, 256, 512, 1024]
            self.channels_decoder = [256, 512, 1024]
        else:
            raise NotImplementedError(f'Only convnext_t or convnext_b are supported for depth encoder. Got {depth_encoder}')
        
        # fusion module
        if fusion_module == 'SE':
            self.stem_fusion = SEFusion(self.fusion_input[0])
            self.layer1_fusion = SEFusion(self.fusion_input[1])
            self.layer2_fusion = SEFusion(self.fusion_input[2])
            self.layer3_fusion = SEFusion(self.fusion_input[3])
            self.layer4_fusion = SEFusion(self.fusion_input[4])
        elif fusion_module == 'ECA':
            self.stem_fusion = ECAFusion(self.fusion_input[0])
            self.layer1_fusion = ECAFusion(self.fusion_input[1])
            self.layer2_fusion = ECAFusion(self.fusion_input[2])
            self.layer3_fusion = ECAFusion(self.fusion_input[3])
            self.layer4_fusion = ECAFusion(self.fusion_input[4])
        else:
            raise NotImplementedError(f'Only SE or ECA are supported for fusion module. Got {fusion_module}')

        # skip connection layers
        layers_skip1 = list()
        if self.rgb_encoder.down_4_channels_out != self.channels_decoder[2]:
            layers_skip1.append(SkipConnectBlock(
                self.rgb_encoder.down_4_channels_out,
                self.channels_decoder[2],
                kernel_size=1))
        self.skip_layer1 = nn.Sequential(*layers_skip1)

        layers_skip2 = list()
        if self.rgb_encoder.down_8_channels_out != self.channels_decoder[1]:
            layers_skip2.append(SkipConnectBlock(
                self.rgb_encoder.down_8_channels_out,
                self.channels_decoder[1],
                kernel_size=1))
        self.skip_layer2 = nn.Sequential(*layers_skip2)

        layers_skip3 = list()
        if self.rgb_encoder.down_16_channels_out != self.channels_decoder[0]:
            layers_skip3.append(SkipConnectBlock(
                self.rgb_encoder.down_16_channels_out,
                self.channels_decoder[0],
                kernel_size=1))
        self.skip_layer3 = nn.Sequential(*layers_skip3)

        # context module
        if 'learned-3x3' in upsampling:
            upsampling_context_module = 'nearest'
        else:
            upsampling_context_module = upsampling
        self.context_module, channels_after_context_module = \
            get_context_module(
                'ppm',
                self.rgb_encoder.down_32_channels_out,
                self.channels_decoder[0],
                input_size=(height // 32, width // 32),
                activation=nn.ReLU(inplace=True),
                upsampling_mode=upsampling_context_module
            )
        
        # decoder
        self.decoder = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=self.channels_decoder,
            upsampling_mode=upsampling,
            num_classes=self.num_classes
        )
        

    def forward(self, rgb, depth):

        # stem
        rgb = self.rgb_encoder.forward_stem(rgb)
        depth = self.depth_encoder.forward_stem(depth)
        fusion = self.stem_fusion(rgb, depth)
        
        # block 1
        rgb = self.rgb_encoder.forward_layer1(fusion)
        depth = self.depth_encoder.forward_layer1(depth)
        fusion = self.layer1_fusion(rgb, depth)
        skip1 = self.skip_layer1(fusion)

        # block 2
        rgb = self.rgb_encoder.forward_layer2(fusion)
        depth = self.depth_encoder.forward_layer2(depth)
        fusion = self.layer2_fusion(rgb, depth)
        skip2 = self.skip_layer2(fusion)

        # block 3
        rgb = self.rgb_encoder.forward_layer3(fusion)
        depth = self.depth_encoder.forward_layer3(depth)
        fusion = self.layer3_fusion(rgb, depth)
        skip3 = self.skip_layer3(fusion)

        # block 4
        rgb = self.rgb_encoder.forward_layer4(fusion)
        depth = self.depth_encoder.forward_layer4(depth)
        fusion = self.layer4_fusion(rgb, depth)
        
        # context module
        out = self.context_module(fusion)

        # decoder
        out = self.decoder(enc_outs=[out, skip3, skip2, skip1])

        return out

class SkipConnectBlock(nn.Sequential):
    r"""SkipConnectBlock is used for recovering the lost of information details.
    The encoder feature map is fused with the decoder results of the corresponding layer, 
    so that the information details lost during encoding can be recovered.
    """
    def __init__(self, channels_in, channels_out, kernel_size, dilation=1, stride=1):
        super(SkipConnectBlock, self).__init__()
        r"""
        Initialize SkipConnectBlock
        Args:
            channels_in: input channels depends on output channel of the previous layer, 
                        layer 1: down_4_channels_out  
                        layer 2: down_8_channels_out
                        layer 2: down_16_channels_out
            channels_out: output channels, self.channels_decoder depends on convnext_x 
                        if using convnext_tiny self.channels_decoder = [192, 384, 768]
                        if using convnext_b self.channels_decoder = [256, 512, 1024]
            kernel_size: default=1
            dilation: default=1
            stride: default=1
        """
        padding = kernel_size // 2 + dilation - 1
        self.add_module('conv', nn.Conv2d(channels_in, channels_out,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          bias=False,
                                          dilation=dilation,
                                          stride=stride))
        self.add_module('bn', nn.BatchNorm2d(channels_out))
        self.add_module('act', nn.ReLU(inplace=True))

class Decoder(nn.Module):
    f"""Decoder Module which is used for recovering resolution of feature map.
    The decoder consists of 3 DecoderModule modules and 2 Upsample modules.
    """
    def __init__(self,
                 channels_in,
                 channels_decoder,
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()
        f"""
        Initialize Decoder
        Args:
            channels_in: output of context modules
            channels_decoder: output channels, self.channels_decoder depends on convnext_x 
                              if using convnext_tiny self.channels_decoder = [192, 384, 768]
                              if using convnext_b self.channels_decoder = [256, 512, 1024]
            upsampling_mode: 
            num_classes: the number of classes of dataset
        """

        self.decoder_module_1 = DecoderModule(
            channels_in=channels_in,
            channels_dec=channels_decoder[0],
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.decoder_module_2 = DecoderModule(
            channels_in=channels_decoder[0],
            channels_dec=channels_decoder[1],
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.decoder_module_3 = DecoderModule(
            channels_in=channels_decoder[1],
            channels_dec=channels_decoder[2],
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.conv_out = nn.Conv2d(channels_decoder[2], num_classes, kernel_size=3, padding=1)

        self.upsample1 = Upsample(mode=upsampling_mode,
                                  channels=num_classes)
        self.upsample2 = Upsample(mode=upsampling_mode,
                                  channels=num_classes)
        
    
    def forward(self, enc_outs):
        enc_out, enc_skip_down_16, enc_skip_down_8, enc_skip_down_4 = enc_outs

        out, out_down_32 = self.decoder_module_1(enc_out, enc_skip_down_16)
        out, out_down_16 = self.decoder_module_2(out, enc_skip_down_8)
        out, out_down_8 = self.decoder_module_3(out, enc_skip_down_4)

        out = self.conv_out(out)
        out = self.upsample1(out)
        out = self.upsample2(out)

        if self.training:
            return out, out_down_8, out_down_16, out_down_32
        return out


class DecoderModule(nn.Module):
    r"""
    Dncoding sub-module, which mainly consists of 3*3 ConvBNAct, 3 NonBottleneck1D blocks,
    and a Upsample modules. This is the core module of the decoder.
    """
    def __init__(self,
                 channels_in,
                 channels_dec,
                 activation=nn.ReLU(inplace=True),
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()
        r"""
        Initialize DecoderModule
        Args:
            channels_in: 
            channels_dec: 
            activation=nn.ReLU(inplace=True),
            upsampling_mode='bilinear',
            num_classes: the number of classes of dataset
        """

        self.conv3x3 = ConvBNAct(channels_in, channels_dec, kernel_size=3,
                                 activation=activation)
        
        blocks = []
        for _ in range(3):
            blocks.append(NonBottleneck1D(channels_dec,
                                          channels_dec,
                                          activation=activation))
        self.decoder_blocks = nn.Sequential(*blocks)

        self.upsample = Upsample(mode=upsampling_mode,
                                 channels=channels_dec)

        # for pyramid supervision
        self.side_output = nn.Conv2d(channels_dec,
                                     num_classes,
                                     kernel_size=1)

    def forward(self, decoder_features, encoder_features):
        out = self.conv3x3(decoder_features)
        out = self.decoder_blocks(out)

        if self.training:
            out_side = self.side_output(out)
        else:
            out_side = None

        out = self.upsample(out)

        out += encoder_features

        return out, out_side

class Upsample(nn.Module):
    def __init__(self, mode, channels=None):
        super(Upsample, self).__init__()
        self.interp = nn.functional.interpolate

        if mode == 'bilinear':
            self.align_corners = False
        else:
            self.align_corners = None

        if 'learned-3x3' in mode:
            # mimic a bilinear interpolation by nearest neigbor upscaling and
            # a following 3x3 conv. Only works as supposed when the
            # feature maps are upscaled by a factor 2.

            if mode == 'learned-3x3':
                self.pad = nn.ReplicationPad2d((1, 1, 1, 1))
                self.conv = nn.Conv2d(channels, channels, groups=channels, # type: ignore
                                      kernel_size=3, padding=0)
            elif mode == 'learned-3x3-zeropad':
                self.pad = nn.Identity()
                self.conv = nn.Conv2d(channels, channels, groups=channels, # type: ignore
                                      kernel_size=3, padding=1)

            # kernel that mimics bilinear interpolation
            w = torch.tensor([[[
                [0.0625, 0.1250, 0.0625],
                [0.1250, 0.2500, 0.1250],
                [0.0625, 0.1250, 0.0625]
            ]]])

            self.conv.weight = torch.nn.Parameter(torch.cat([w] * channels)) # type: ignore

            # set bias to zero
            with torch.no_grad():
                self.conv.bias.zero_() # type: ignore

            self.mode = 'nearest'
        else:
            # define pad and conv just to make the forward function simpler
            self.pad = nn.Identity()
            self.conv = nn.Identity()
            self.mode = mode

    def forward(self, x):
        size = (int(x.shape[2]*2), int(x.shape[3]*2))
        x = self.interp(x, size, mode=self.mode,
                        align_corners=self.align_corners)
        x = self.pad(x)
        x = self.conv(x)
        return x

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

class NonBottleneck1D(nn.Module):
    """
    ERFNet-Block
    Paper:
    http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf
    Implementation from:
    https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=None, dilation=1, norm_layer=None,
                 activation=nn.ReLU(inplace=True), residual_only=False):
        super().__init__()
        dropprob = 0
        self.conv3x1_1 = nn.Conv2d(inplanes, planes, (3, 1),
                                   stride=(stride, 1), padding=(1, 0),
                                   bias=True)
        self.conv1x3_1 = nn.Conv2d(planes, planes, (1, 3),
                                   stride=(1, stride), padding=(0, 1),
                                   bias=True)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-03)
        self.act = activation
        self.conv3x1_2 = nn.Conv2d(planes, planes, (3, 1),
                                   padding=(1 * dilation, 0), bias=True,
                                   dilation=(dilation, 1))
        self.conv1x3_2 = nn.Conv2d(planes, planes, (1, 3),
                                   padding=(0, 1 * dilation), bias=True,
                                   dilation=(1, dilation))
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)
        self.downsample = downsample
        self.stride = stride
        self.residual_only = residual_only

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = self.act(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.act(output)

        output = self.conv3x1_2(output)
        output = self.act(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        if self.downsample is None:
            identity = input
        else:
            identity = self.downsample(input)

        if self.residual_only:
            return output
        # +input = identity (residual connection)
        return self.act(output + identity)