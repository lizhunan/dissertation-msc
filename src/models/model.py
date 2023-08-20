
import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion import SEFusion, ECAFusion
from context_modules import get_context_module
from convnext import convnext_tiny, convnext_base

class EISSegNet(nn.Module):

    def __init__(self, 
                 height=480,
                 width=640,
                 num_classes=37,
                 rgb_encoder='convnext_t',
                 depth_encoder='convnext_t',
                 encoder_block='BasicBlock',
                 fusion_module='SE',
                 channels_decoder=[768, 384, 192],
                 nr_decoder_blocks=[3,3,3],
                 pretrained=True,
                 pretrained_dir='./trained_models',
                 upsampling='bilinear'):
        super(EISSegNet, self).__init__()

        self.activation = nn.ReLU(inplace=True)

        # rgb encoder
        if rgb_encoder == 'convnext_t':
            self.rgb_encoder = convnext_tiny(pretrained=True)
        elif rgb_encoder == 'convnext_b':
            self.rgb_encoder = convnext_base(pretrained=True)
        else:
            raise NotImplementedError(f'Only convnext_t or convnext_b are supported for rgb encoder. Got {rgb_encoder}')
        
        # depth encoder
        if depth_encoder == 'convnext_t':
            self.depth_encoder = convnext_tiny(pretrained=True)
        elif depth_encoder == 'convnext_b':
            self.depth_encoder = convnext_base(pretrained=True)
        else:
            raise NotImplementedError(f'Only convnext_t or convnext_b are supported for depth encoder. Got {depth_encoder}')
        
        # fusion module
        if fusion_module == 'SE':
            self.stem_fusion = SEFusion(96)
            self.layer1_fusion = SEFusion(96)
            self.layer2_fusion = SEFusion(192)
            self.layer3_fusion = SEFusion(384)
            self.layer4_fusion = SEFusion(768)
        elif fusion_module == 'ECA':
            self.stem_fusion = ECAFusion(96)
            self.layer1_fusion = ECAFusion(96)
            self.layer2_fusion = ECAFusion(192)
            self.layer3_fusion = ECAFusion(384)
            self.layer4_fusion = ECAFusion(768)
        else:
            raise NotImplementedError(f'Only SE or ECA are supported for fusion module. Got {fusion_module}')

        # skip connection layers
        layers_skip1 = list()
        if self.rgb_encoder.down_4_channels_out != channels_decoder[2]:
            layers_skip1.append(SkipConnectBlock(
                self.rgb_encoder.down_4_channels_out,
                channels_decoder[2],
                kernel_size=1))
        self.skip_layer1 = nn.Sequential(*layers_skip1)

        layers_skip2 = list()
        if self.rgb_encoder.down_8_channels_out != channels_decoder[1]:
            layers_skip2.append(SkipConnectBlock(
                self.rgb_encoder.down_8_channels_out,
                channels_decoder[1],
                kernel_size=1))
        self.skip_layer2 = nn.Sequential(*layers_skip2)

        layers_skip3 = list()
        if self.rgb_encoder.down_16_channels_out != channels_decoder[0]:
            layers_skip3.append(SkipConnectBlock(
                self.rgb_encoder.down_16_channels_out,
                channels_decoder[0],
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
                channels_decoder[0],
                input_size=(height // 32, width // 32),
                activation=self.activation,
                upsampling_mode=upsampling_context_module
            )
        
        # decoder
        
        self.decoder = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion='add',
            upsampling_mode=upsampling,
            num_classes=num_classes
        )
        

    def forward(self, rgb, depth):
        _LOGGER('####################', 'EISSegNet forward input shape',
                rgb_shape=rgb.shape, depth_shape=depth.shape)
        
        rgb = self.rgb_encoder.forward_stem(rgb)
        depth = self.rgb_encoder.forward_stem(depth)
        fusion = self.stem_fusion(rgb, depth)
        _LOGGER('####################', 'EISSegNet forward_stem shape',
                rgb_shape=rgb.shape, depth_shape=depth.shape, fusion_shape=fusion.shape)

        # block 1
        rgb = self.rgb_encoder.forward_layer1(fusion)
        depth = self.depth_encoder.forward_layer1(depth)
        fusion = self.layer1_fusion(rgb, depth)
        skip1 = self.skip_layer1(fusion)
        _LOGGER('####################', 'EISSegNet forward_layer1 shape',
                rgb_shape=rgb.shape, depth_shape=depth.shape, 
                fusion_shape=fusion.shape, skip1=skip1.shape)

        # block 2
        rgb = self.rgb_encoder.forward_layer2(fusion)
        depth = self.depth_encoder.forward_layer2(depth)
        fusion = self.layer2_fusion(rgb, depth)
        skip2 = self.skip_layer2(fusion)
        _LOGGER('####################', 'EISSegNet forward_layer2 shape',
                rgb_shape=rgb.shape,depth_shape=depth.shape, 
                fusion_shape=fusion.shape, skip2=skip2.shape)

        # block 3
        rgb = self.rgb_encoder.forward_layer3(fusion)
        depth = self.depth_encoder.forward_layer3(depth)
        fusion = self.layer3_fusion(rgb, depth)
        skip3 = self.skip_layer3(fusion)
        _LOGGER('####################', 'EISSegNet forward_layer3 shape',
                rgb_shape=rgb.shape,depth_shape=depth.shape, 
                fusion_shape=fusion.shape, skip3=skip3.shape)

        # block 4
        rgb = self.rgb_encoder.forward_layer4(fusion)
        depth = self.depth_encoder.forward_layer4(depth)
        fusion = self.layer4_fusion(rgb, depth)
        _LOGGER('####################', 'EISSegNet forward_layer4 shape',
                rgb_shape=rgb.shape,depth_shape=depth.shape, fusion_shape=fusion.shape)
        
        # context module
        out = self.context_module(fusion)

        # decoder
        out = self.decoder(enc_outs=[out, skip3, skip2, skip1])

        return out

class SkipConnectBlock(nn.Sequential):
    def __init__(self, channels_in, channels_out, kernel_size, dilation=1, stride=1):
        super(SkipConnectBlock, self).__init__()
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
    def __init__(self,
                 channels_in,
                 channels_decoder,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=1,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()
    
    def forward(self, enc_outs):
        pass

counter = 0
def _LOGGER(tag, info, **kwargs):
    global counter
    counter += 1
    print('===============================================================')
    print(f'EISSegNet Counter: {counter}')
    print(f'TAG: {tag}')
    print(f'INFO: {info}')
    for key, value in kwargs.items():
        print(f"kwarg: {key}: {value}")
    print('===============================================================')

if __name__ == '__main__':
    _LOGGER('11111111111111111111111111', 'START')
    model = EISSegNet()

    model.eval()
    print(model)

    rgb = torch.randn(1, 3, 480, 640)
    depth = torch.randn(1, 3, 480, 640)

    _LOGGER('', 'input rgb shape', shape=rgb.shape)
    _LOGGER('', 'input depth shape', shape=depth.shape)
    
    with torch.no_grad():
        outputs = model(rgb, depth)
        _LOGGER('oooooooooooooooooooooooout', 'output shape', shape=len(outputs))
    for tensor in outputs:
        _LOGGER('ttttttttttttttttttttttttttttttenor', 'output tensor\'s shape', shape=tensor.shape)
        print(tensor.shape)