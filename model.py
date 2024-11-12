from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
# import segmentation_models_pytorch as smp
from torchvision.models import squeezenet1_0, mobilenet_v3_small, shufflenet_v2_x0_5, efficientnet_b0
# from efficientnet_pytorch import EfficientNet
# from torchvision.models import shufflenet_v2_x0_5
# from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead
# from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
# from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.conv5 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv4(x)))
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.relu(self.bn5(self.conv5(x)))
        return x

class Mobi(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.enc = squeezenet1_0(pretrained=False)
        #self.enc.conv5 = nn.Identity()
        self.enc.classifier  = nn.Identity()
        self.aspp = ASPP(512, 82)
        self.segmentation_head = nn.Conv2d(82, 19, kernel_size=1)
        #self.enc = self.enc[:-1]
        # encoder_rgb = get_encoder(
        #     encoder_name,
        #     in_channels=in_channels,
        #     depth=encoder_depth,
        #     weights=encoder_weights,
        # )
        # self.decoder = smp.DeepLabV3Plus(
        #     encoder_name='squeezenet1_0',    # Use SqueezeNet1_0 as the encoder
        #     encoder_weights=None,            # Already loaded pretrained weights for encoder
        #     in_channels=3,                   # Input is 3-channel RGB images
        #     classes=19               # Number of output segmentation classes
        # )
        #self.decoder = DeepLabHead(1024, 19)
        #self.seg_head = SegmentationHead(decoder_channels, classes, activation=activation, upsampling=upsampling)

    def forward(self, x):
        features = self.enc.features(x)
        #print(self.enc)
        #print(features.shape)
        decoder_output = self.aspp(features)
        logit = self.segmentation_head(decoder_output)
        #multi_features = [decoder_output]
        return F.interpolate(logit, size=x.shape[2:], mode='bilinear', align_corners=False)
    
class Mobief(nn.Module):
    def __init__(self):
        
        super().__init__()
        enc = efficientnet_b0(pretrained=False)
        self.enc = nn.Sequential(*list(enc.features.children()))
        #self.enc.conv5 = nn.Identity()
        #self.enc.classifier  = nn.Identity()
        self.aspp = ASPP(768, 19)
        #self.segmentation_head = nn.Conv2d(82, 19, kernel_size=1)
        #self.enc = self.enc[:-1]
        # encoder_rgb = get_encoder(
        #     encoder_name,
        #     in_channels=in_channels,
        #     depth=encoder_depth,
        #     weights=encoder_weights,
        # )
        # self.decoder = smp.DeepLabV3Plus(
        #     encoder_name='squeezenet1_0',    # Use SqueezeNet1_0 as the encoder
        #     encoder_weights=None,            # Already loaded pretrained weights for encoder
        #     in_channels=3,                   # Input is 3-channel RGB images
        #     classes=19               # Number of output segmentation classes
        # )
        #self.decoder = DeepLabHead(1024, 19)
        #self.seg_head = SegmentationHead(decoder_channels, classes, activation=activation, upsampling=upsampling)

    def forward(self, x):
        features = self.enc(x)
        #print(self.enc)
        #print(features.shape)
        decoder_output = self.aspp(features)
        #logit = self.segmentation_head(decoder_output)
        #multi_features = [decoder_output]
        return F.interpolate(decoder_output, size=x.shape[2:], mode='bilinear', align_corners=False)
