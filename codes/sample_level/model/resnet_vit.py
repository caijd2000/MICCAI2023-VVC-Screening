import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision import utils as vutils
from torch.nn import functional as F
#from positionembedding import Position_Embedding
#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import torch
import cv2
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model import vit
from model.transformer import Transformer
from kmeans_pytorch import kmeans
 


def refine_cams(cam_original, image_shape ):
    if image_shape[0] != cam_original.size(2) or image_shape[1] != cam_original.size(3):
        cam_original = F.interpolate(
            cam_original, image_shape, mode="bilinear", align_corners=True
        )
    B, C, H, W = cam_original.size()
    cams = []
    for idx in range(C):
        cam = cam_original[:, idx, :, :]
        cam = cam.view(B, -1)
        cam_min = cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]
        norm = cam_max - cam_min
        norm[norm == 0] = 1e-5
        cam = (cam - cam_min) / norm
        cam = cam.view(B, H, W).unsqueeze(1)
        cams.append(cam)
    cams = torch.cat(cams, dim=1)
    return cams

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class FC(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.fc= nn.Conv2d(2048,2,kernel_size=1)
    def forward(self, x):
        return self.fc(x)



class GMP(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self,H,W):
        super().__init__()
        self.gmp=torch.nn.AdaptiveMaxPool2d((H,W))
    def forward(self, x):
        out=self.gmp(x)
        return out.permute(0,2,3,1)

class GAP(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self,H,W):
        super().__init__()
        self.gap=torch.nn.AdaptiveAvgPool2d((H,W))
    def forward(self, x):
        out=self.gap(x)
        return out
        

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)


    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]





class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        for p in self.parameters():
            p.requires_grad=False
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.sigmoid=torch.nn.Sigmoid()
        self.gap1= GAP(1,1)


        self.fc=nn.Linear(256,2)

        self.softmax=torch.nn.Softmax(dim=1)

        self.encoder_t14 = Transformer(d_model=256, nhead=8, num_decoder_layers=2,
                                    num_encoder_layers=2, dim_feedforward=256, dropout=0.1)

        self.dropout=nn.Dropout(0.5)

        self.vit256=vit.ViT(
        image_size = 256,
        patch_size = 16,
        num_classes = 2,
        dim = 256,
        depth = 1,
        heads = 16,
        mlp_dim = 256,
        channels = 256,
        dropout = 0.1,
        emb_dropout = 0.1
        )
        
        
        
        self.vit32=vit.ViT(
        image_size = 32,
        patch_size = 2,
        num_classes = 2,
        dim = 256,
        depth = 1,
        heads = 16,
        mlp_dim = 256,
        channels = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
        )
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        
        img_batch = inputs

        x = self.conv1(img_batch)#512
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)#256
        x1 = self.layer1(x)#256*256
        x2 = self.layer2(x1)#512*128
        x3 = self.layer3(x2)#1024*64
        x4 = self.layer4(x3)#b*2048*32*32
        
        att256 = self.vit256(x1)#b*65*256
        att32 = self.vit32(x4)#b*65*256
        att_combine14 = self.encoder_t14(att256,att32)
        feature = att_combine14[:,0,:]
        em = att_combine14
        
        em = self.fc(em)
        final=em[:,0,:]
        em_reshape=em[:,1:,:].reshape(em.shape[0],16,16,2)
        em_reshape=em_reshape.transpose(2, 3).transpose(1, 2)
        cam=refine_cams(em_reshape, image_shape=[1024,1024] )
        
        
        return final,cam,feature

def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model

