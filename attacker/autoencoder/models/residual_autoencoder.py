import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet

from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1, conv3x3

class ResidualBlock_Enc(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock_Enc, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace = True)
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class ResidualBlock_Dec(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, upsample = None):
        super(ResidualBlock_Dec, self).__init__()
        self.deconv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
        )
        self.deconv2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
        )
        self.relu = nn.ReLU(inplace = True)
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.deconv1(x)
        out = self.deconv2(out)
        if self.upsample:
            residual = self.upsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class Residual_Autoencoder(nn.Module):
    def __init__(self, block, layers, num_feats) -> None:
        super(Residual_Autoencoder, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )
        self.MaxPool_enc = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0_enc = self._make_layer_enc(block, 64, layers[0], stride=1)
        self.layer1_enc = self._make_layer_enc(block, 128, layers[1], stride=2)
        self.layer2_enc = self._make_layer_enc(block, 256, layers[2], stride=2)
        self.layer3_enc = self._make_layer_enc(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(512, num_feats)

        self.fc_dec = nn.Linear(num_feats, 512)
        self.unflatten = nn.Unflatten(1, (512, 1, 1))
        self.avgunpool = nn.MaxUnpool2d(7, stride=1)
        self.layer3_dec = self._make_layer_dec(block, 512, layers[3], stride=2)
        self.layer2_dec = self._make_layer_dec(block, 256, layers[2], stride=2)
        self.layer1_dec = self._make_layer_dec(block, 128, layers[1], stride=2)
        self.layer0_dec = self._make_layer_dec(block, 64, layers[0], stride=1)
        self.MaxUnpool_dec = nn.MaxUnpool2d(kernel_size = 3, stride = 2, padding = 1)
        self.deconv1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 3, kernel_size = 7, stride = 2, padding = 3, bias = False),
        )

        self.encoder = nn.Sequential(
            self.conv1,
            self.MaxPool_enc,
            self.layer0_enc,
            self.layer1_enc,
            self.layer2_enc,
            self.layer3_enc,
            self.avgpool,
            self.flatten,
            self.fc_enc
        )

        self.decoder = nn.Sequential(
            self.fc_dec,
            self.unflatten,
            self.avgunpool,
            self.layer3_dec,
            self.layer2_dec,
            self.layer1_dec,
            self.layer0_dec,
            self.MaxUnpool_dec,
            self.deconv1
        )


    def _make_layer_enc(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size = 3, stride = stride, bias = False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace = True)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, block):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def _make_layer_dec(self, block, planes, stride=1):
        if stride != 1 or self.inplanes != planes:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes, kernel_size=3, stride=stride, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(planes)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes
        for i in range(1, block):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)