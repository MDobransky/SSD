# based on PyTorch ResNet implementation
# from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

#ResNet and extra layers of ResNet34-SSDTC model in 'resnet_ssdtc.py'
#used exclusivelly for training paired with 'ssdtc_head' module
# used in 'ssdtc.py'

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssd.lib.models.modules import BasicBlock, add_extras


class ResNet_features(nn.Module):

    def __init__(self, block, layers, extras):
        super(ResNet_features, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.extras = nn.ModuleList(extras)

        self.base = [self.conv1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            features = []

            x = self.layer1(x)
            x = self.layer2(x)
            features.append(x)
            x = self.layer3(x)
            features.append(x)
            x = self.layer4(x)
            features.append(x)

            for k, v in enumerate(self.extras):
                x = F.relu(v(x), inplace=True)
                if k % 2 == 1:
                    features.append(x)

            return features


cfg = {
    'extras': [256, 'S', 512, 128, 'S', 256, 128, 256],
}


def resnet34_features():
    extras = add_extras(cfg['extras'], 512)
    model = ResNet_features(BasicBlock, [3, 4, 6, 3], extras)
    return model
