# based on PyTorch ResNet implementation
# from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssd.lib.models.modules import BasicBlock, add_extras
from ssd.lib.prior_boxes import PriorBox

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResNet_SSDTC(nn.Module):

    def __init__(self, block, layers, extras, num_classes=1000, img_size=(224, 224), phase='train'):
        super(ResNet_SSDTC, self).__init__()
        self.height, self.width = img_size
        self.num_classes = num_classes
        self.phase = phase
        self.inplanes = 64
        self.chunk = 32

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.extras = nn.ModuleList(extras)

        self.temp_layers = nn.ModuleList()
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        self.softmax = nn.Softmax(dim=-1)
        self.base = [self.conv1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4]

        maps, self.feature_sizes = self._mock_run()

        # generate priorboxes for predetermined image size and feature layer sizes
        self.box_sizes = [(30, 60), (60, 111), (111, 162), (162, 213), (213, 264), (264, 315)]
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

        self.priors = PriorBox((self.height, self.width), maps, self.box_sizes, self.aspect_ratios)
        pbox = self.priors.num_boxes

        channels = [512, 512, 512, 256, 256, 256]
        for f, ch in zip(self.feature_sizes, channels):
            self.temp_layers.append(
                nn.Sequential(nn.Conv3d(f[1], ch // 2, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0)).cuda(),
                              nn.BatchNorm3d(ch // 2),
                              nn.ReLU(inplace=True),
                              nn.Conv3d(ch // 2, ch, kernel_size=(3, 3, 3), stride=1, padding=(0, 1, 1)).cuda(),
                              nn.BatchNorm3d(ch),
                              nn.ReLU(inplace=True)
                              ).cuda())

        # create multibox confidence and location heads
        for b, ch in zip(pbox, channels):
            self.loc_layers.append(nn.Conv2d(ch, b * 4, kernel_size=3, stride=1, padding=1).cuda())
            self.conf_layers.append(nn.Conv2d(ch, b * num_classes, kernel_size=3, stride=1, padding=1).cuda())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        c = ["conf_layers." + k for k, v in self.conf_layers.named_parameters()]
        l = ["loc_layers." + k for k, v in self.loc_layers.named_parameters()]
        self.head = c + l

    def regen_priors(self, img_size, mock):
        self.height, self.width = img_size
        feature_sizes = self._mock_run(mock)
        self.priors = PriorBox(img_size, feature_sizes, self.box_sizes, self.aspect_ratios)

    def _mock_run(self, mock=None):
        with torch.no_grad():
            x = mock
            if x is None:
                x = torch.zeros([1, 3, self.height, self.width], dtype=torch.float)

            feature_sizes = []

            x = self.conv1(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            feature_sizes.append(list(x.shape))
            x = self.layer3(x)
            feature_sizes.append(list(x.shape))
            x = self.layer4(x)
            feature_sizes.append(list(x.shape))

            for k, v in enumerate(self.extras):
                x = v(x)
                if k % 2 == 1:
                    feature_sizes.append(list(x.shape))

            maps = feature_sizes
            feature_sizes = [[1, f[1], self.chunk, f[2], f[3]] for f in feature_sizes]

        return maps, feature_sizes

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

        features = [torch.unsqueeze(torch.transpose(f, 0, 1), 0) for f in features]
        temp_features = []

        for f, t in zip(features, self.temp_layers):
            temp_features.append(torch.transpose(torch.squeeze(t(f), 0), 0, 1))

        loc = []
        conf = []

        for f, l, c in zip(temp_features, self.loc_layers, self.conf_layers):
            loc.append(l(f).permute(0, 2, 3, 1).contiguous())
            conf.append(c(f).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([l.view(l.size(0), -1) for l in loc], 1)
        conf = torch.cat([c.view(c.size(0), -1) for c in conf], 1)

        if self.phase == 'infer':
            output = (
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1, self.num_classes))
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes)
            )
        return output

cfg = {
    'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    'extras': [256, 'S', 512, 128, 'S', 256, 128, 256],
    'mbox': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
}


# def resnet18(num_classes, img_size, phase):
#     extras = add_extras(cfg['extras'], 512)
#     model = ResNet_SSD(BasicBlock, [2, 2, 2, 2], extras, num_classes, img_size, phase)
#     return model


def resnet34_ssdtc(num_classes, img_size, phase):
    extras = add_extras(cfg['extras'], 512)
    model = ResNet_SSDTC(BasicBlock, [3, 4, 6, 3], extras, num_classes, img_size, phase)
    return model

#
# def resnet50(num_classes, img_size, phase):
#     extras = add_extras(cfg['extras'], 2048)
#     model = ResNet_SSD(Bottleneck, [3, 4, 6, 3], extras, num_classes, img_size, phase)
#     return model
#
#
# def resnet101(num_classes, img_size, phase):
#     extras = add_extras(cfg['extras'], 2048)
#     model = ResNet_SSD(Bottleneck, [3, 4, 23, 3], extras, num_classes, img_size, phase)
#     return model
#
#
# def resnet152(num_classes, img_size, phase):
#     extras = add_extras(cfg['extras'], 2048)
#     model = ResNet_SSD(Bottleneck, [3, 8, 36, 3], extras, num_classes, img_size, phase)
#     return model
