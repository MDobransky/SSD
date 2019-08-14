# based on PyTorch VGG implementation
# from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
#
# SSD adaptation based on SSD from amdegroot
# https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssd.lib.models.modules import L2Norm, add_extras
from ssd.lib.prior_boxes import PriorBox

__all__ = [
    'VGG_SSD', 'vgg16',
]

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}


class VGG_SSD(nn.Module):

    def __init__(self, base, extras, head, num_classes=1000, img_size=(224, 224), phase='train', init_weights=True):
        super(VGG_SSD, self).__init__()
        self.height, self.width = img_size
        self.phase = phase
        self.num_classes = num_classes
        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.loc_layers = nn.ModuleList(head[0])
        self.conf_layers = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1)
        self.L2Norm = L2Norm(512, 20)

        self.feature_sizes = self._mock_run()

        box_sizes = [(30, 60), (60, 111), (111, 162), (162, 213), (213, 264), (264, 315)]
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.priors = PriorBox(img_size, self.feature_sizes, box_sizes, aspect_ratios)

        if init_weights:
            self._initialize_weights()

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

            for k in range(23):
                x = self.base[k](x)
            feature_sizes.append(list(x.shape))

            for k in range(23, len(self.base)):
                x = self.base[k](x)
            feature_sizes.append(list(x.shape))

            for k, v in enumerate(self.extras):
                x = v(x)
                if k % 2 == 1:
                    feature_sizes.append(list(x.shape))

        return feature_sizes

    def forward(self, x):

        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.base[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to conv7
        for k in range(23, len(self.base)):
            x = self.base[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        for (x, l, c) in zip(sources, self.loc_layers, self.conf_layers):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)


cfg = {
    'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    'extras': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    'mbox': [4, 6, 6, 6, 4, 4],
}


def vgg16(num_classes, img_size, phase):
    base = make_layers(cfg['base'])
    extras = add_extras(cfg['extras'], in_channels=1024)
    head = multibox(base, extras, cfg['mbox'], num_classes)
    return VGG_SSD(base, extras, head, num_classes, img_size, phase)
