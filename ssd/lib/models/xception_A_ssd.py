"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)

@author: tstandley
Adapted by cadene

Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

Adapted to SSD by Marek Dobransky
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssd.lib.models.modules import SeparableConv2d, Xception_Block, add_extras
from ssd.lib.prior_boxes import PriorBox


class Xception(nn.Module):

    def __init__(self, extras, num_classes, img_size, phase):

        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.height, self.width = img_size
        self.phase = phase

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Xception_Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Xception_Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Xception_Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Xception_Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Xception_Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Xception_Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Xception_Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Xception_Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Xception_Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Xception_Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Xception_Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Xception_Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.extras = nn.ModuleList(extras)

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        self.softmax = nn.Softmax(dim=-1)

        self.feature_sizes = self._mock_run()

        self.box_sizes = [(30, 60), (60, 111), (111, 162), (162, 213), (213, 264), (264, 315)]
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

        self.priors = PriorBox((self.height, self.width), self.feature_sizes, self.box_sizes, self.aspect_ratios)
        pbox = self.priors.num_boxes

        # create multibox confidence and location heads
        for b, f in zip(pbox, self.feature_sizes):
            self.loc_layers.append(nn.Conv2d(f[1], b * 4, kernel_size=3, stride=1, padding=1).cuda())
            self.conf_layers.append(nn.Conv2d(f[1], b * num_classes, kernel_size=3, stride=1, padding=1).cuda())

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        c = ["conf_layers." + k for k, v in self.conf_layers.named_parameters()]
        l = ["loc_layers." + k for k, v in self.loc_layers.named_parameters()]
        self.head = c + l

    def _mock_run(self, mock=None):
        with torch.no_grad():
            x = mock
            if x is None:
                x = torch.zeros([1, 3, self.height, self.width], dtype=torch.float)

            feature_sizes = []

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

            x = self.block1(x)
            x = self.block2(x)
            feature_sizes.append(list(x.shape))

            x = self.block3(x)

            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
            x = self.block8(x)
            x = self.block9(x)
            x = self.block10(x)
            x = self.block11(x)
            feature_sizes.append(list(x.shape))

            x = self.block12(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)

            x = self.conv4(x)
            x = self.bn4(x)

            feature_sizes.append(list(x.shape))

            for k, v in enumerate(self.extras):
                x = v(x)
                if k % 2 == 1:
                    feature_sizes.append(list(x.shape))

        return feature_sizes

    def forward(self, x):
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        features.append(x)

        x = self.block3(x)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        features.append(x)

        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        features.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                features.append(x)

        loc = []
        conf = []

        for f, l, c in zip(features, self.loc_layers, self.conf_layers):
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
    'extras': [256, 'S', 512, 128, 'S', 256, 128, 256],
}


def xception(num_classes, img_size, phase):
    extras = add_extras(cfg['extras'], 2048)
    model = Xception(extras, num_classes, img_size, phase)
    return model
