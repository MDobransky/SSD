# Temporal and detection layers for ResNet34-SSDTC model in 'resnet_ssdtc.py'
# used exclusivelly for training paired with 'ssdtc_base' module
# used in 'ssdtc.py'

import torch
import torch.nn as nn

from ssd.lib.prior_boxes import PriorBox


class SSDTC_det(nn.Module):

    def __init__(self, num_classes=1000, img_size=(224, 224), phase='train'):
        super(SSDTC_det, self).__init__()
        self.height, self.width = img_size
        self.num_classes = num_classes
        self.phase = phase
        self.inplanes = 64
        self.chunk = 32

        self.temp_layers = nn.ModuleList()
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        self.softmax = nn.Softmax(dim=-1)

        # detection feature map sizes of ResNet34-SSD
        maps = [[1, 128, 38, 38], [1, 256, 19, 19], [1, 512, 10, 10], [1, 512, 5, 5], [1, 256, 3, 3], [1, 256, 1, 1]]

        # generate priorboxes for predetermined image size and feature layer sizes
        self.box_sizes = [(30, 60), (60, 111), (111, 162), (162, 213), (213, 264), (264, 315)]
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

        self.priors = PriorBox((self.height, self.width), maps, self.box_sizes, self.aspect_ratios)
        pbox = self.priors.num_boxes

        channels = [512, 512, 512, 256, 256, 256]
        for m, ch in zip(maps, channels):
            self.temp_layers.append(nn.Sequential(nn.Conv3d(m[1], ch // 2, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0)),
                                                  nn.BatchNorm3d(ch // 2),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv3d(ch // 2, ch, kernel_size=(3, 3, 3), stride=1, padding=(0, 1, 1)),
                                                  nn.BatchNorm3d(ch),
                                                  nn.ReLU(inplace=True)).cuda())

        # create multibox confidence and location heads
        for b, ch in zip(pbox, channels):
            self.loc_layers.append(nn.Conv2d(ch, b * 4, kernel_size=3, stride=1, padding=1).cuda())
            self.conf_layers.append(nn.Conv2d(ch, b * num_classes, kernel_size=3, stride=1, padding=1).cuda())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        temp_features = []

        for f, t in zip(x, self.temp_layers):
            tf = torch.squeeze(t(torch.transpose(f, 1, 2)), 2)
            temp_features.append(tf)

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
