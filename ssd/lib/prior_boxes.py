# inspired by https://github.com/ruinmessi/RFBNet/blob/master/layers/functions/prior_box.py

import itertools

import numpy as np
import torch


class PriorBox:
    def __init__(self, img_size, features, box_sizes, aspect_ratios, clip=True):
        self.boxes = []
        self.num_boxes = [2 + 2 * len(x) for x in aspect_ratios]
        self.height, self.width = img_size

        for idx in range(0, len(features)):
            h_scale = self.height / np.ceil(self.height / features[idx][2])
            w_scale = self.width / np.ceil(self.width / features[idx][3])
            for i, j in itertools.product(range(features[idx][2]), range(features[idx][3])):
                x_center = (j + 0.5) / w_scale
                y_center = (i + 0.5) / h_scale

                # small sized square box
                size = box_sizes[idx][0]
                h = size / self.height
                w = size / self.width
                self.boxes.append([x_center, y_center, w, h])

                # rectangular boxes
                for ratio in aspect_ratios[idx]:
                    ratio = np.sqrt(ratio)
                    self.boxes.append([x_center, y_center, w * ratio, h / ratio])
                    self.boxes.append([x_center, y_center, w / ratio, h * ratio])

                # big sized square box
                size = np.sqrt(box_sizes[idx][0] * box_sizes[idx][1])
                h = size / self.height
                w = size / self.width
                self.boxes.append([x_center, y_center, w, h])

        self.boxes = np.array(self.boxes)
        if clip:
            self.boxes = np.clip(self.boxes, 0.0, 1.0)

        self.boxes = torch.from_numpy(np.array(self.boxes)).float().cuda()
