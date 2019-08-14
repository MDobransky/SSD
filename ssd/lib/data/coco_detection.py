import os.path

import cv2
import numpy as np
import torch
import torch.utils.data as data
from pycocotools.coco import COCO

from ssd.lib.data.preprocess import BaseTransform, normalize_targets

cv2.setNumThreads(2)


class CocoDetection(data.Dataset):

    def __init__(self, root, annFile, preprocess, classes=None):
        """
           Args:
               root (string): image folder
               annFile (string): annotation file.
               preprocess (callable, optional): A function/transform that  takes in an cv2 image
                   and returns a tensor of correct size
        """

        self.root = root
        self.coco = COCO(annFile)

        self.catIds = self.coco.getCatIds()
        if classes is not None:
            self.catIds = self.coco.getCatIds(catNms=classes)
        self.ids = self.coco.getImgIds()

        self.preproc = preprocess
        self.transform = BaseTransform()
        self.categories = len(self.catIds) + 1
        self.catid_to_netid = dict(zip([0] + self.catIds, range(self.categories)))

    def __len__(self):
        return len(self.ids)

    def reformat_bbox(self, bbox, width, height):
        # fix size and reformat to 2 point representation
        x1 = np.max((0, bbox[0]))
        y1 = np.max((0, bbox[1]))
        x2 = np.min((width - 1, x1 + np.max((0, bbox[2] - 1))))
        y2 = np.min((height - 1, y1 + np.max((0, bbox[3] - 1))))
        return [x1, y1, x2, y2]

    def valid_bbox(self, bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > 0

    def load_anns(self, ann_ids, img_info):
        target = self.coco.loadAnns(ann_ids)
        bboxes = []
        for t in target:
            bbox = self.reformat_bbox(t['bbox'], img_info['width'], img_info['height'])
            category = self.catid_to_netid[t['category_id']]
            if self.valid_bbox(bbox):
                bbox.append(category)
                bboxes.append(bbox)
        if len(bboxes) == 0:
            bboxes = [[0., 0., 0., 0., 0.]]
        bboxes = np.array(bboxes)

        return bboxes

    def __getitem__(self, index):
        img, targets = self.get_image(index)
        # cv2.imwrite("data/JPEGImages/img" + str(index) + ".png", img)
        img = self.transform(img)
        targets = normalize_targets(targets, img.shape[2], img.shape[1])

        if len(targets) == 0:
            targets = np.array([[0., 0., 0., 0., 0]])

        return img, targets

    def get_image(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.catIds, iscrowd=None)

        image_info = self.coco.loadImgs(img_id)[0]
        path = image_info['file_name']

        targets = self.load_anns(ann_ids, image_info)

        path = os.path.join(self.root, path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        if self.preproc is not None:
            img, targets = self.preproc(img, targets)

        return img, targets


def collate(batch):
    """
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets, imgs = [], []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                t = torch.from_numpy(tup).float()
                targets.append(t)
    return (torch.stack(imgs, 0), targets)
