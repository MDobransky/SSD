import json
import os.path
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data

from ssd.lib.data.preprocess import BaseTransform, normalize_targets

cv2.setNumThreads(2)


class HHeadsDetection(data.Dataset):

    def __init__(self, root, annFile, chunk=None, preprocess=None, eval=False):
        """
           Args:
               root (string): image folder
               annFile (string): annotation file.
               preprocess (callable, optional): A function/transform that  takes in an cv2 image
                   and returns a tensor of correct size
        """

        self.root = root
        self.chunk = chunk
        self.eval = eval

        with open(annFile) as f:
            self.data = json.load(f)

        self.ids = sorted(list(self.data["images"].keys()))

        self.catIds = [1]

        self.preproc = preprocess
        self.transform = BaseTransform()
        self.categories = len(self.catIds) + 1

    def __len__(self):
        if self.chunk is not None:
            return len(self.ids) #// self.chunk
        else:
            return len(self.ids)

    def valid_bbox(self, bbox):
        return (bbox[2] - bbox[0]) > 0 and (bbox[3] - bbox[1]) > 0

    def get_image(self, index):
        id = self.ids[index]

        filename = self.data["images"][str(id)]["filename"]
        path = os.path.join(self.root, filename)
        image = cv2.imread(path, cv2.IMREAD_COLOR)

        bboxes = []
        for box in self.data["annotations"][str(id)]:
            bbox = [float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])]
            if self.valid_bbox(bbox):
                bbox.append(1)
                bboxes.append(bbox)
        if len(bboxes) == 0:
            bboxes = [[0., 0., 0., 0., 0.]]
        targets = np.array(bboxes)

        if self.preproc is not None:
            image, targets = self.preproc(image, targets)

        image = self.transform(image)
        targets = normalize_targets(targets, image.shape[2], image.shape[1])

        if len(targets) == 0:
            targets = np.array([[0., 0., 0., 0., 0]])

        return image, targets

    def get_chunk(self, index):
        index = index #* self.chunk
        if self.eval is True:
            frame_skip = 1
        else:
            frame_skip = random.randint(1, 15)
        key = int(self.ids[index])
        movie = self.data["images"][str(key)]["movie"]
        ids = [self.ids[index]]

        # select chunk ids
        for i in range(1, self.chunk):
            id = key + i * frame_skip
            if str(id) in self.data["images"] and self.data["images"][str(id)]["movie"] == movie:
                ids.append(id)
            else:
                break
        while len(ids) < self.chunk:
            id = key - i * frame_skip
            if self.data["images"][str(id)]["movie"] == movie:
                ids = [id] + ids

        # get images
        images = []
        for id in ids:
            filename = self.data["images"][str(id)]["filename"]
            path = os.path.join(self.root, filename)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            # cv2.imwrite("test/imgs_"+str(index)+"_"+str(id)+".jpg", img)
            images.append(img)

        # get annotations
        targets = []
        for id in ids:
            bboxes = []
            for box in self.data["annotations"][str(id)]:
                bbox = [float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])]
                if self.valid_bbox(bbox):
                    bbox.append(1)
                    bboxes.append(bbox)
            if len(bboxes) == 0:
                bboxes = [[0., 0., 0., 0., 0.]]
            bboxes = np.array(bboxes)
            targets.append(bboxes)

        if self.preproc is not None:
            images, targets = self.preproc(images, targets)
        targets = targets[2:-2]

        images = [self.transform(img) for img in images]
        targets = [normalize_targets(t, images[0].shape[2], images[0].shape[1]) for t in targets]

        for t in range(len(targets)):
            if len(targets[t]) == 0:
                targets[t] = np.array([[0., 0., 0., 0., 0]])

        images = torch.stack(images, 0)
        targets = [torch.from_numpy(t).float() for t in targets]

        return images, targets

    def __getitem__(self, index):
        if self.chunk is not None:
            return self.get_chunk(index)
        else:
            return self.get_image(index)


def collate_chunk(batch):
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
            elif isinstance(tup, list):
                targets = targets + tup
    return (imgs, targets)


def collate_chunk_s(batch):
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
                imgs = tup
            elif isinstance(tup, list):
                targets = tup
    return (imgs, targets)
