import random

import cv2
import numpy as np
import torch.distributions
from torchvision import transforms

random.seed(42)


class Augment(object):
    def __init__(self, size=(300, 300), eval=False):
        self.h, self.w = size
        self.eval = eval

    def crop(self, image, boxes, labels, new_h0=None, new_w0=None):
        height, width, _ = image.shape

        if new_h0 is None or new_w0 is None:
            new_h0, new_w0 = 0, 0
            if height > self.h:
                new_h0 = random.randint(0, height - self.h)
            if width > self.w:
                new_w0 = random.randint(0, width - self.w)

        roi = [new_w0, new_h0, new_w0 + self.w, new_h0 + self.h]

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        # filter valid boxes
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask]
        labels_t = labels[mask]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

        return image_t, boxes_t, labels_t

    def _find_scale(self, height, width):
        if float(height) / self.h < float(width) / self.w:
            new_h = random.randint(self.h + 1, int(self.h * 1.1))
            if self.eval is True:
                new_h = self.h + 1
            scale = new_h / height
            new_w = int(width * scale)
        else:
            new_w = random.randint(self.w + 1, int(self.w * 1.1))
            if self.eval is True:
                new_w = self.w + 1
            scale = new_w / width
            new_h = int(height * scale)
        return new_w, new_h, scale

    def scale(self, image, boxes, new_w=None, new_h=None, scale=None):
        height, width, _ = image.shape
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = random.choice(interp_methods)
        if self.eval is True:
            interp_method = cv2.INTER_LINEAR

        if new_w is None or new_h is None or scale is None:
            new_w, new_h, scale = self._find_scale(height, width)

        image = cv2.resize(image, (new_w, new_h), interpolation=interp_method)
        boxes = boxes * scale
        return image, boxes

    def mirror(self, image, boxes):
        _, width, _ = image.shape
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes

    def distort(self, image, dists=None, a=None, b=None, c=None, d=None, e=None):
        def _convert(image, alpha=1, beta=0):
            tmp = image.astype(float) * alpha + beta
            tmp[tmp < 0] = 0
            tmp[tmp > 255] = 255
            image[:] = tmp

        if dists is None or a is None or b is None or c is None or d is None or e is None:
            dists = [random.randrange(2) for _ in range(5)]
            a = random.uniform(0.5, 1.5)
            b = random.uniform(-32, 32)
            c = random.randint(-18, 18)
            d = random.uniform(0.2, 1.7)
            e = random.uniform(0.5, 1.5)

        if dists[0]:
            _convert(image, beta=b)

        if dists[1]:
            _convert(image, alpha=a)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if dists[2]:
            tmp = image[:, :, 0].astype(int) + c
            tmp %= 180
            image[:, :, 0] = tmp

        if dists[3]:
            _convert(image[:, :, 1], d)

        if dists[4]:
            _convert(image[:, :, 2], e)

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image

    def noise(self, image, type=None):
        noises = ["gauss", "s&p", "speckle", ""]
        if type is None:
            noise_type = random.choice(noises)
        else:
            noise_type = noises[type]

        if noise_type == "gauss":
            row, col, ch = image.shape
            mean = 0
            sigma = 30

            norm = torch.distributions.Normal(mean, sigma)
            gauss = norm.sample((row, col, ch)).numpy()
            gauss = gauss.reshape(row, col, ch)
            noisy = image + gauss
            return np.uint8(np.clip(noisy, 0, 255))

        elif noise_type == "s&p":
            s_vs_p = 0.5
            amount = 0.01
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [torch.randint(0, i - 1, (int(num_salt),)).numpy().astype(int) for i in image.shape]
            out[tuple(coords)] = 255

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [torch.randint(0, i - 1, (int(num_pepper),)).numpy().astype(int) for i in image.shape]
            out[tuple(coords)] = 0
            return out

        elif noise_type == "speckle":
            row, col, ch = image.shape
            norm = torch.distributions.Uniform(0, 0.8)
            gauss = norm.sample((row, col, ch)).numpy()
            gauss = gauss.reshape(row, col, ch)
            noisy = image + image * gauss
            return np.uint8(np.clip(noisy, 0, 255))
        else:
            return image

    def __call__(self, image, targets=np.zeros((1, 5))):
        boxes = targets[:, :-1]
        labels = targets[:, -1]
        image = np.uint8(image)

        image, boxes = self.scale(image, boxes)

        image, boxes, labels = self.crop(image, boxes, labels)

        if random.randrange(2) == 1:
            image, boxes = self.mirror(image, boxes)

        image = self.distort(image)

        image = self.noise(image)

        image = np.uint8(image)

        targets = np.concatenate((boxes, np.expand_dims(labels, axis=1)), axis=1)
        return image, targets


class Crop(Augment):
    def __init__(self, size=(300, 300), eval=False):
        super(Crop, self).__init__()
        self.h, self.w = size
        self.eval = eval

    def __call__(self, image, targets=np.zeros((1, 5))):
        boxes = targets[:, :-1]
        labels = targets[:, -1]

        image = np.uint8(image)
        image, boxes = self.scale(image, boxes)

        if self.eval is True:
            image, boxes, labels = self.crop(image, boxes, labels, new_h0=0, new_w0=0)
        else:
            image, boxes, labels = self.crop(image, boxes, labels)
        targets = np.concatenate((boxes, np.expand_dims(labels, axis=1)), axis=1)
        return image, targets


# augment chunk of images with consistent parameters
class AugmentChunk(Augment):
    def __init__(self, size=(300, 300), eval=False):
        super(AugmentChunk, self).__init__()
        self.h, self.w = size
        self.eval = eval

    def _find_crop(self, height, width):
        new_h0, new_w0 = 0, 0
        if height > self.h:
            new_h0 = random.randint(0, height - self.h)
        if width > self.w:
            new_w0 = random.randint(0, width - self.w)
        return new_h0, new_w0

    def __call__(self, images, targets=None):
        boxes = []
        labels = []
        for target in targets:
            boxes.append(target[:, :-1])
            labels.append(target[:, -1])

        images = [np.uint8(image) for image in images]

        mirror = random.randrange(2)

        new_w, new_h, scale = self._find_scale(images[0].shape[0], images[0].shape[1])
        new_h0, new_w0 = self._find_crop(new_h, new_w)

        # distort
        dists = [random.randrange(2) for _ in range(5)]
        a = random.uniform(0.5, 1.5)
        b = random.uniform(-32, 32)
        c = random.randint(-18, 18)
        d = random.uniform(0.2, 1.7)
        e = random.uniform(0.5, 1.5)

        # noise
        type = random.randint(0, 3)

        for i in range(len(images)):
            image = images[i]
            bboxes = boxes[i]
            # scale
            image, bboxes = self.scale(image, bboxes, new_w, new_h, scale)
            # crop
            image, bboxes, labels[i] = self.crop(image, bboxes, labels[i], new_h0, new_w0)
            # mirror
            if mirror == 1:
                image, bboxes = self.mirror(image, bboxes)
            # distort
            image = self.distort(image, dists, a, b, c, d, e)
            # noise
            image = self.noise(image, type)

            image = np.uint8(image)

            images[i] = image
            boxes[i] = bboxes

            targets[i] = np.concatenate((boxes[i], np.expand_dims(labels[i], axis=1)), axis=1)
            # cv2.imwrite("test/img_" + str(index) + "_" + str(i) + ".jpg", image)

        return images, targets


class CropChunk(AugmentChunk):
    def __init__(self, size=(300, 300), eval=False):
        super(CropChunk, self).__init__()
        self.h, self.w = size
        self.eval = eval

    def __call__(self, images, targets=None):
        boxes = []
        labels = []
        for target in targets:
            boxes.append(target[:, :-1])
            labels.append(target[:, -1])

        images = [np.uint8(image) for image in images]

        new_w, new_h, scale = self._find_scale(images[0].shape[0], images[0].shape[1])
        if self.eval is True:
            new_h0, new_w0 = 0, 0
        else:
            new_h0, new_w0 = self._find_crop(new_h, new_w)

        for i in range(len(images)):
            image = images[i]
            bboxes = boxes[i]
            # scale
            image, bboxes = self.scale(image, bboxes, new_w, new_h, scale)
            # crop
            image, bboxes, labels[i] = self.crop(image, bboxes, labels[i], new_h0, new_w0)

            image = np.uint8(image)

            images[i] = image
            boxes[i] = bboxes

            targets[i] = np.concatenate((boxes[i], np.expand_dims(labels[i], axis=1)), axis=1)

        return images, targets


class BaseTransform(object):
    def __init__(self):
        self.tranform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        return self.tranform(image)


def normalize_targets(targets, width, height):
    targets[:, :4] = targets[:, :4] / np.array([width, height, width, height])
    return targets
