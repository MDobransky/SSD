# two part ssd detector with separate feature extractor and separate temporal detector
# used primarily for training

import os
import sys

import torch.backends.cudnn as cudnn

from ssd.lib.data.preprocess import BaseTransform
from ssd.lib.detection import Detect
from ssd.lib.eval_utils import *
from ssd.lib.multibox_loss import MultiBoxLoss
from ssd.lib.models.ssdtc_base import resnet34_features
from ssd.lib.models.ssdtc_head import SSDTC_det


class SSDTC(object):

    def __init__(self, resume=None,
                 base=None,
                 weights_only=False,
                 classes=100,
                 evalLoader=None,
                 trainLoader=None,
                 phase='train',
                 save_dir=None,
                 learning_rate=0.001,
                 batch_size=64,
                 save_interval=10,
                 eval_interval=10,
                 width=224,
                 height=224):

        self.export_dir = save_dir
        self.start_epoch = 0
        self.save_interval = save_interval
        self.eval_interval = eval_interval

        self.batch_size = batch_size
        self.classes = classes
        self.width = width
        self.height = height
        self.norm_matrix = torch.tensor(np.array([self.width, self.height, self.width, self.height]),
                                        dtype=torch.float).cuda()
        self.transform = BaseTransform()

        self.evalLoader = evalLoader
        self.trainLoader = trainLoader

        self.loss_balance = 1
        self.lr = learning_rate

        self.extractor = resnet34_features().cuda().eval()
        self.model = SSDTC_det(classes, (self.height, self.width), phase).cuda()
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print("Total parameters: ", pytorch_total_params)

        self.extractor = torch.nn.DataParallel(self.extractor)
        self.model = torch.nn.DataParallel(self.model)
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate,
                                         momentum=0.9, weight_decay=5e-4)
        self.loss = MultiBoxLoss(self.model.module.priors.boxes.cuda(), classes)

        self.load_model(base, resume, weights_only)

        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Trainable parameters: ", pytorch_total_params)

        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    def unnormalize_loc(self, loc):
        return loc * self.norm_matrix

    def predict(self, detections):
        with torch.no_grad():
            labels, scores, coords = [[list() for _ in range(detections.size(0))] for _ in range(3)]
            for batch_element in range(detections.size(0)):
                for classes in range(detections.size(1)):
                    num = 0
                    while num < detections.size(2) and detections[batch_element, classes, num, 0] > 0:
                        scores[batch_element].append(detections[batch_element, classes, num, 0])
                        labels[batch_element].append(classes - 1)
                        coords[batch_element].append(self.unnormalize_loc(detections[batch_element, classes, num, 1:]))
                        num += 1
        return labels, scores, coords

    def save_model(self, epoch):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, os.path.join(self.export_dir, "epoch" + str(epoch) + ".pth"))

    def load_model(self, base, resume, weights_only):
        # load feature extractor (resnet34 with extras)
        base_dict = self.extractor.state_dict()
        state = torch.load(base)
        if len(state) == 3:
            pretrained_dict = state['state_dict']
        else:
            pretrained_dict = state
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in base_dict}
        base_dict.update(pretrained_dict)
        self.extractor.load_state_dict(base_dict)

        model_dict = self.model.state_dict()

        # load detection head
        if resume is not None:
            state = torch.load(resume)
            if len(state) == 3:
                pretrained_dict = state['state_dict']
                if not weights_only:
                    self.optimizer.load_state_dict(state['optimizer'])
                    self.start_epoch = state['epoch'] + 1
            else:
                pretrained_dict = state

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

    def train_epoch(self, epoch, epochs, iteration, step_index):
        batches = len(self.trainLoader)

        model = self.model.train()
        cumulative_class_loss = 0
        cumulative_loc_loss = 0

        for batch_index, data in enumerate(self.trainLoader):
            images, targets = data

            features = []

            with torch.no_grad():
                for imgs in images:
                    imgs = imgs.cuda()
                    features.append(self.extractor(imgs))

            tf = []
            for f in range(6):
                fs = []
                for b in range(len(features)):
                    fs.append(features[b][f])
                tf.append(torch.stack(fs))

            targets = [t.cuda() for t in targets]

            predictions = model(tf)

            self.optimizer.zero_grad()

            loc_loss, class_loss = self.loss(predictions, targets)

            loss = class_loss + self.loss_balance * loc_loss
            cumulative_class_loss += class_loss.item()
            cumulative_loc_loss += loc_loss.item()

            loss.backward()
            self.optimizer.step()

            sys.stdout.write('\r==>Train: || epoch %d/%d, batch %d/%d, class loss: %f, loc loss: %f' % (
                epoch, epochs, batch_index, batches, class_loss.item(), loc_loss.item()))
            sys.stdout.flush()

            iteration += 1

        if iteration in (280000, 360000, 400000):
            step_index += 1
            lr = self.lr * (0.1 ** (step_index))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        sys.stdout.write(
            '\r==>Train: || Done epoch %d/%d, class loss: %f, loc loss: %f                         \n' % (
                epoch, epochs, cumulative_class_loss / batches, cumulative_loc_loss / batches))
        sys.stdout.flush()

    def train_model(self, epochs):
        iteration = (self.start_epoch - 1) * len(self.trainLoader)
        step_index = 0
        if iteration > 280000:
            step_index += 1
        if iteration > 360000:
            step_index += 1
        if iteration > 400000:
            step_index += 1
        lr = self.lr * (0.1 ** (step_index))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        for i in range(self.start_epoch, epochs):
            self.train_epoch(i, epochs, iteration, step_index)
            if i > 0 and i % self.save_interval == 0:
                self.save_model(i)
            if i > 0 and i % self.eval_interval == 0:
                self.eval_epoch(epoch=i)

    def export_detections(self, names):
        with torch.no_grad():
            model = self.model.eval()
            batches = len(self.evalLoader)

            for batch_index, data in enumerate(self.evalLoader):
                images, targets = data

                features = []

                for imgs in images:
                    imgs = imgs.cuda()
                    features.append(self.extractor(imgs))

                tf = []
                for f in range(6):
                    fs = []
                    for b in range(len(features)):
                        fs.append(features[b][f])
                    tf.append(torch.stack(fs))

                predictions = model(tf)

                predictions = (predictions[0], model.module.softmax(predictions[1].view(-1, model.module.num_classes)))
                detector = Detect(self.model.module.priors.boxes, self.classes)
                detections = detector.forward(predictions)
                la, cf, loc = self.predict(detections)
                loc, la, cf = loc[0], la[0], cf[0]

                gt = open("groundtruths/" + str(10000000 + batch_index) + ".txt", "w")
                dt = open("detections/" + str(10000000 + batch_index) + ".txt", "w")

                targets = targets[0].numpy()
                targets[:, :4] = targets[:, :4] * np.array([300, 300, 300, 300])
                for t in targets:
                    if t[4] != 0:
                        gt.write(
                            names[int(t[4] - 1)].replace(" ", "_") + " " + str(t[0]) + " " + str(t[1]) + " " + str(
                                t[2]) + " " + str(t[3]) + "\n")

                for i in range(0, len(la)):
                    c = la[i]
                    l = loc[i].cpu().numpy()
                    con = cf[i].cpu().numpy()
                    dt.write(names[int(c)].replace(" ", "_") + " " + str(con) + " " + str(l[0]) + " " + str(
                        l[1]) + " " + str(l[2]) + " " + str(
                        l[3]) + "\n")

                sys.stdout.write('\r==>Eval: batch %d/%d' % (batch_index, batches))
                sys.stdout.flush()
