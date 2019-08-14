import os
import sys

import torch.backends.cudnn as cudnn

from ssd.lib.models import xception_A_ssd, xception_B_ssd, xception_C_ssd, xception_D_ssd, xception_F_ssd, \
    xception_E_ssd, xception_H_ssd, xception_G_ssd, xception_J_ssd, nasnet_ssd, vgg_ssd, resnet_ssdtc, resnet_ssd

from ssd.lib.data.preprocess import BaseTransform
from ssd.lib.detection import Detect
from ssd.lib.eval_utils import *
from ssd.lib.multibox_loss import MultiBoxLoss


class SSD(object):

    def __init__(self, resume=None,
                 resume_head=True,
                 lock_base=False,
                 lock_extras=False,
                 size=18,
                 base="resnet",
                 weights_only=False,
                 classes=100,
                 evalLoader=None,
                 trainLoader=None,
                 phase='train',
                 save_dir=None,
                 learning_rate=0.001,
                 loss_balance=1,
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

        self.loss_balance = loss_balance
        self.lr = learning_rate

        self.size = size
        self.base = base

        self.model = self.get_net()(classes, (self.height, self.width), phase).cuda()
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print("Total parameters: ", pytorch_total_params)

        self.model = torch.nn.DataParallel(self.model)

        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate,
                                         momentum=0.9, weight_decay=5e-4)
        self.loss = MultiBoxLoss(self.model.module.priors.boxes.cuda(), classes)

        if resume == "imgnet":
            self.load_imgnet()
            print("Loading ImageNet weights")
        elif resume is not None:
            self.load_model(resume, resume_head, weights_only)

        if lock_base is True:
            print("Freezing base layers")
            for layer in self.model.module.base:
                for parameter in layer.parameters():
                    parameter.requires_grad = False

        if lock_extras is True:
            print("Freezing extra layers")
            for layer in self.model.module.extras:
                for parameter in layer.parameters():
                    parameter.requires_grad = False

        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Trainable parameters: ", pytorch_total_params)

        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    def unnormalize_loc(self, loc):
        return loc * self.norm_matrix

    def get_net(self):
        if self.base == "resnet":
            if self.size == 18:
                return resnet_ssd.resnet18
            elif self.size == 34:
                return resnet_ssd.resnet34
            elif self.size == 50:
                return resnet_ssd.resnet50
            elif self.size == 101:
                return resnet_ssd.resnet101
            elif self.size == 152:
                return resnet_ssd.resnet152
        elif self.base == "vgg":
            return vgg_ssd.vgg16
        elif self.base == "xception_A":
            return xception_A_ssd.xception
        elif self.base == "xception_B":
            return xception_B_ssd.xception_B
        elif self.base == "xception_C":
            return xception_C_ssd.xception_C
        elif self.base == "xception_D":
            return xception_D_ssd.xception_D
        elif self.base == "xception_G":
            return xception_G_ssd.xception_G
        elif self.base == "xception_E":
            return xception_E_ssd.xception_E
        elif self.base == "xception_F":
            return xception_F_ssd.xception_F
        elif self.base == "xception_H":
            return xception_H_ssd.xception_H
        elif self.base == "xception_J":
            return xception_J_ssd.xception_J
        elif self.base == "nasnet":
            return nasnet_ssd.nasnetmobile
        elif self.base == "ssdtc":
            return resnet_ssdtc.resnet34_ssdtc

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

    def load_model(self, resume, resume_head, weights_only):
        model_dict = self.model.state_dict()

        state = torch.load(resume)
        if len(state) == 3:
            pretrained_dict = state['state_dict']
            if not weights_only:
                self.optimizer.load_state_dict(state['optimizer'])
                self.start_epoch = state['epoch'] + 1
        else:
            pretrained_dict = state

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if resume_head is not True:
            print('Ignoing saved states for multibox head')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in self.model.head}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def load_imgnet(self):
        import torchvision.models as models
        model_dict = self.model.state_dict()

        if self.base == "resnet":
            pretrained = models.resnet18(pretrained=True)
            if self.size == 34:
                pretrained = models.resnet34(pretrained=True)
            elif self.size == 50:
                pretrained = models.resnet50(pretrained=True)
            elif self.size == 101:
                pretrained = models.resnet101(pretrained=True)
            elif self.size == 152:
                pretrained = models.resnet152(pretrained=True)
            pretrained_dict = pretrained.state_dict()

        elif self.base == "vgg":
            pretrained = models.vgg16(pretrained=True)
            pretrained_dict = pretrained.state_dict()

        elif self.base == "xception":
            # from https://github.com/Cadene/pretrained-models.pytorch
            # model url http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth
            pretrained_dict = torch.load("weights/pretrained/xception.pth")


        elif self.base == "nasnet":
            # from https://github.com/Cadene/pretrained-models.pytorch
            # model url http://data.lip6.fr/cadene/pretrainedmodels/nasnetamobile-7e03cead.pth
            pretrained_dict = torch.load("weights/pretrained/nasnetamobile.pth")

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

            images = images.cuda()
            targets = [t.cuda() for t in targets]

            predictions = model(images)

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

    def eval_epoch(self, epoch=0):
        with torch.no_grad():
            model = self.model.eval()

            cumulative_class_loss = 0
            cumulative_loc_loss = 0
            batches = len(self.evalLoader)

            label = [list() for _ in range(model.module.num_classes)]
            gt_label = [list() for _ in range(model.module.num_classes)]
            score = [list() for _ in range(model.module.num_classes)]
            size = [list() for _ in range(model.module.num_classes)]
            npos = [0] * model.module.num_classes

            for batch_index, data in enumerate(self.evalLoader):
                images, targets = data

                images = images.cuda()
                targets = [t.cuda() for t in targets]

                predictions = model(images)

                loc_loss, class_loss = self.loss(predictions, targets)

                cumulative_class_loss += class_loss.item()
                cumulative_loc_loss += loc_loss.item()

                sys.stdout.write('\r==>Eval: batch %d/%d, class loss: %f, loc loss: %f' % (
                    batch_index, batches, class_loss.item(), loc_loss.item()))
                sys.stdout.flush()

                predictions = (predictions[0], model.module.softmax(predictions[1].view(-1, model.module.num_classes)))
                detector = Detect(self.model.module.priors.boxes, self.classes)
                detections = detector.forward(predictions)

                label, score, npos, gt_label = true_false_positive(detections, targets, label, score, npos, gt_label)
                size = cal_size(detections, targets, size)

            prec, rec, ap = precision(label, score, npos)

            sys.stdout.write(
                '\r==>Eval: || Done, class loss: %f, loc loss: %f, mAP %f\n' % (
                    cumulative_class_loss / batches, cumulative_loc_loss / batches, ap))
            sys.stdout.flush()

    def inference(self, images, conf=0.2):
        with torch.no_grad():
            model = self.model.eval()
            imgs = []

            for i in images:
                imgs.append(self.transform(i))

            imgs = torch.stack(imgs).cuda()

            predictions = model(imgs)
            detector = Detect(self.model.module.priors.boxes.cuda(), classes=self.classes, conf_threshold=conf)
            detections = detector.forward(predictions)

            la, cf, loc = self.predict(detections)
            return (loc, la, cf)

    def export_detections(self, names):
        with torch.no_grad():
            model = self.model.eval()
            batches = len(self.evalLoader)

            for batch_index, data in enumerate(self.evalLoader):
                images, targets = data

                (width, height) = (images.shape[2], images.shape[3])

                images = images.cuda()

                predictions = model(images)

                predictions = (predictions[0], model.module.softmax(predictions[1].view(-1, model.module.num_classes)))
                detector = Detect(self.model.module.priors.boxes, self.classes)
                detections = detector.forward(predictions)
                la, cf, loc = self.predict(detections)
                loc, la, cf = loc[0], la[0], cf[0]

                gt = open("groundtruths/" + str(10000000 + batch_index) + ".txt", "w")
                dt = open("detections/" + str(10000000 + batch_index) + ".txt", "w")

                targets = targets[0].numpy()
                targets[:, :4] = targets[:, :4] * np.array([width, height, width, height])
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
