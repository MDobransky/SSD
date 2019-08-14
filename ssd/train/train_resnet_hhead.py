import argparse

from torch.utils.data import DataLoader

import ssd.lib.data.coco_detection as cd
import ssd.lib.data.hheads_detection as hhd
from ssd.lib.data.preprocess import Augment
from ssd.lib.ssd import SSD

parser = argparse.ArgumentParser()
parser.add_argument("--resume", default=None, type=str, help="Network checkpoint")
parser.add_argument("--export", default=None, type=str, help="Checkpoint save dir", required=True)
parser.add_argument("--epochs", default=10000, type=int, help="Number of epochs")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--loss_balance", default=1.0, type=float)
parser.add_argument("--batch", default=32, type=int, help="Batch size")
parser.add_argument("--save_interval", default=20, type=int)
parser.add_argument("--eval_interval", default=40, type=int)
parser.add_argument('--h', action='store_false', default=True, dest='resume_head', help='Do not resume head')
parser.add_argument('--l', action='store_true', default=False, dest='lock_base', help='Lock base')
parser.add_argument("--weights_only", action='store_true', default=False, dest='weights_only', help='Resume only weights from loaded model')
parser.add_argument("--loc", default=None, type=str, help="Dataset location")
parser.add_argument("--size", default=34, type=int, help="resnet size")

args = parser.parse_args()

height = 300
width = 300

classes = ["person", "bicycle", "car", "motorcycle", "bus", "train", "truck"]

hheadsTrainSet = hhd.HHeadsDetection(root=args.loc + "JPEGImages/",
                                     annFile=args.loc + "annotations/train.json",
                                     preprocess=Augment(size=(height, width))
                                     )

trainLoader = DataLoader(hheadsTrainSet, args.batch, shuffle=False, num_workers=1, pin_memory=True, collate_fn=cd.collate)

SSD = SSD(resume=args.resume,
          resume_head=args.resume_head,
          lock_base=True,
          lock_extras=True,
          weights_only=args.weights_only,
          classes=2,
          base="resnet",
          size=34,
          # evalLoader=cocoEvalLoader,
          trainLoader=trainLoader,
          loss_balance=args.loss_balance,
          save_dir=args.export,
          learning_rate=args.lr,
          batch_size=args.batch,
          save_interval=args.save_interval,
          eval_interval=args.eval_interval,
          height=height,
          width=width)

SSD.train_model(args.epochs)
