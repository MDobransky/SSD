import argparse

from torch.utils.data import DataLoader

import ssd.lib.data.hheads_detection as hhd
from ssd.lib.data.preprocess import AugmentChunk
from ssd.lib.ssdtc import SSDTC

parser = argparse.ArgumentParser()
parser.add_argument("--resume", default=None, type=str, help="Network checkpoint")
parser.add_argument("--base", default=None, type=str, help="Network checkpoint")
parser.add_argument("--export", default=None, type=str, help="Checkpoint save dir", required=True)
parser.add_argument("--epochs", default=10000, type=int, help="Number of epochs")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--batch", default=6, type=int, help="Batch size")
parser.add_argument("--save_interval", default=20, type=int)
parser.add_argument("--eval_interval", default=40, type=int)
parser.add_argument('--h', action='store_false', default=True, dest='resume_head', help='Do not resume head')
parser.add_argument("--weights_only", action='store_true', default=False, dest='weights_only', help='Resume only weights from loaded model')
parser.add_argument("--loc", default=None, type=str, help="Dataset location")

args = parser.parse_args()

height = 300
width = 300

hheadsTrainSet = hhd.HHeadsDetection(root=args.loc + "JPEGImages/",
                                     annFile=args.loc + "annotations/train.json",
                                     chunk=5,
                                     preprocess=AugmentChunk(size=(height, width))
                                     )

trainLoader = DataLoader(hheadsTrainSet, args.batch, shuffle=True, num_workers=1, pin_memory=True, collate_fn=hhd.collate_chunk)

SSD = SSDTC(resume=args.resume,
            base=args.base,
            weights_only=args.weights_only,
            classes=2,
            trainLoader=trainLoader,
            save_dir=args.export,
            learning_rate=args.lr,
            batch_size=args.batch,
            save_interval=args.save_interval,
            eval_interval=args.eval_interval,
            height=height,
            width=width)

SSD.train_model(args.epochs)
