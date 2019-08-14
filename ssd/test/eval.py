#exports data to "detections" and "groundtruhts" folders
#use https://github.com/rafaelpadilla/Object-Detection-Metrics for evaluation

import argparse

from torch.utils.data import DataLoader

import ssd.lib.data.coco_detection as cd
import ssd.lib.data.hheads_detection as hhd
from ssd.lib.data.preprocess import Crop, CropChunk
from ssd.lib.ssd import SSD

parser = argparse.ArgumentParser()
parser.add_argument("--resume", default=None, type=str, help="Network checkpoint")
parser.add_argument("--batch", default=1, type=int)
parser.add_argument("--loc", default=None, type=str, help="Dataset location")
parser.add_argument("--net", default=None, type=str, help="network")
parser.add_argument("--size", default=None, type=int, help="resnet size")

args = parser.parse_args()

print("Preparing validation dataset")
width = 300
height = 300

import json

with open('ssd/lib/data/coco_labels.json') as f:
    coco = json.load(f)["categories"]
coco = [x["name"] for x in coco]
small = ["person", "bicycle", "car", "motorcycle", "bus", "train", "truck"]
head = ["head"]

# surveillance dataset for SSD
cocoEvalSet = cd.CocoDetection(root=args.loc + "images/",
                               annFile=args.loc + "annotations/instances_val2014.json",
                               preprocess=Crop(size=(height, width)),
                               classes=small)

evalLoader = DataLoader(cocoEvalSet, 1, shuffle=False, num_workers=1, pin_memory=True, collate_fn=cd.collate)

# HollywoodHead dataset for SSDTC
# hheadsTrainSet =  hhd.HHeadsDetection(root=args.loc + "JPEGImages/",
#                                       annFile=args.loc+"annotations/val.json",
#                                       preprocess=CropChunk(size=(height, width), eval=True),
#                                       chunk=5,
#                                       eval=True
#                                       )
# evalLoader = DataLoader(hheadsTrainSet, 1, shuffle=False, num_workers=1, pin_memory=True, collate_fn=hhd.collate_chunk_s)


# HollywoodHeads dataset for SSD
# hheadsTrainSet =  hhd.HHeadsDetection(root=args.loc + "JPEGImages/",
#                                      annFile=args.loc+"annotations/val.json",
#                                      preprocess=Crop(size=(height, width))
#                                      )
#
# evalLoader = DataLoader(hheadsTrainSet, 1, shuffle=False, num_workers=1, pin_memory=True, collate_fn=cd.collate)

SSD = SSD(resume=args.resume,
          classes=8,
          size=args.size,
          base=args.net,
          evalLoader=evalLoader,
          batch_size=args.batch,
          width=width,
          height=height,
          weights_only=True,
          save_dir="saves"
          )

SSD.export_detections(small)
