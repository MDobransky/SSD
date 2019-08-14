#evaluation using two stage extractor-detector approach

#exports data to "detections" and "groundtruhts" folders
#use https://github.com/rafaelpadilla/Object-Detection-Metrics for evaluation

import argparse

from torch.utils.data import DataLoader

import ssd.lib.data.hheads_detection as hhd
from ssd.lib.data.preprocess import CropChunk
from ssd.lib.ssdtc import SSDTC

parser = argparse.ArgumentParser()
parser.add_argument("--resume", default=None, type=str, help="Network checkpoint")
parser.add_argument("--batch", default=1, type=int)
parser.add_argument("--loc", default=None, type=str, help="Dataset location")
parser.add_argument("--base", default=None, type=str, help="base weights")

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

hheadsTrainSet = hhd.HHeadsDetection(root=args.loc + "JPEGImages/",
                                     annFile=args.loc + "annotations/val.json",
                                     preprocess=CropChunk(size=(height, width), eval=True),
                                     chunk=5,
                                     eval=True
                                     )
evalLoader = DataLoader(hheadsTrainSet, 1, shuffle=False, num_workers=1, pin_memory=True, collate_fn=hhd.collate_chunk)

SSD = SSDTC(resume=args.resume,
            classes=2,
            base=args.base,
            evalLoader=evalLoader,
            batch_size=args.batch,
            width=width,
            height=height,
            weights_only=True
            )

SSD.export_detections(head)
