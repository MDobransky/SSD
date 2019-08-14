#draws detections on images from given folder
#uses batch 1
#requires manual configuration in this file

import argparse
import json
import os

from ssd.lib.data.preprocess import *
from ssd.lib.ssd import SSD

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default=None, type=str, help="Input image directory", required=True)
parser.add_argument("--target_dir", default=None, type=str, help="Output image directory", required=True)
parser.add_argument("--resume", default=None, type=str, help="checkpoint")
args = parser.parse_args()

with open('ssd/lib/data/coco_labels.json') as f:
    coco = json.load(f)["categories"]
coco =  [x["name"] for x in coco]

shapes = ["circle", "square", "triangle", "rectangle", "ellipse"]

small = ["PERSON", "BICYCLE", "CAR", "MOTOCYCLE", "BUS", "TRAIN", "TRUCK"]

#set dataset
dataset = small

w, h = 300, 300

#set network type and classes number
SSD = SSD(args.resume,
          classes=len(dataset) + 1,
          phase='infer',
          width=w,
          height=h,
          size=34,
          base="xception_H")

augment = Crop(size=(h, w))
for subdir, dirs, files in os.walk(args.dir):
    for image_file in files:
        image = cv2.imread(os.path.join(args.dir, image_file))
        image, _ = augment(image)

        detections = SSD.inference([image], conf=0.2)
        t = ""

        if detections:
            loc, labels, conf = detections
            loc, labels, conf = loc[0], labels[0], conf[0]
            for i in range(0, len(labels)):
                lc = loc[i]
                if not lc.equal(lc):
                    continue
                item = labels[i]
                cv2.putText(image, dataset[item] + " " + str(int(conf[i] * 100) / 100), (max(lc[0]+5, 5), max(lc[1] + 15, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)
                cv2.rectangle(image, (max(lc[0], 0), max(lc[1], 0)), (min(lc[2], w), min(lc[3], h)), (0, 0, 255), 1)

        cv2.imwrite(os.path.join(args.target_dir, image_file), image)
