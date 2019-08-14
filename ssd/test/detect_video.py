# generated json with detections for video
# format: {frame_number:[{"label": x, "conf": c, "left": l, "top": t, "right": r, "bottom": b}, {"label": x, ....}], frame_number : [{},{}...],...}


import argparse
import json
import sys
import time

import cv2

from ssd.lib.ssd import SSD

parser = argparse.ArgumentParser()
parser.add_argument("--video", default=None, type=str)
parser.add_argument("--batch", default=1, type=int)
parser.add_argument("--weights", default=None, type=str, help="Network checkpoint")
parser.add_argument("--net", default=None, type=str)
parser.add_argument("--size", default=None, type=int, help="ResNet size")

args = parser.parse_args()

with open('ssd/lib/data/coco_labels.json') as f:
    coco = json.load(f)["categories"]
coco = [x["name"] for x in coco]
small = ["PERSON", "BICYCLE", "CAR", "MOTOCYCLE", "BUS", "TRAIN", "TRUCK"]
head = ["Head"]

# set dataset
dataset = small

video = cv2.VideoCapture(args.video)

w = int(video.get(3))
h = int(video.get(4))

# set classes
SSD = SSD(args.weights,
          classes=len(dataset) + 1,
          phase='infer',
          width=w,
          height=h,
          size=args.size,
          base=args.net)

frame = 0
predictions = {}
timestamp = time.time()
batch = []

# for ssdtc
predictions[0] = []
predictions[1] = []

while video.isOpened():
    # leave last 4 frames if ssdtc
    batch = []# batch[-4:]
    for b in range(0, args.batch):
        success, image = video.read()
        if success:
            batch.append(image)
        else:
            break

    #limit on 5 for ssdtc
    if len(batch) < 1:
        break

    detections = SSD.inference(batch, conf=0.4)
    if detections:
        location, labels, confidence = detections
        # for each image
        for b in range(0, len(labels)):
            loc = location[b]
            lab = labels[b]
            conf = confidence[b]
            pred = []
            # for each box
            for i in range(0, len(lab)):
                lc = loc[i].cpu()
                if not lc.equal(lc):
                    continue
                lc = lc.numpy()
                cf = int(conf[i] * 100) / 100
                pred.append({"label": dataset[lab[i]], "conf": cf, "left": int(max(lc[0], 0)),
                             "top": int(max(lc[1], 0)), "right": int(min(lc[2], w)), "bottom": int(min(lc[3], h))})
            predictions[frame] = pred
            frame += 1
    sys.stdout.write('\rFPS: %d' % (frame / (time.time() - timestamp)))
    sys.stdout.flush()

video.release()

with open('detections.json', 'w') as outfile:
    json.dump(predictions, outfile)
