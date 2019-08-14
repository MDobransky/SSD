import argparse
import os
import time

import cv2

from ssd.lib.ssd import SSD

pics = []

def process(ssd, dir, batch_size):
    batch = []
    for p in pics:
        batch.append(p)
        if len(batch) == batch_size:
            ssd.inference(batch)
            batch = []
    if len(batch) > 0:
        ssd.inference(batch)


parser = argparse.ArgumentParser()
parser.add_argument("--batch", default=8, type=int)
parser.add_argument("--dir", default=None, type=str, help="Dataset location 300x300")
parser.add_argument("--ndir", default=None, type=str, help="Dataset location 224x224")
parser.add_argument("--weights", default=None, type=str, help="Weights folder location")

args = parser.parse_args()


#load images
for subdir, dirs, files in os.walk(args.dir):
    for image_file in files:
        image = cv2.imread(os.path.join(args.dir, image_file))
        pics.append(image)

width = 300
height = 300

FPS = []


#SSDTC
ssd = SSD(resume=args.weights + "ssdtc_c.pth",
          classes=2,
          base="ssdtc",
          width=width,
          height=height,
          phase='infer'
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("ssdtc: " + str(7500. / duration))
ssd = None


# SSD

ssd = SSD(resume=args.weights + "hhead.pth",
          classes=2,
          size=34,
          base="resnet",
          width=width,
          height=height,
          phase='infer'
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("hhead: " + str(10000. / duration))
ssd = None


ssd = SSD(resume=args.weights + "vgg_coco.pth",
          classes=81,
          size=16,
          base="vgg",
          width=width,
          height=height,
          phase='infer',
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("vgg coco: " + str(10000. / duration))
ssd = None

ssd = SSD(resume=args.weights + "resnet34_small.pth",
          classes=8,
          size=34,
          base="resnet",
          width=width,
          height=height,
          phase='infer',
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("resnet34 small: " + str(10000. / duration))
ssd = None

ssd = SSD(resume=args.weights + "resnet34_coco.pth",
          classes=81,
          size=34,
          base="resnet",
          width=width,
          height=height,
          phase='infer',
          weights_only=True
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("resnet34 coco: " + str(10000. / duration))
ssd = None

ssd = SSD(resume=args.weights + "resnet101_small.pth",
          classes=8,
          size=101,
          base="resnet",
          width=width,
          height=height,
          phase='infer',
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("resnet101 small: " + str(10000. / duration))
ssd = None

ssd = SSD(resume=args.weights + "resnet101_coco.pth",
          classes=81,
          size=101,
          base="resnet",
          width=width,
          height=height,
          phase='infer',
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("resnet101 coco: " + str(10000. / duration))
ssd = None

ssd = SSD(resume=args.weights + "xceptionA_small.pth",
          classes=8,
          size=50,
          base="xception_A",
          width=width,
          height=height,
          phase='infer',
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("xceptionA small: " + str(10000. / duration))
ssd = None

ssd = SSD(resume=args.weights + "xceptionA_coco.pth",
          classes=81,
          size=50,
          base="xception_A",
          width=width,
          height=height,
          phase='infer',
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("xceptionA coco: " + str(10000. / duration))
ssd = None

ssd = SSD(resume=args.weights + "xceptionB_small.pth",
          classes=8,
          size=50,
          base="xception_B",
          width=width,
          height=height,
          phase='infer',
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("xceptionB small: " + str(10000. / duration))
ssd = None

ssd = SSD(resume=args.weights + "xceptionC_small.pth",
          classes=8,
          size=50,
          base="xception_C",
          width=width,
          height=height,
          phase='infer',
          weights_only=True
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("xceptionC small: " + str(10000. / duration))
ssd = None

ssd = SSD(resume=args.weights + "xceptionD_small.pth",
          classes=8,
          size=50,
          base="xception_D",
          width=width,
          height=height,
          phase='infer',
          weights_only=True
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("xceptionD small: " + str(10000. / duration))
ssd = None

ssd = SSD(resume=args.weights + "xceptionE_small.pth",
          classes=8,
          size=50,
          base="xception_E",
          width=width,
          height=height,
          phase='infer',
          weights_only=True
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("xceptionE small: " + str(10000. / duration))
ssd = None

ssd = SSD(resume=args.weights + "xceptionF_small.pth",
          classes=8,
          size=50,
          base="xception_F",
          width=width,
          height=height,
          phase='infer',
          weights_only=True
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("xceptionF small: " + str(10000. / duration))
ssd = None

ssd = SSD(resume=args.weights + "xceptionG_small.pth",
          classes=8,
          size=50,
          base="xception_G",
          width=width,
          height=height,
          phase='infer',
          weights_only=True
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("xceptionG small: " + str(10000. / duration))
ssd = None

ssd = SSD(resume=args.weights + "xceptionH_small.pth",
          classes=8,
          size=50,
          base="xception_H",
          width=width,
          height=height,
          phase='infer',
          weights_only=True
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("xceptionH small: " + str(10000. / duration))
ssd = None

ssd = SSD(resume=args.weights + "xceptionJ_small.pth",
          classes=8,
          size=50,
          base="xception_J",
          width=width,
          height=height,
          phase='infer',
          weights_only=True
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("xceptionJ small: " + str(10000. / duration))
ssd = None

# resnets 50

ssd = SSD(resume=args.weights + "resnet50_small.pth",
          classes=8,
          size=50,
          base="resnet",
          width=width,
          height=height,
          phase='infer',
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("resnet50 small: " + str(10000. / duration))
ssd = None

ssd = SSD(resume=args.weights + "resnet50_25.pth",
          classes=26,
          size=50,
          base="resnet",
          width=width,
          height=height,
          phase='infer',
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("resnet50 25: " + str(10000. / duration))
ssd = None

ssd = SSD(resume=args.weights + "resnet50_43.pth",
          classes=44,
          size=50,
          base="resnet",
          width=width,
          height=height,
          phase='infer',
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("resnet50 43: " + str(10000. / duration))
ssd = None

ssd = SSD(resume=args.weights + "resnet50_61.pth",
          classes=62,
          size=50,
          base="resnet",
          width=width,
          height=height,
          phase='infer',
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("resnet50 61: " + str(10000. / duration))
ssd = None

ssd = SSD(resume=args.weights + "resnet50_coco.pth",
          classes=81,
          size=50,
          base="resnet",
          width=width,
          height=height,
          phase='infer',
          )

timestamp = time.time()
process(ssd, args.dir, args.batch)
duration = time.time() - timestamp
print("resnet50 coco: " + str(10000. / duration))
ssd = None

# nastet

width = 224
height = 224

pics = []

for subdir, dirs, files in os.walk(args.ndir):
    for image_file in files:
        image = cv2.imread(os.path.join(args.ndir, image_file))
        pics.append(image)

ssd = SSD(resume=args.weights + "nasnet_small.pth",
          classes=8,
          size=50,
          base="nasnet",
          width=width,
          height=height,
          phase='infer',
          )

timestamp = time.time()
process(ssd, args.ndir, args.batch)
duration = time.time() - timestamp
print("nasnet small: " + str(10000. / duration))
ssd = None

ssd = SSD(resume=args.weights + "nasnet_coco.pth",
          classes=81,
          size=50,
          base="nasnet",
          width=width,
          height=height,
          phase='infer',
          )

timestamp = time.time()
process(ssd, args.ndir, args.batch)
duration = time.time() - timestamp
print("nasnet small: " + str(10000. / duration))
ssd = None
