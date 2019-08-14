#convers HHeas dataset annotations to json format

import json
import xml.etree.ElementTree

set = 'Splits/val.txt'

with open(set) as f:
    lines = f.read().splitlines()

lines = ['Annotations/' + l + '.xml' for l in lines]

images = {}
annotations = {}

id = 0

for file in lines:
    bboxes = []

    root = xml.etree.ElementTree.parse(file).getroot()
    for child in root.iter('filename'):
        filename = child.text

    movie = filename.split('_')[1]
    frame = int(filename.split('_')[2].split('.')[0])

    images[id] = {"filename": filename, "movie": movie, "frame": frame}

    for child in root.iter('object'):
        for bndbox in child.iter('bndbox'):
            bboxes.append({"x1": bndbox.findtext('xmin'),
                           "y1": bndbox.findtext('ymin'),
                           "x2": bndbox.findtext('xmax'),
                           "y2": bndbox.findtext('ymax')
                           })
    annotations[id] = bboxes
    id += 1

data = {"images": images, "annotations": annotations}

with open('val.json', 'w') as outfile:
    json.dump(data, outfile)
