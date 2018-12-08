import glob
import xml.etree.ElementTree as ET

import numpy as np

from kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = "Annotations" #copy this file and kmean.py to some folder of Annotations (VOC xml-like)
CLUSTERS = 9 #cluster number
input_size = 416 #input of network (squared)

def load_dataset(path):
	dataset = []
	for xml_file in glob.glob("{}/*xml".format(path)):
		tree = ET.parse(xml_file)

		height = int(tree.findtext("./size/height"))
		width = int(tree.findtext("./size/width"))

		for obj in tree.iter("object"):
			xmin = int(obj.findtext("bndbox/xmin")) * 1.0 / width
			ymin = int(obj.findtext("bndbox/ymin")) * 1.0 / height
			xmax = int(obj.findtext("bndbox/xmax")) * 1.0 / width
			ymax = int(obj.findtext("bndbox/ymax")) * 1.0 / height

			dataset.append([xmax - xmin, ymax - ymin])

	return np.array(dataset)


data = load_dataset(ANNOTATIONS_PATH)
out = kmeans(data, k=CLUSTERS)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))

print("Boxes:\n {}".format(out))

bbox_list = []
for bbox in out:
    w,h = bbox
    bbox_list.append( [w * h, int(w * input_size), int(h * input_size)] )

bbox_list = sorted(bbox_list, key = lambda x:x[0], reverse=False)


lines = []
for bbox in bbox_list:
    s,w,h = bbox
    lines.append('{},{}'.format(w,h))
lines = ','.join(lines)
with open('anchors.txt','wb') as f:
    f.write(lines)
print("anchor:{}\n".format(lines))





ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))
