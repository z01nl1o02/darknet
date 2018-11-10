./darknet detector train examples/traffic/image.data examples/traffic/yolov3.cfg pretrained/darknet53.conv.74 2>&1 | tee examples/traffic/train.log
