#./darknet detector test2 examples/traffic/image.data examples/traffic/yolov3.cfg examples/traffic/yolov3_800.weights examples/traffic/test.txt -out predict/

./darknet detector train examples/traffic/image.data examples/traffic/yolov3.cfg pretrained/darknet53.conv.74 2>&1 | tee examples/traffic/train.log
./darknet detector recall2 examples/traffic/image.data examples/traffic/yolov3.cfg examples/traffic/yolov3_900.weights examples/traffic/test.txt  2>&1 | tee examples/traffic/yolov3_900.test.log
