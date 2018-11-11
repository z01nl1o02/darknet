#./darknet detector test2 examples/traffic/image.data examples/traffic/yolov3.cfg examples/traffic/yolov3_800.weights examples/traffic/test.txt -out predict/

./darknet detector recall2 examples/traffic/image.data examples/traffic/yolov3.cfg examples/traffic/yolov3_500.weights examples/traffic/test.txt  2>&1 | tee examples/traffic/yolov3_500.test.log
