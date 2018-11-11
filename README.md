# darknet #
[darknet](pjreddie.com/darknet/)

# FGSM #
Fast Gradient Sign Attatck(FGSM) 出自 [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572).沿梯度正方向修改输入数据，达到迷惑模型的目的。   
![fgsm](https://github.com/z01nl1o02/darknet/blob/dev/fgsm.jpg)
## 流程说明
   * 计算网络error相对于输入数据的梯度   
     forward($data_{old}$)   
     grad = backword()
   * 沿梯度正方向修改输入数据   
     $data_{new} = data_{old} + eps \times sign(grad)$
   * 计算修改后数据的预测结果   
     forward($data_{new}$)   

## 结果   
  随着eps的增加，模型在新数据上预测准确率逐渐降低   
  ![curve](https://github.com/z01nl1o02/darknet/blob/dev/fgsm_curve.jpg)


      
## 使用方式
 ./darknet classifier fgsm cfg/mnist.data  cfg/mnist.cfg backup/mnist_5.weights -eps 0.1



# yolov3训练
  [修改example代码](https://blog.csdn.net/z0n1l2/article/details/83933765)
  train.sh/test.sh
## 基于voc数据集格式+预训练模型
   ./darknet detector train examples/traffic/image.data examples/traffic/yolov3.cfg pretrained/darknet53.conv.74 2>&1 | tee examples/traffic/train.log

## 统计模型recalling
  ./darknet detector recall2 examples/traffic/image.data examples/traffic/yolov3.cfg examples/traffic/yolov3_900.weights examples/traffic/test.txt  2>&1 | tee examples/traffic/yolov3_900.test.log
          
## 批量运行，保存结果
  ./darknet detector test2 examples/traffic/image.data examples/traffic/yolov3.cfg examples/traffic/yolov3_800.weights examples/traffic/test.txt -out predict/


