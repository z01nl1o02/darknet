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




