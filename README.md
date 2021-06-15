# iris-classification-based-on-neural-network
Iris dataset classification based on two-layer neural network （PyTorch）



使用两层全连接神经网络对iris数据集进行分类

iris数据集：150个样本，共3类，每个样本4个特征。sklearn内置该数据集

iris可视化分析:  iris_visualization.ipynb

模型训练及测试: iris_NN.py



模型结构：输入层（4个神经元）-- 隐藏层（10） -- 输出层（3）

激活函数：ReLU

优化方法：SGD with Momentum （learning rate 0.001 momentum 0.9)

分类结果：Test Accuracy: 100%



