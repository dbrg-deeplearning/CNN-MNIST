# CNN-MNIST
本本例程旨在通过用CNN实现手写体识别项目使实验室人员了解如何使用本服务器。
# 数据准备
## 数据集介绍
MNIST 数据集可在 http://yann.lecun.com/exdb/mnist/ 获取, 它包含了四个部分:

* Training set images: train-images-idx3-ubyte.gz (9.9 MB, 解压后 47 MB, 包含 60,000 个样本)
* Training set labels: train-labels-idx1-ubyte.gz (29 KB, 解压后 60 KB, 包含 60,000 个标签)
* Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 解压后 7.8 MB, 包含 10,000 个样本)
* Test set labels: t10k-labels-idx1-ubyte.gz (5KB, 解压后 10 KB, 包含 10,000 个标签)

MNIST数据集来自美国国家标准与技术研究所，National Institute of Standard and Technology（NIST）。训练集(Training set)由来自250个不同人手写的数字构成，其中50%是高中生，50%来自人口普查局(the Census Bureau)的工作人员。测试集(Test set)也是同样比例的手写数据。

```input_data.py``` 文件是TensorFlow官方的MNIST文档里的文件，首次执行时会从网上下载数据，由于实验室服务器不连接外网，所以首先我们需要将数据导入服务器中。

## 将数据导入服务器 
