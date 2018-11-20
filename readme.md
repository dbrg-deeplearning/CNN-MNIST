# CNN-MNIST
本例程旨在通过「用CNN实现手写体识别项目」使实验室人员了解如何使用本服务器。
# 准备数据
## 数据集介绍
MNIST 数据集可在 http://yann.lecun.com/exdb/mnist/ 获取, 它包含了四个部分:

* Training set images: train-images-idx3-ubyte.gz (9.9 MB, 解压后 47 MB, 包含 60,000 个样本)
* Training set labels: train-labels-idx1-ubyte.gz (29 KB, 解压后 60 KB, 包含 60,000 个标签)
* Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 解压后 7.8 MB, 包含 10,000 个样本)
* Test set labels: t10k-labels-idx1-ubyte.gz (5KB, 解压后 10 KB, 包含 10,000 个标签)

MNIST数据集来自美国国家标准与技术研究所，National Institute of Standard and Technology（NIST）。训练集(Training set)由来自250个不同人手写的数字构成，其中50%是高中生，50%来自人口普查局(the Census Bureau)的工作人员。测试集(Test set)也是同样比例的手写数据。

```input_data.py``` 文件是TensorFlow官方的MNIST文档里的文件，首次执行时会从网上下载数据，由于实验室服务器不连接外网，所以首先我们需要将数据导入服务器中。

## 在服务器与本地之间拷贝数据
1. 通过scp命令实现互相拷贝

  ```
  # 从本地向远程拷贝
  # dir/filename 为本地文件路径，/cluster/origindata 为要拷贝到服务器的路径
  scp dir/filename higis@10.1.1.57:/cluster/origindata
  # 需要输入密码
  higis@10.1.1.57's password:

  # 从远程向本地拷贝
  # /cluster/origindata/filename 为服务器上的文件路径 /dir 为要拷贝到的本地路径，注意两者中间有个空格
  scp higis@10.1.1.57:/cluster/origindata/filename /dir
  ```

2. 通过「连接到服务器」功能拷贝

  在 Ubuntu14、16的文件管理器中，左侧最小角有「连接到服务器」，Ubuntu18中左侧最下角为「其他位置」点击后有「连接到服务器」选项，在服务器地址中输入```sftp://10.1.1.57/home/higis```即可连接到服务器，首次连接需要输入密码。

  连接到服务器之后，默认进入```/home/higis```目录，可通过键盘「退格」键回退到根目录，进入/cluster里。

  之后可以直接在本机和服务器进行复制、粘贴。但是注意只有对服务器中用户为```higis```的目录有写入和删除的权限。

  可通过```ll```命令查看文件属于哪个用户。如修改某```file```文件所属用户，可通过下面命令：

    
    chown file higis
  

## 执行程序

通过同样的方式，将```main.py```与```input_data.py```拷到服务器上的项目目录下。

执行程序前首先进入（启动）虚拟环境。

在```/home/higis```下执行命令：

```
source deeplearning/deeplearning/bin/activate
```
即可进入名为```deeplearning```的虚拟环境。虚拟环境一旦启动，全局有效。
> 关于虚拟环境：
  虚拟环境是用来不同项目的包。你常常要使用依赖于某个库的不同版本的代码。例如，你的代码可能使用了 Numpy 中的新功能，或者使用了已删除的旧功能。实际上，不可能同时安装两个 Numpy 版本。你要做的应该是，为每个 Numpy 版本创建一个环境，然后在项目的对应环境中工作。虚拟环境管理器常用的有```conda```,```virtualenv```和```pyenv```，本机使用```virtualenv```管理环境，目前只创建了一个环境。如有创建新的虚拟环境的需求，请在微信群中与大家讨论。避免损坏现有环境。

启动虚拟环境后，回到我们的项目目录，运行：

```
python3 main.py
```
即可开始执行程序，本项目的训练与测试都写在```main.py```内，不涉及存储与导出模型，程序中有详细注释，可供参考。

## GPU使用配置(zhouyirong09-11.20)

第13行至21行为GPU使用初始配置并附解释，可自行Google其它更详细资料。

重点强调第15行，实验发现若不添加此句时程序启动后tensorflow使用机器内所有GPU并分配指定显存导致浪费，请参见https://github.com/keras-team/keras/issues/6031

其它同学若有更好的配置方法，请及时分享!
