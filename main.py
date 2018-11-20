#!usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:yaoli 
@file: mnistdemo.py
@time: 2018/06/26 
"""

import input_data
import tensorflow as tf
import os


# 配置gpu可使用情况, 取值可为"-1"或"0"或"0,1"等, 机器有多块gpu时各gpu编号依次为0至m-1
# "-1"表示不可使用gpu, "0"表示可使用0号gpu, "0,1"表示可同时使用0号和1号gpu
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# 设置使用规则, allow_soft_placement为True时指若程序中指定使用的gpu満负荷则自动使用可用的gpu
config = tf.ConfigProto(allow_soft_placement=True)
# allow_growth为True时指根据需要, gpu显存使用随程序进行分配, 为False时则程序启动直接分配整块gpu
config.gpu_options.allow_growth = True
# per_process_gpu_memory_fraction指最多使用gpu显存的比例, 若不指定tensorflow直接分配所用gpu所有显存
config.gpu_options.per_process_gpu_memory_fraction = 0.9


# 读取数据
mnist = input_data.read_data_sets('MNIST-data', one_hot=True)
sess = tf.InteractiveSession()


# 构建cnn网络结构
# 自定义卷积函数（后面卷积时就不用写太多）

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

# 自定义池化函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 设置占位符，尺寸为样本输入和输出的尺寸
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_img = tf.reshape(x, [-1, 28, 28, 1])

# 设置第一个卷积层和池化层
w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv1 = tf.nn.relu(conv2d(x_img, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 设置第二个卷积层和池化层
w_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 50], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[50]))
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 设置第一个全连接层
w_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 50, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 50])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout（随机权重失活）
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 设置第二个全连接层
w_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y_out = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# 建立loss function，为交叉熵
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_out), reduction_indices=[1]))
# 配置Adam优化器，学习速率为1e-4
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# 建立正确率计算表达式
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 开始喂数据，训练
tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
        print("step %d,train_accuracy= %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 训练之后，使用测试集进行测试，输出最终结果
print("test_accuracy= %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1}))
