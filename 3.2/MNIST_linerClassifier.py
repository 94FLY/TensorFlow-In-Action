# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#载入MNIST数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("E:/Programming/python/TF-in-action/3.2",one_hot=True)

#查看数据维度
print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)

#简单的线性分类器
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run(feed_dict = {x: batch_xs, y_: batch_ys})
    
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))