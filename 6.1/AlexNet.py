# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())
    
def inference(images):
    parameters = []
    
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], stddev=0.1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), name='biases')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]
    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',
                           name='pool1')
    print_activations(pool1)
    
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], stddev=0.1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), name='biases')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
        print_activations(conv2)
        parameters +=[kernel, biases]
    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',
                           name='pool2')
    print_activations(pool2)
    
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], stddev=0.1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), name='biases')
        conv3 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
        print_activations(conv3)
        parameters +=[kernel, biases]
    
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), name='biases')
        conv4 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
        print_activations(conv4)
        parameters +=[kernel, biases]

    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), name='biases')
        conv5 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope)
        print_activations(conv5)
        parameters +=[kernel, biases]
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',
                           name='pool5')
    print_activations(pool5)
     
    #AlexNet前5层卷积层
    return pool5, parameters
    
    #AlexNet后3个全连接层
    with tf.name_scope('fc6') as scope:
        reshape = tf.reshape(pool5, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weight = tf.Variable(tf.truncated_normal([dim, 4096], stddev=0.04), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[4096]), name='biases')
        fc6 = tf.nn.relu(tf.matmul(reshape, weight)+biases, name=scope)
        print_activations(fc6) 
        parameters +=[weight, biases]
        
    with tf.name_scope('fc7') as scope:
        weight = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.04), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[4096]), name='biases')
        fc7 = tf.nn.relu(tf.matmul(fc6, weight)+biases, name=scope)
        print_activations(fc7)
        parameters +=[weight, biases]
        
    with tf.name_scope('fc8') as scope:
        weight = tf.Variable(tf.truncated_normal([4096, 1000], stddev=0.04), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[1000]), name='biases')
        logits = tf.add(tf.matmul(fc7, weight), biases, name=scope)
        print_activations(logits)
        parameters +=[weight, biases]
    
    return logits, parameters
    
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches+num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time()-start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' % (datetime.now(), i-num_steps_burn_in,
                                                        duration))
            total_duration += duration
            total_duration_squared += duration*duration
    
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec/batch' % (datetime.now(), info_string,
                                                               num_batches, mn, sd))
    
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3],
                                              stddev=0.1))
        pool5, parameters = inference(images)
        
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        
        time_tensorflow_run(sess, pool5, 'Forward')
        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, 'Forward-backward')
        
run_benchmark()
