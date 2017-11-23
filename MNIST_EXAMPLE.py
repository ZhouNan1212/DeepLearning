# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#载入MNIST数据集

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

print "Training data size: ", mnist.train.num_examples
print "Training data size: ", mnist.validation.num_examples
print "Training data size: ", mnist.test.num_examples

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
# 从train中选取batch_size个训练数据。
print "X shape:", xs.shape
print "Y shape:", ys.shape

# MNIST数据集相关参数
INPUT_NODE = 784
OUTPUT_NODE = 10

# 配置神经网络的参数
LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.8  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 正则项的系数
TRAINING_STEPS = 30000  # 训练论数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights2)) +
            avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2) + avg_class.average(biases2))


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    #生成隐藏层的参数
    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    #生成输出层参数
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, weights1, biases1, weights2, biases2)

    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())




















