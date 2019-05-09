import os
import time
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

print("GET MINIST!!!",flush=True)
mnist = input_data.read_data_sets("-",one_hot=True)
print("END GETIING DATASET!")

x = tf.placeholder(tf.float32,[None,784])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x,w) + b
z=tf.placeholder(tf.float32,[None,10])



