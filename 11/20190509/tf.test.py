import os
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

saver = tf.train.Saver()
session =  tf.Session()

path = os.path.abspath(os.path.dirname(__file__))
saver.restore(session, os.path.join(path, "tf_model"))

yl = tf.argmax(y,1)
zl = tf.argmax(z,1)
ac = tf.reduce_mean(tf.cast(tf.equal(yl,zl) , tf.float32))
y_label, z_label, accuracy = session.run(
    (yl,zl,ac), feed_dict={x:mnist.test.images, z:mnist.test.labels})


print("TEST RESULT : ")
count = [[ 0 for i in range(10)] for j in range(10)]
for i,j in zip(y_label, z_label):
    count[i][j] += 1
print("Anser  ",end ='')
for i in range(10):
    print("  [{0}]".format(i),end="")
print()
for i in range(10):
    print("Predict[{0}]".format(i), end="")
    for j in range(10):
        print("{0:6d}".format(count[i][j]), end="")
    print()
    
print("ACCURACY : ",accuracy*100, "%")