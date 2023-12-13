print ("hello world")

import sys
print(sys.path)
import struct;print(struct.calcsize("P") * 8)
import tensorflow as tf
print(tf.reduce_sum(tf.random.normal([1000, 1000])))

