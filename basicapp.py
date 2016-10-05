""" basicapp.py - Basic usage of DAGRNN Library

Training Goal:
    composer learns to add all things together
    core learns to swap two columns
    composer + core learns to divide all things by 2

"""

import numpy as np
import tensorflow as tf
import dagrnn as dn
import basicapp_util as util

# Construction

unit_shape = tf.TensorShape([1, 2])

print unit_shape


def core(x):
    # x is supposed to be unit shape
    W = tf.Variable(tf.random_normal([2, 2]), name="W_core")
    y = tf.matmul(x, W, name="y_core")
    return y


def composer1(x):
    # composer for degree 1, x is supposed to be unit shape
    W = tf.Variable(tf.random_normal([1, 1]), name="W_composer1")
    y = tf.matmul(W, x, name="y_composer1")
    return y


def composer2(x):
    # composer for degree 2, x is supposed to be 2 * unit shape
    W = tf.Variable(tf.random_normal([1, 2]), name="W_composer2")
    y = tf.matmul(W, x, name="y_composer2")
    return y


def composer3(x):
    # composer for degree 3, x is supposed to be 3 * unit shape
    W = tf.Variable(tf.random_normal([1, 3]), name="W_composer3")
    y = tf.matmul(W, x, name="y_composer3")
    return y

composers = [composer1, composer2, composer3]

y = dn.assemble(composers, core)

y_ = tf.placeholder(tf.float32, [None])

loss = tf.reduce_sum(tf.pow(y_ - y, 2))

train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)

init = tf.initialize_all_variables()

# Execution

sess = tf.Session()

sess.run(init)

util.write_graph()

steps = 1000

for i in range(steps + 1):
    flag = i % 100 == 0
    if flag:
        print ('Step : %d' % i), sess.run(['W_core', 'W_composer1', 'W_composer2', 'W_composer3'])
    vertices, edges, outputs = util.gen_DAG(verbose=False)
    sess.run(train_step, feed_dict={y_ : outputs}.update(dn.analyze(vertices, edges)))

# Check

vertices, edges, outputs = util.gen_DAG(verbose=False)
ty, ty_ = sess.run([y, y_], feed_dict={y_ : outputs}.update(dn.analyze(vertices, edges)))
print 'Output : ', ty, 'Ground Truth : ', ty_



