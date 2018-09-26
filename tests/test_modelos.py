import unittest
import sys
import os

import tensorflow as tf
import numpy as np

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CURRENT_PATH, '../'))

from axon_conabio.models.tf_model import TFModel


class Model1(TFModel):
    def _predict(self, inputs):
        with tf.variable_scope('layer_1'):
            matrix = tf.get_variable(
                'matrix1',
                shape=[2, 2],
                dtype=tf.float32)
            bias = tf.get_variable(
                'bias1',
                shape=[2],
                dtype=tf.float32,
                initializer=tf.zeros_initializer)
            result = tf.matmul(matrix, inputs) + bias

        return result

    def build_loss(self, inputs, labels):
        pass


class TestTFModelClass(unittest.TestCase):
    def test_same_variables_over_multiple_builds(self):
        input1 = tf.placeholder(tf.float32, shape=[2, 1])
        input2 = tf.placeholder(tf.float32, shape=[2, 1])

        model1 = Model1()

        output1_tensor = model1.predict(input1)
        output2_tensor = model1.predict(input2)

        test_input = np.random.random([2, 1])

        sess = tf.Session()

        sess.run(model1.init_op())

        output1, output2 = sess.run(
            [output1_tensor, output2_tensor],
            feed_dict={input1: test_input, input2: test_input})

        self.assertTrue((output1 == output2).all())

    def test_save_and_restore(self):
        path = '/tmp/axon_conabio/model1'
        test_input = np.random.random([2, 1])

        with tf.Graph().as_default():
            input1 = tf.placeholder(tf.float32, shape=[2, 1])
            model1 = Model1()
            output1_tensor = model1.predict(input1)

            sess = tf.Session()
            sess.run(model1.init_op())
            model1.save(sess, path)

            output1 = sess.run(output1_tensor, feed_dict={input1: test_input})

        with tf.Graph().as_default():
            input2 = tf.placeholder(tf.float32, shape=[2, 1])
            model2 = Model1()
            output2_tensor = model2.predict(input2)

            sess = tf.Session()
            model2.restore(sess, path)

            output2 = sess.run(output2_tensor, feed_dict={input2: test_input})

        self.assertTrue((output1 == output2).all())

    def test_save_and_restore_different_models(self):
        path1 = '/tmp/axon_conabio/model1'
        path2 = '/tmp/axon_conabio/model2'

        test_input = np.random.random([2, 1])

        with tf.Graph().as_default():
            input1 = tf.placeholder(tf.float32, shape=[2, 1])
            model1 = Model1()
            model2 = Model1()
            output1_tensor = model1.predict(input1)
            output2_tensor = model2.predict(input1)

            sess = tf.Session()

            sess.run(model1.init_op())
            sess.run(model2.init_op())

            model1.save(sess, path1)
            model2.save(sess, path2)

            output1_pre = sess.run(
                output1_tensor,
                feed_dict={input1: test_input})
            output2_pre = sess.run(
                output2_tensor,
                feed_dict={input1: test_input})

        with tf.Graph().as_default():
            input1 = tf.placeholder(tf.float32, shape=[2, 1])
            model1 = Model1()
            model2 = Model1()
            output1_tensor = model1.predict(input1)
            output2_tensor = model2.predict(input1)

            sess = tf.Session()

            model1.restore(sess, path1)
            model2.restore(sess, path2)

            output1_post = sess.run(
                output1_tensor,
                feed_dict={input1: test_input})
            output2_post = sess.run(
                output2_tensor,
                feed_dict={input1: test_input})

        self.assertTrue((output1_pre == output1_post).all())
        self.assertTrue((output2_pre == output2_post).all())
