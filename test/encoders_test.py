"""Tests for tensorflow.contrib.layers.python.layers.encoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import encoders
from tensorflow.contrib.layers.python.ops import sparse_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables

def _get_const_var(name, shape, value):
  return variable_scope.get_variable(
      name, shape, initializer=init_ops.constant_initializer(value))

if __name__ == '__main__':
    with tf.Session() as sess:
      docs = [[1, 1], [2, 3]]
      with variable_scope.variable_scope('test'):
        v = _get_const_var('embeddings', (4, 3),
                           [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
      #  self.assertEqual(v.name, 'test/embeddings:0')

#      emb = encoders.embed_sequence(docs, 4, 3, scope='test', reuse=True)
      emb = encoders.embed_sequence(docs, 4, 3, scope='test', reuse=True)
      sess.run(variables.global_variables_initializer())
      print(emb.eval())
      #self.assertAllClose(
      #    [[[3., 4., 5.], [3., 4., 5.]], [[6., 7., 8.], [9., 10., 11.]]],
      #    emb.eval())

#[
# [[  3.   4.   5.]
#  [  3.   4.   5.]]
# [[  6.   7.   8.]
#  [  9.  10.  11.]]
#]

