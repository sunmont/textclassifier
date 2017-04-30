from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import encoders
from tensorflow.python.ops import nn

import setting

learn = tf.contrib.learn

BOW_EMBEDING_DIM = 80 #50

#activation_fn = nn.relu #tf.nn.relu
#activation_fn = nn.sigmoid
#activation_fn = nn.tanh

ACTIVATIONS = {
        "relu" : nn.relu,
        "sigmod" : nn.sigmoid,
        "tanh" : nn.tanh
        }

activation_fn = ACTIVATIONS[setting.activation_fn]
def bag_of_words_model(features, target):
  """A bag-of-words model. Note it disregards the word order in the text."""
  target = tf.one_hot(target, 15, 1, 0)
  features = encoders.bow_encoder(
      features, vocab_size=setting.n_words, embed_dim=BOW_EMBEDING_DIM)
  logits = tf.contrib.layers.fully_connected(features, 15, activation_fn) #=None)
  loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
  #loss = tf.losses.softmax_cross_entropy(logits, target)
  train_op = tf.contrib.layers.optimize_loss(
      loss,
      tf.contrib.framework.get_global_step(),
      optimizer='Adam',
      learning_rate=0.01)
  return ({
      'class': tf.argmax(logits, 1),
      'prob': tf.nn.softmax(logits)
  }, loss, train_op)

def emb_bag_of_words_model(features, target):
  """A bag-of-words model. Note it disregards the word order in the text."""
  target = tf.one_hot(target, 15, 1, 0)
##  features = encoders.bow_encoder(
##      features, vocab_size=setting.n_words, embed_dim=BOW_EMBEDING_DIM)
  logits = tf.contrib.layers.fully_connected(features, 15, activation_fn) #=None)
  loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
  #loss = tf.losses.softmax_cross_entropy(logits, target)
  train_op = tf.contrib.layers.optimize_loss(
      loss,
      tf.contrib.framework.get_global_step(),
      optimizer='Adam',
      learning_rate=0.01)
  return ({
      'class': tf.argmax(logits, 1),
      'prob': tf.nn.softmax(logits)
  }, loss, train_op)


# test
if __name__ == '__main__':
    with tf.Session() as sess:
        docs = [[0, 1], [2, 3]]
        enc = encoders.bow_encoder(docs, 4, 3)
        sess.run(tf.global_variables_initializer())
	#self.assertAllEqual([2, 3], enc.eval().shape)
        print(enc.eval())
