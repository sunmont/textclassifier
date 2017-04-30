from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import encoders

import setting

learn = tf.contrib.learn

RNN_EMBEDDING_SIZE = 300 #50

def rnn_model(features, target):
  """RNN model to predict from sequence of words to a class."""
  # Convert indexes of words into embeddings.
  # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
  # maps word indexes of the sequence into [batch_size, sequence_length,
  # EMBEDDING_SIZE].
  word_vectors = tf.contrib.layers.embed_sequence(
      features, vocab_size=setting.n_words, embed_dim=RNN_EMBEDDING_SIZE, scope='words')

  # Split into list of embedding per word, while removing doc length dim.
  # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
  word_list = tf.unstack(word_vectors, axis=1)

  # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
  cell = tf.contrib.rnn.GRUCell(RNN_EMBEDDING_SIZE)

  # Create an unrolled Recurrent Neural Networks to length of
  # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
  _, encoding = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)

  # Given encoding of RNN, take encoding of last step (e.g hidden size of the
  # neural network of last step) and pass it as features for logistic
  # regression over output classes.
  target = tf.one_hot(target, 15, 1, 0)
  logits = tf.contrib.layers.fully_connected(encoding, 15, activation_fn=None)
  loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

  # Create a training op.
  train_op = tf.contrib.layers.optimize_loss(
      loss,
      tf.contrib.framework.get_global_step(),
      optimizer='Adam',
      learning_rate=0.01)

  return ({
      'class': tf.argmax(logits, 1),
      'prob': tf.nn.softmax(logits)
  }, loss, train_op)


