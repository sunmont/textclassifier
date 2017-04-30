from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import encoders
from tensorflow.contrib.learn.python.learn.estimators import estimator

import setting

from bow import * #bag_of_words_model 
from rnn import rnn_model 

from gensim.models import Word2Vec

learn = tf.contrib.learn

FLAGS = None

MAX_DOCUMENT_LENGTH = 50 #10

from tokenizer import Tokenizer
from cat_vocab import CatVocabulary
def process_cat(x_train, x_test):
    tokenizer = Tokenizer()
 
    cat_vocab = CatVocabulary(50, tokenizer.tokenizer0)
    #fit
    cat_vocab.fit(x_train)

    return bag_of_words_model, cat_vocab, cat_vocab.get_vec(x_train), cat_vocab.get_vec(x_test)

from emb_vocab import EmbVocabulary
def process_emb(x_train, x_test):
    tokenizer = Tokenizer()
 
    emb_vocab = EmbVocabulary(50, tokenizer.tokenizer0)
    #fit
    emb_vocab.fit()

    return emb_bag_of_words_model, emb_vocab, emb_vocab.get_vec(x_train), emb_vocab.get_vec(x_test)

def main(unused_argv):
 # global n_words
  # Prepare training and testing data
  dbpedia = learn.datasets.load_dataset(
      'dbpedia', test_with_fake_data=False) #FLAGS.test_with_fake_data)
  x_train = pandas.DataFrame(dbpedia.train.data)[1]
  y_train = pandas.Series(dbpedia.train.target)
  x_test = pandas.DataFrame(dbpedia.test.data)[1]
  y_test = pandas.Series(dbpedia.test.target)

  if FLAGS.embeddings:
    model_, vocabulary_, x_transform_train, x_transform_test = process_emb(x_train, x_test)
  else:
    model_, vocabulary_, x_transform_train, x_transform_test = process_cat(x_train, x_test)

  x_train = np.array(list(x_transform_train))
  x_test = np.array(list(x_transform_test))
  
  setting.n_words = len(vocabulary_)

  print('Total words: %d' % setting.n_words)
  print('x_train shape: ' + str(x_train.shape))
  print('x_test shape: ' + str(x_test.shape))

  # Build model
  # Switch between rnn_model and bag_of_words_model to test different models.
  model_fn = rnn_model
  if FLAGS.bow_model:
    model_fn = model_
  else:
    model_fn = rnn_model

  classifier = estimator.Estimator(model_fn=model_fn)

  # Train and predict
  estimator.SKCompat(classifier).fit(x_train, y_train, steps=100)
  y_predicted = [
      p['class'] for p in classifier.predict(
          x_test, as_iterable=True)
  ]
  score = metrics.accuracy_score(y_test, y_predicted)
  print('Accuracy: {0:f}'.format(score))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--bow_model',
      default=True,
      help='Run with BOW model instead of RNN.',
      action='store_true')
  parser.add_argument(
      '--embeddings',
      default=False,
      help='Use embedings or category vocabulary',
      action='store_true')
  parser.add_argument(
      '--activation',
      default="tanh",
      help='activation function')
  FLAGS, unparsed = parser.parse_known_args()

  setting.activation_fn = FLAGS.activation

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
