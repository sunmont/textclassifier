from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six
import numpy as np

from gensim.models import Word2Vec

class Word2vecVocabulary(object):
  def __init__(self, unknown_token="<UNK>", support_reverse=True):
    self._freeze = False
    self.text9_model = Word2Vec.load_word2vec_format('model/text9.vec')
    self.len = 0

  def __len__(self):
    return self.len

  def freeze(self, freeze=True):
    self._freeze = freeze

  def get(self, w):
    self.len += 1
    #if self._freeze:
    #    return 0
    if w in self.text9_model.vocab:
        #print(self.text9_model[w])
        return np.average(self.text9_model[w])
    else:
        return 0

  def add(self, category, count=1):
      return

  def trim(self, min_frequency, max_frequency=-1):
      return

  def reverse(self, class_id):
      return

  def __bool__(self):
      return True
