from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six
import numpy as np

class CatVocabulary(object):
  def __init__(self, max_doc_len, tokenizer_func, unknown_token="<UNK>", support_reverse=True):
    self._freeze = False
    self.len = 0
    self.vocab = {unknown_token : 0}
    self.vec_dim = max_doc_len 
    self.tokenizer_func = tokenizer_func

  def __len__(self):
    return len(self.vocab)

  def freeze(self, freeze=True):
    self._freeze = freeze
  
  def fit(self, docs):
    for tokens in self.tokenizer_func(docs):
      for token in tokens:
        if token not in self.vocab:
          self.vocab[token] = len(self.vocab)

  def get_vec(self, docs):
    for tokens in self.tokenizer_func(docs):
      ids = np.zeros(self.vec_dim, np.int32) #float)
      for idx, token in enumerate(tokens):
        if idx >= self.vec_dim:
            break
        ids[idx] = self.get(token)
      yield ids

  def get(self, w):
    #self.len += 1
    #if self._freeze:
    #    return 0
    if w in self.vocab:
        return self.vocab[w]
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

# test
from tokenizer import Tokenizer
raw_docs = [
" Abbott of Farnham E D Abbott Limited was a British coachbuilding business based in Farnham Surrey trading under that name from 1929. A major part of their output was under sub-contract to motor vehicle manufacturers. Their business closed in 1972."
," Schwan-STABILO is a German maker of pens for writing colouring and cosmetics as well as markers and highlighters for office use. It is the world's largest manufacturer of highlighter pens Stabilo Boss."
" Q-workshop is a Polish company located in Poznań that specializes in designand production of polyhedral dice and dice accessories for use in various games (role-playing gamesboard games and tabletop wargames). They also run an online retail store and maintainan active forum community.Q-workshop was established in 2001 by Patryk Strzelewicz – a student from Poznań. Initiallythe company sold its products via online auction services but in 2005 a website and online store wereestablished."
]

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import encoders
if __name__ == '__main__':
    tokenizer = Tokenizer()
    docs =[]
 
    cat_vocab = CatVocabulary(50, tokenizer.tokenizer0)
    #fit
    cat_vocab.fit(raw_docs)
    # get ids
    for id in cat_vocab.get_vec(raw_docs):
        print(id)
        docs.append(id)

    print(docs)
    docs = np.array(list(docs))
    print(docs.shape)
    with tf.Session() as sess:
        #docs = [[0, 1], [2, 3]]
        enc = encoders.bow_encoder(docs, len(cat_vocab), 80)
        sess.run(tf.global_variables_initializer())
	#self.assertAllEqual([2, 3], enc.eval().shape)
        print(enc.eval())
        print(enc.eval().shape)
