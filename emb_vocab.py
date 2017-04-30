from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six
import numpy as np

from gensim.models import Word2Vec

DIM_TEXT9 = 300 #100 #50 #100

class EmbVocabulary(object):
  def __init__(self, max_doc_len = 50, tokenizer_func = None, unknown_token="<UNK>", support_reverse=True):
    self._freeze = False
    self.len = 0
    self.max_doc_len = max_doc_len
    self.vec_dim = DIM_TEXT9 
    self.tokenizer_func = tokenizer_func

  def __len__(self):
    return self.len

  def freeze(self, freeze=True):
    self._freeze = freeze
  
  def fit(self, docs = None):
    #word2vec - fast2text
    #self._model = Word2Vec.load_word2vec_format('model/text9.vec')
    # glove
    self._model = Word2Vec.load_word2vec_format('model/glove.6B.300d.w2vformat.txt', binary=False)
    #self._model = Word2Vec.load_word2vec_format('model/glove.6B.100d.w2vformat.txt', binary=False)
    #self._model = Word2Vec.load_word2vec_format('model/glove.6B.50d.w2vformat.txt', binary=False)

  # mean of each word embeding
  def get_vec(self, docs):
    for tokens in self.tokenizer_func(docs):
      doc_vec = np.zeros(self.vec_dim, np.float64)  #float)
      n = 0
      for idx, token in enumerate(tokens):
        if idx >= self.max_doc_len:
            break
        doc_vec = np.add(doc_vec, self.get(token))
        n += 1
        
      yield np.divide(doc_vec, n)

  def get(self, w):
    self.len += 1
    #if self._freeze:
    #    return 0
    if w in self._model.vocab:
      return self._model[w]
    else:
      return np.zeros(self.vec_dim, np.float64) #float)

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

if __name__ == '__main__':
    tokenizer = Tokenizer()
 
    emb_vocab = EmbVocabulary(50, tokenizer.tokenizer0)
    #fit
    emb_vocab.fit()
    # get ids
    for vec in emb_vocab.get_vec(raw_docs):
        print(vec)
        print(vec.shape)

    vec_list = list(emb_vocab.get_vec(raw_docs))    
    for v in vec_list:
        print(v)
