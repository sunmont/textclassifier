from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import six

TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",
                          re.UNICODE)

class Tokenizer(object):

    def __init__(self):
        pass

    def tokenizer0(self, iter):
        for str in iter:
            yield TOKENIZER_RE.findall(str)

    def tokenizer1(self, iter):
        for str in iter:
            #tokens = re.sub(r"[^a-z0-9]+", " ", str).split()
            tokens = re.sub(r"(?!)[^a-z0-9]+", " ", str).split()
            yield tokens 

raw_doc = [
" Abbott of Farnham E D Abbott Limited was a British coachbuilding business based in Farnham Surrey trading under that name from 1929. A major part of their output was under sub-contract to motor vehicle manufacturers. Their business closed in 1972."
," Schwan-STABILO is a German maker of pens for writing colouring and cosmetics as well as markers and highlighters for office use. It is the world's largest manufacturer of highlighter pens Stabilo Boss."
" Q-workshop is a Polish company located in Poznań that specializes in designand production of polyhedral dice and dice accessories for use in various games (role-playing gamesboard games and tabletop wargames). They also run an online retail store and maintainan active forum community.Q-workshop was established in 2001 by Patryk Strzelewicz – a student from Poznań. Initiallythe company sold its products via online auction services but in 2005 a website and online store wereestablished."
]

# test
if __name__ == '__main__':
    tokenizer = Tokenizer()
    
    tokenizer_ = tokenizer.tokenizer1

    for tokens in tokenizer_(raw_doc):
      for token in tokens:
        print(token)
