Text Classifier
======
An attempt for Multi-class classification NLP text classification using Tensorflow (tflearn).
It tries with different embedding models (bag-of-mean, word2vec or GloVe) with variant activation functions
(logistic sigmoid, ReLU, tanh)

Model:
x = embedding(text)
h = tanh(Wx + b)
u = Vh + c
p = softmax(u)
if testing:
    prediction = arg maxy py
    else: # training, with y as the given gold label
        loss = -log(py) # cross entropy criterion

Run:
./test.sh

test result [bow, categorial, tanh]: Accuracy: 0.942857                                                                                                                                         1 test result [bow, categorial, sigmoid]: Accuracy: 0.900000
test result [bow, categorial, relu]: Accuracy: 0.914286
test result [bow, embeddings, tanh]: Accuracy: 0.900000
test result [bow, embeddings, sigmoid]: Accuracy: 0.885714
test result [rnn, relu]: Accuracy: 0.628571

Requirements:
Tensorflow
pandas
gensim
numpy
sklearn
