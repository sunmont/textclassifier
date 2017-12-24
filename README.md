Text Classifier<br>
======<br>
An attempt for Multi-class classification NLP text classification using Tensorflow (tflearn).<br>
It tries with different embedding models (bag-of-mean, word2vec or GloVe) with variant activation functions<br>
(logistic sigmoid, ReLU, tanh)<br>

```
Model:<br>
x = embedding(text)<br>
h = tanh(Wx + b)<br>
u = Vh + c<br>
p = softmax(u)<br>
if testing:<br>
    prediction = arg maxy py<br>
    else: # training, with y as the given gold label<br>
        loss = -log(py) # cross entropy criterion<br>
```

Run:<br>
./test.sh<br>
<br>
```
test result [bow, categorial, tanh]: Accuracy: 0.942857<br>                                                                               test result [bow, categorial, sigmoid]: Accuracy: 0.900000<br>
test result [bow, categorial, relu]: Accuracy: 0.914286<br>
test result [bow, embeddings, tanh]: Accuracy: 0.900000<br>
test result [bow, embeddings, sigmoid]: Accuracy: 0.885714<br>
test result [rnn, relu]: Accuracy: 0.628571<br>
```

Requirements:<br>
Tensorflow<br>
pandas<br>
gensim<br>
numpy<br>
sklearn<br>
