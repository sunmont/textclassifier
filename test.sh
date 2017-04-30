#!/usr/bin/env bash

METHOD_RNN=--rnn_model
VOCABULARY_CATE=--categorial
VOCABULARY_EMBED=--embeddings
ACTIVATION_TANH=tanh
ACTIVATION_SIGMOD=sigmod
ACTIVATION_RELU=relu

echo test result [bow, categorial, tanh]: $(python text_classifier.py --bow_model 2>/dev/null|grep Accuracy)
echo test result [bow, categorial, sigmoid]: $(python text_classifier.py --bow_model --activation sigmod 2>/dev/null|grep Accuracy)
echo test result [bow, categorial, relu]: $(python text_classifier.py --bow_model --activation relu 2>/dev/null|grep Accuracy)
echo --------
echo test result [bow, embeddings, tanh]: $(python text_classifier.py --bow_model --embeddings 2>/dev/null|grep Accuracy)
echo test result [bow, embeddings, sigmoid]: $(python text_classifier.py --bow_model --embeddings --activation sigmoid 2>/dev/null|grep Accuracy)
echo --------
echo test result [rnn, relu]: $(python text_classifier.py --rnn_model 2>/dev/null|grep Accuracy)

