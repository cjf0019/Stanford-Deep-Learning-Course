# Stanford-Deep-Learning-Course

The following files contain my answers to the homework assignments for Stanford's
CS 224n, Natural Language Processing with Deep Learning course:
http://web.stanford.edu/class/cs224n/syllabus.html . 

Summary of Assignments:
Assignment 1 focuses on coding neural networks and on word2vec embeddings with
numpy. "softmax" and "sigmoid" contain code for numerically stable versions of
the activation functions that can also handle matrices as inputs, while "sgd"
computes stochastic gradient descent for training. 

"word2vec" is a raw numpy coding of the algorithm, implementing skip-gram and
allowing for negative sampling. "sentiment" applies word2vec for sentiment analysis
on the IMDB data set. Word vectors of individual sentences are averaged and 
logistic regression applied.

Assignment 2 deals with tensorflow basics and dependency parsing. "softmax" writes
a manual version of the cross entropy loss written in tensorflow. "classifier" is
a fully blown network in tensorflow for using a softmax classifier on a dataset,
including training. "initialization" is a manual version of xavier initialization
(sqrt(6 / sum(matrix_dims))) 

"parser_transitions" performs transition-based parsing on sentences, where a sequence
a transitions for the sentences serve as input (possible transitions include "SHIFT",
"RIGHT-ARC", and "LEFT-ARC"). "parser_model" is then the network for predicting 
the next transition of a sentence. It utilizes "parser_transitions" to convert a 
body of text into parsings for training. The network consists of an embedding and a
Relu layer, which is trained with softmax cross entropy. 

Assignment 3 tackles RNNs and GRUs,specitically for the task of named entity 
recognition (NER). "window" performs a Relu-Softmax classification using a window
around each entity in order to predict what kind it is (Person, Organization,
Location, Miscellaneous)

The same NER problem is tackled with a RNN and GRU, using sequential windows for
each state. "rnn_cell" contains a class written in tensor flow for representing 
one state of a RNN neural network. "rnn" contains the full RNN network, which,
in addition to setting up the network, training, and prediction, also includes 
window padding to make the inputs of equal length. 

"gru_cell" and "gru" contain one state of a gru and a fully implementable GRU
network respectively.

NOTE: The codes included are only the ones for which I have contributed. The full
codes can be found from the above syllabus link. My contributions are noted by
the commments "###YOUR CODE HERE" and "###END YOUR CODE"