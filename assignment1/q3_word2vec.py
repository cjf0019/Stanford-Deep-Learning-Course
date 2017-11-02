#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    rowsum = np.expand_dims(np.sqrt(np.sum(np.square(x), axis=1)), axis=1)    ### END YOUR CODE
    x = x / rowsum
    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """
     
    predicted = np.expand_dims(predicted, axis=1) 
    softy = softmax(np.dot(outputVectors, predicted), 0)
    ### YOUR CODE HERE
    onehotmatrix = np.eye(len(outputVectors), len(outputVectors))  
    cost = -np.log(softy[target])
#    gradPred = np.sum(np.dot(outputVectors, predicted), axis=0) - outputVectors[target]
    gradPred = np.dot((softy-np.expand_dims(onehotmatrix[target].T, axis=1)).T, outputVectors)    
    grad = np.dot((softy-np.expand_dims(onehotmatrix[target].T, axis=1)), predicted.T)
    gradPred = np.squeeze(gradPred)
    grad = np.squeeze(grad)    
    print 'gradpred is ', gradPred
    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    predicted = np.expand_dims(predicted, axis=1)        
    indices.remove(target)
    cost = -np.sum(np.log(sigmoid(np.dot(outputVectors[target].T, predicted)))) \
       -np.sum(np.log(sigmoid(-np.dot(outputVectors[indices], predicted))))
    gradPred = np.dot((np.squeeze(sigmoid(np.dot(outputVectors[target].T, predicted)))-1), \
       outputVectors[target]) - np.sum(np.dot((np.squeeze(sigmoid(-np.dot(outputVectors[indices], predicted)))-1), \
       outputVectors[indices]))
    gradfxn = lambda negvec: -np.dot((sigmoid(-np.dot(negvec, predicted))-1), predicted.T)
    grad = np.zeros(np.shape(outputVectors))
    grad[indices] = gradfxn(outputVectors[indices])     #calculate gradient for only negatively sampled words]    
    grad[target] = -gradfxn(-outputVectors[target])
   ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    if len(contextWords) > 2*C:
        raise ValueError('Context window cannot exceed 2C.')
    predicted = inputVectors[tokens[currentWord]]           
    if word2vecCostAndGradient == negSamplingCostAndGradient:
        for word in contextWords:
            target = tokens[word]          
            onecost, onegradIn, onegradOut = negSamplingCostAndGradient(predicted, 
                                                    target, outputVectors, dataset)
            cost += onecost        
            gradIn[tokens[currentWord]] += onegradIn
            gradOut += onegradOut
    else:
        for word in contextWords:           
            target = tokens[word]                    
            onecost, onegradIn, onegradOut = softmaxCostAndGradient(predicted, 
                                                target, outputVectors, dataset)
            cost += onecost          
            gradIn[tokens[currentWord]] += onegradIn
            gradOut += onegradOut

    ### END YOUR CODE
####CHECK WHAT HAPPENS IF YOU DIVIDE BY CONTEXT SIZE (THE GRADS)
    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    """
Splits word2vec training into batches, with the total number of batches occurring
over one iteration. This function can be fed into stochastic gradient descent.
The "wordVectors" includes both the initial "input" weights in word2vec (what is
commonly considered the word vector) as well as the "output" weights that are 
used in the softmax layer. It is assumed the two sets of weights are of the same
size and split such that the first half is input weights and the second half 
output weights. The gradients are separately updated after each batch. Weights
are updated after each batch set.
    """
    batchsize = 50
    cost = 0.0
    np.seterr(all='raise')    
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)   #choose a context for training

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(     #calculate cost and gradients for one context
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
#    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
#        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
#        dummy_vectors)
#    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
#        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
#        dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
#    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
#        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
#        dummy_vectors)
#    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
#        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
#        dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()