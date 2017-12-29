import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    exp_scores = np.exp(scores)
    pred_prob = exp_scores / np.sum(exp_scores)
    correct_class_prob = pred_prob[y[i]]
    loss -= np.log(correct_class_prob)
    grad_scores = pred_prob
    grad_scores[y[i]] -= 1
    dW += np.reshape(X[i], (3073, 1)).dot(np.reshape(grad_scores, (1, 10)))

  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  exp_scores = np.exp(scores)
  pred_prob = exp_scores / np.reshape(np.sum(exp_scores, axis=1), (num_train, 1))
  correct_class_prob = pred_prob[np.arange(num_train), y]
  loss -= np.sum(np.log(correct_class_prob))
  grad_scores = pred_prob
  grad_scores[np.arange(num_train), y] -= 1
  dW += X.T.dot(grad_scores)

  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

