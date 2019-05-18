'''
hw3_q3c.py

Given a dataset whose response variable is a class y = 1,2,3,..., k (i.e. a classification problem with k classes)
determine the optimal classifier according to the loss functions specified in the homework specification

'''

import numpy as np
import random
import matplotlib.pyplot as plt
from mnist.loader import MNIST


def grad_J(X, y, W, lam=0):
    '''

    X = data matrix with rows as measurements and columns are features (n x d)
    y = response variable (n x k) in one-hot encoding; each row is a label
    W = weights vector (d x k)
    lam = L2 regularizer (more specifically frobenius norm)
    '''

    yhat = np.dot(X, W)

    return -1 * np.dot(X.T, y - yhat) + 2 * lam * W

    #temp = y - yhat
    #n = X.shape[0]
    #result2 = -1 * sum([np.outer(X[i, :], temp[i, :]) for i in range(n)])
    #diff = abs(result1 - result2)
    #print(np.array_equal(result1, result2))
    #return result2



def J_function(X, y, W):
    '''

    X = data matrix with rows as measurements and columns are features (n x d)
    y = response variable (n x k) in one-hot encoding; each row is a label
    W = weights matrix (d x k)
    '''

    # NOTE: since we are summing over the two norms, we could equivalently just square all
    # elements and sum over entire matrix
    return 0.5 * np.sum(np.square(y - np.dot(X, W)))

def grad_L(X,y, W):
    '''

    X = data matrix with rows as measurements and columns are features (n x d)
    y = response variable (n x k) in one-hot encoding; each row is one label
    W = weights vector (d x k)
    '''
    # perform the softmax operation on each row
    yhat = np.exp(np.dot(X, W))
    yhat = yhat / np.expand_dims(np.sum(yhat, axis=1), axis=1)
    
    return -1 * np.dot(X.T, y - yhat)


#def L_function(X,y,W):
#    '''
#
#    X = data matrix with rows as measurements and columns are features (n x d)
#    y = response variable (n x k) in one-hot encoding; each row is one label
#    W = weights vector (d x k)
#    '''
#    Wyx = np.dot(X, np.dot(W, y.T))
#    summand = np.sum(np.exp(np.dot(W.T, X.T)) , axis=0)
#    return -1 * np.sum(Wyx - np.log(summand))


def error_rate(X, labels, W):
    '''

    X = data matrix with rows as measurements and columns are features (n x d)
    labels = class number in {1,2,...,l}
    W = weights vector (d x 1)
    '''
    predictions = np.argmax(np.dot(W.T, X.T), axis=0)

    return np.sum(np.where(predictions != labels, 1, 0)) / len(labels)


def gradient_descent(x_init, gradient_function, eta=0.1, delta=1e-4, X=None, y=None):
    '''
    Runs gradient descent to calculate minimizer x of the function whose gradient
    is defined by the given gradient_function.

    x_init = [w,b] is the initial values to set to the vector being descended on (in this problem w and b)
    gradient_function = a function that takes in a vector x and outputs gradient evaluated at that point
    eta  = the learning rate for gradient descent
    delta = stopping condition; stop if all entries in gradient change by less than delta in an iteration.

    '''
    x = x_init
    grad = gradient_function(x)
    it = 0
    while np.amax(np.abs(grad)) > delta:
        # perform a step in gradient descent

        it += 1
        print(it)
        x = x - eta * grad

        print(J_function(X, y, x))
        grad = gradient_function(x)

    # x is the best variable values
    return x



# =====================================================================================================================================

# Load MNIST Data
mndata = MNIST('./data')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())
X_train = X_train / 255.0
X_test = X_test / 255.0


#convert training labels to one hot
# i.e. encode an i as [0, 0, ..., 0, 1, 0, ... 0] where only the ith entry is nonzero
Y_train = np.zeros((X_train.shape[0], 10))
for i, digit in enumerate(labels_train):
        Y_train[i, digit] = 1
# =====================================================================================================================================

# Problem 3c)
n = X_train.shape[0]
d = X_train.shape[1] # 28 x 28 = 784
k = 10 # 10 classes, one for each digit


# define initial vector and the gradient function
W_init_L = np.zeros((d, k))
W_init_J = np.zeros((d, k))
gradient_function_J = lambda W: grad_J(X=X_train, y=Y_train, W=W, lam=lam)
gradient_function_L = lambda W: grad_L(X=X_train, y=Y_train, W=W)
delta = 1e-5
eta = 0.01 # learning rate
lam = 0

J_best_train = gradient_descent(W_init_J, gradient_function_J, eta, delta, X=X_train, y=Y_train)
#L_best_train = gradient_descent(W_init_L, gradient_function_L, eta, delta)

J_training_error_rate = error_rate(X_train, labels_train, J_best_train)
J_testing_error_rate = error_rate(X_test, labels_test, J_best_train)

print("J train: ", J_training_error_rate)
print("J test: ", J_testing_error_rate)

#L_training_error_rate = error_rate(X_train, labels_train, L_best_train)
#L_testing_error_rate = error_rate(X_test, labels_test, L_best_train)

