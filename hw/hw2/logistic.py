'''
logistic.py

Given a dataset with a binary response variable, this file has code capable of running
Binary Logistic Regression.

'''

import numpy as np
import random
import matplotlib.pyplot as plt
from mnist.loader import MNIST

def grad_J(X, y, w, b, lam):
    '''
    returns a vector that represents the gradient of J at the given points w,b

    X = data matrix with rows as measurements and columns are features (n x d)
    y = response variable (n x 1)
    w = weights vector (d x 1)
    b = bias term (1 x 1)
    lam = lambda value used for regularization
    '''

    n = y.size

    # y is a (n, ) dim vector instead of a (n, 1) vector. We adjust the dims so the elementwise product
    # performed below behaves as expected
    y = np.expand_dims(y, axis=1)

    temp = b + np.dot(w.T, X.T)

    # temp.T as current temp is a row vector
    mu = 1 / (1 + np.exp(np.multiply(-1*y, temp.T)))

    nabla_w = 1/n * np.dot(np.multiply(-y, X).T, 1-mu) + 2 * lam * w
    nabla_b = 1/n * np.dot(-y.T, 1-mu)

    return np.vstack((nabla_w, nabla_b))

def J_function(X, y, w,b, lam):
    '''
    Calculates the value of the J(w,b) the (normalized) logistic error function

    X = data matrix with rows as measurements and columns are features (n x d)
    y = response variable (n x 1)
    w = weights vector (d x 1)
    b = bias term (1 x 1)
    lam = lambda value used for regularization
    '''

    inside = np.multiply(-y, b + np.dot(w.T, X.T))
    n = y.size

    # squeeze removes the unnecessary dimensions: i.e. (25,1,1) --> (25,)
    return np.squeeze(1/n * np.sum(np.log(1 + np.exp(inside))) + lam * np.dot(w.T, w))

def error_rate(X, y, w, b):
    '''
    Uses the given weights w and bias b to make a classification sign(wTx + b) for each data
    measurement x in X (these measurements are stored as rows). Compares these against the true
    label stored in y to calculate an error rate.

    X = data matrix with rows as measurements and columns are features (n x d)
    y = response variable (n x 1)
    w = weights vector (d x 1)
    b = bias term (1 x 1)
    lam = lambda value used for regularization
    '''

    n = y.size
    predictions = np.sign(b + np.dot(w.T, X.T))

    # since np.sign(0) returns zero, we have to choose to set these to 1 or -1. Arbitrarily, we choose 1
    # Here if the element is zero, we replace it with 1, else we take whatever was in predictions.
    predictions = np.where(predictions == 0, 1, predictions)

    # NOTE: if we add the predictions and the labels, each correct prediction will be either -2 or 2 and each
    # incorrect prediction will be zero as all labels/guesses are -1 or 1. Thus the error rate will be:
    #  1 - sum of abs((predictions + labels)) / (2*total)
    error = 1 - np.sum( np.abs(predictions + y) ) / (2*n)
    return error


def gradient_descent(x_init, gradient_function, eta=0.1, delta=10e-4):
    '''
    Runs gradient descent to calculate minimizer x of the function whose gradient
    is defined by the given gradient_function.

    x_init = [w,b] is the initial values to set to the vector being descended on (in this problem w and b)
    gradient_function = a function that takes in a vector x and outputs gradient evaluated at that point
    eta  = the learning rate for gradient descent
    delta = stopping condition; stop if all entries in gradient change by less than delta in an iteration.

    '''
    x = x_init
    all_xs = [x]
    grad = gradient_function(x)
    while np.max(np.abs(grad)) > delta:
        # perform a step in gradient descent

        x = x - eta * grad
        grad = gradient_function(x)
        all_xs.append(x)
    
    # x is the best variable values; all_xs shows x value at each iteration
    return (x, all_xs)


def SGD(X, y, x_init, gradient_function, batch_size, eta=0.1, iteration_num=100):
    '''
    Runs STOCHASTIC gradient descent to calculate minimizer x of the given function whose
    gradient is defined by gradient_function.

    x_init = [w,b] is the initial values to set to the vector being descended on (in this problem w and b)
    gradient_function = a function that takes in a vector x and outputs gradient evaluated at that point
    batch_size = size of mini-batches used for SGD to approximate gradient across whole dataset
    eta  = the learning rate for gradient descent
    iteration_num = number of iterations (updates of x) to perform before returning the current x
    '''

    x = x_init
    n = y.size
    all_xs = [x]

    itr = 0
    while itr < iteration_num:
        # Loop through full dataset once, performing a round of updates

        train_order = list(range(n))
        random.shuffle(train_order)

        for batch_num in range(n // batch_size):

            itr += 1
            sample = train_order[batch_num * batch_size : (batch_num + 1) * batch_size]
            X_batch = X[sample, :]
            y_batch = y[sample]

            grad = gradient_function(X_batch=X_batch, y_batch=y_batch, x=x)
            x = x - eta * grad

            all_xs.append(x)
            if itr > iteration_num: break


        # handle case where there is one uneven batch left
        else:
            if n % batch_size != 0:
                itr += 1
                sample = train_order[(n // batch_size) * batch_size : ]
                X_batch = X[sample ,:]
                y_batch = y[sample]

                grad = gradient_function(X_batch, y_batch, x)
                x = x - eta * grad

                all_xs.append(x)

    return (x, all_xs)



# =====================================================================================================================================

# Load MNIST Data and filter for 2's and 7's
mndata = MNIST('./data')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())
X_train = X_train / 255.0
X_test = X_test / 255.0

# Filter out all digits besides two or seven
X_train = X_train[(labels_train == 2) + (labels_train == 7), :]
X_test = X_test[(labels_test == 2) + (labels_test == 7), :]

# filter out all digits besides two and seven. Convert label
# for a 7 to 1 and a 2 to -1
Y_train = labels_train[(labels_train == 2) + (labels_train == 7)]
Y_train = np.where(Y_train == 7, 1, -1)
Y_test = labels_test[(labels_test == 2) + (labels_test == 7)]
Y_test = np.where(Y_test == 7, 1, -1)

# =====================================================================================================================================

n = X_train.shape[0]
d = X_train.shape[1] # 28 x 28 = 784
lam = 0.1

# define initial vector and the gradient function
x_init = np.zeros((d+1, 1)) # = [w, b]
gradient_function_train = lambda x: grad_J(X_train, Y_train, x[:-1], x[-1], lam)
delta = 0.01
eta = 0.1 # learning rate

(x_best_train, all_xs_train) = gradient_descent(x_init, gradient_function_train, eta, delta)


# Problem 9b part i)

plt.figure(1)
plt.plot([J_function(X=X_train, y=Y_train, w=x[:-1], b=x[-1], lam=lam) for x in all_xs_train])
plt.plot([J_function(X=X_test, y=Y_test, w=x[:-1], b=x[-1], lam=lam) for x in all_xs_train]) # use parameters learned from training dataset
plt.title('Function Value at each iteration')
plt.ylabel('J(w,b)')
plt.xlabel('Iteration Number')
plt.legend(['Training', 'Testing'])
#plt.show()

# Problem 9b part ii)

plt.figure(2)
# skips plotting the error rate at iteration zero with w = 0
plt.plot([error_rate(X=X_train, y=Y_train, w=x[:-1], b=x[-1]) for x in all_xs_train[1:]])
plt.plot([error_rate(X=X_test, y=Y_test, w=x[:-1], b=x[-1]) for x in all_xs_train[1:]]) # use parameters learned from training dataset
plt.title('Error rate at each iteration')
plt.ylabel('Error rate')
plt.xlabel('Iteration Number')
plt.legend(['Training', 'Testing'])
plt.show()



# =====================================================================================================================================

# Problem 9c)

n = X_train.shape[0]
d = X_train.shape[1] # 28 x 28 = 784
x_init = np.zeros((d+1, 1)) # = [w, b]
batch_size = 1
eta = 0.01 # learning rate 0.00005, delta = 0.01
lam = 0.1

num_iterations = 200

# This time, the gradient function accepts some data as an input (i.e. the batch)
sgd_grad_function = lambda X_batch, y_batch, x: grad_J(X_batch, y_batch, x[:-1], x[-1], lam)
(x_best_sgd, all_xs_sgd) = SGD(X=X_train, y=Y_train, x_init=x_init, gradient_function=sgd_grad_function, \
                               batch_size=batch_size, eta=eta, iteration_num=num_iterations)

# 9ci)

plt.figure(3)
plt.plot([J_function(X=X_train, y=Y_train, w=x[:-1], b=x[-1], lam=lam) for x in all_xs_sgd])
plt.plot([J_function(X=X_test, y=Y_test, w=x[:-1], b=x[-1], lam=lam) for x in all_xs_sgd]) # use parameters learned from training dataset
plt.title('SGD Function Value at each iteration (batch_size = 1)')
plt.ylabel('J(w,b)')
plt.xlabel('Iteration Number')
plt.legend(['Training', 'Testing'])
#plt.show()


# 9cii)

plt.figure(4)
# skips plotting the error rate at iteration zero with w = 0
plt.plot([error_rate(X=X_train, y=Y_train, w=x[:-1], b=x[-1]) for x in all_xs_sgd[1:]])
plt.plot([error_rate(X=X_test, y=Y_test, w=x[:-1], b=x[-1]) for x in all_xs_sgd[1:]]) # use parameters learned from training dataset
plt.title('SGD Error rate at each iteration (batch_size = 1)')
plt.ylabel('Error rate')
plt.xlabel('Iteration Number')
plt.legend(['Training', 'Testing'])
plt.show()


# =====================================================================================================================================

# Problem 9d)
n = X_train.shape[0]
d = X_train.shape[1] # 28 x 28 = 784
x_init = np.zeros((d+1, 1)) # = [w, b]
batch_size = 100
eta = 0.1 # learning rate 0.00005, delta = 0.01
lam = 0.1

num_iterations = 150


# This time, the gradient function accepts some data as an input (i.e. the batch)
sgd_grad_function = lambda X_batch, y_batch, x: grad_J(X_batch, y_batch, x[:-1], x[-1], lam)
(x_best_sgd, all_xs_sgd100) = SGD(X=X_train, y=Y_train, x_init=x_init, gradient_function=sgd_grad_function, \
                                  batch_size=batch_size, eta=eta, iteration_num=num_iterations)

# 9di)

plt.figure(5)
plt.plot([J_function(X=X_train, y=Y_train, w=x[:-1], b=x[-1], lam=lam) for x in all_xs_sgd100])
plt.plot([J_function(X=X_test, y=Y_test, w=x[:-1], b=x[-1], lam=lam) for x in all_xs_sgd100]) # use parameters learned from training dataset
plt.title('SGD Function Value at each iteration (batch_size = 100)')
plt.ylabel('J(w,b)')
plt.xlabel('Iteration Number')
plt.legend(['Training', 'Testing'])
#plt.show()


# 9dii)

plt.figure(6)
# skips plotting the error rate at iteration zero with w = 0
plt.plot([error_rate(X=X_train, y=Y_train, w=x[:-1], b=x[-1]) for x in all_xs_sgd100[1:]])
plt.plot([error_rate(X=X_test, y=Y_test, w=x[:-1], b=x[-1]) for x in all_xs_sgd100[1:]]) # use parameters learned from training dataset
plt.title('SGD Error rate at each iteration (batch_size = 100)')
plt.ylabel('Error rate')
plt.xlabel('Iteration Number')
plt.legend(['Training', 'Testing'])
plt.show()

