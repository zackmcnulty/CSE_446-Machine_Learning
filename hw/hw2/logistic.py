'''
logistic.py

'''

import numpy as np
import matplotlib.pyplot as plt
from mnist.loader import MNIST

def grad_J(X, y, w, b,n, lam):
    '''
    returns a vector that represents the gradient of J at the given points w,b
    '''
    
    #nabla_w = 1/n * sum([ -y[i]*X[i, :] * (1/(1+np.exp(-y[i]*(b + X[i, :] * w)) - 1)) for i in range(n) ]) + 2 * lam * w
    #nabla_b = 1/n * sum([-y[i] * (1/(1+np.exp(-y[i]*(b + X[i, :] * w)) - 1)) for i in range(n)])

    # y is a (n, ) dim vector instead of a (n, 1) vector. We adjust the dims so the elementwise product
    # performed below behaves as expected
    y = np.expand_dims(y, axis=1)

    temp = b + np.dot(w.T, X.T)

    # temp.T as current temp is a row vector
    mu = 1 / (1 + np.exp(np.multiply(-1*y, temp.T)))

    # NOTE: Why do I need these extra minuses at the front?
    nabla_w = -1/n * np.dot(np.multiply(y, X).T, 1-mu) + 2 * lam * w
    nabla_b = -1/n * np.dot(y.T, 1-mu)

    return np.vstack((nabla_w, nabla_b))

def J_function(X, y, w,b, lam):
    '''
    Calculates the value of the J(w,b) logistic error function given data matrix X (rows are measurements
    and columns are features), labels y, weights w, bias b, and lambda value lam.
    '''

    inside = np.multiply(-y, b + np.dot(w.T, X.T))
    n = y.size
    print("n: ", n)

    # squeeze removes the unnecessary dimensions: i.e. (25,1,1) --> (25,)
    return np.squeeze(1/n * np.sum(np.log(1 + np.exp(inside))) + lam * np.dot(w.T, w))

def error_rate(X, y, w, b):
    '''
    Given data matrix X (rows are measurements and columns are features), labels y, weights w, and bias b this
    calculates the misclassification rate.
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
    Runs gradient descent to calculate optimum 
    x_init is the initial values to set to the vector being descended on (in this problem w and b)
    gradient_function is a function that takes in a vector x and outputs its gradient
    eta is the learning rate
    delta is the stopping condition; stop if all entries in gradient less than delta.
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

def SGD(X, y, x_init, gradient_function, batch_size, eta=0.1, delta=10e-4):
    '''
    Runs STOCHASTIC gradient descent to calculate optimum
    x_init is the initial values to set to the vector being descended on (in this problem w and b)
    gradient_function is a function that takes in a vector x and outputs its gradient
    eta is the learning rate
    delta is the stopping condition; stop if all entries in gradient less than delta.
    '''
    x = x_init
    all_xs = [x]
    grad = gradient_function(x)
    while np.max(np.abs(grad)) > delta:
        # perform a step in gradient descent

        print(np.max(np.abs(grad)))
        x = x - eta * grad
        grad = gradient_function(x)
        all_xs.append(x)

    return (x, all_xs)




# Load MNIST Data
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


n = X_train.shape[0]
d = X_train.shape[1] # 28 x 28 = 784
lam = 0.1

# define initial vector and the gradient function
x_init = np.zeros((d+1, 1)) # = [w, b]
gradient_function_train = lambda x: grad_J(X_train, Y_train, x[:-1], x[-1], n, lam)
delta = 0.01
eta = 0.1 # learning rate

(x_best_train, all_xs_train) = gradient_descent(x_init, gradient_function_train, eta, delta)


# Problem 10b part i)

plt.figure(1)
plt.plot([J_function(X=X_train, y=Y_train, w=x[:-1], b=x[-1], lam=lam) for x in all_xs_train])
plt.plot([J_function(X=X_test, y=Y_test, w=x[:-1], b=x[-1], lam=lam) for x in all_xs_train]) # use parameters learned from training dataset
plt.title('Function Value at each iteration')
plt.ylabel('J(w,b)')
plt.xlabel('Iteration Number')
plt.legend(['Training', 'Testing'])
#plt.show()

# Problem 10b part ii)

plt.figure(2)
plt.plot([error_rate(X=X_train, y=Y_train, w=x[:-1], b=x[-1]) for x in all_xs_train])
plt.plot([error_rate(X=X_test, y=Y_test, w=x[:-1], b=x[-1]) for x in all_xs_train]) # use parameters learned from training dataset
plt.title('Error rate at each iteration')
plt.ylabel('Error rate')
plt.ylim([0,1])
plt.xlabel('Iteration Number')
plt.legend(['Training', 'Testing'])
plt.show()



# Problem 10c)

batch_size = 1
sgd_grad_function = lambda X_batch, y_batch, x: grad_J()
(x_best_train, all_xs_train) = SGD(X=X_train, y=Y_train, )




# Problem 10d)
batch_size = 100
