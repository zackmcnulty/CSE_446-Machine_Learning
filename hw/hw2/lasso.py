'''
lasso.py

implements the coordinate descent algorithm for LASSO regression.
LASSO promotes sparsity by regularizing the Least Squared Error 
with an L1 norm.

X --> rows are data measurements, columns are specific features
y = vector of response variables

'''

import numpy as np
import matplotlib.pyplot as plt
import time


def generate_synthetic_data(n,d,k, variance, seed=123):
    '''
    Generates synthetic data following to procedure in the homework specification.
    n = number of data samples
    d = number of features per data sample
    k = number of relevant features (i.e. whose coefficient in w we expect to be non-negative)
    variance = variance for the zero-mean gaussian noise added on top of the true system
    seed = RNG seed for reproducibility

    '''
    if k > d:
        raise ValueError("The number of relevant features cannot be more than the number of features")
    elif n < 1 or d < 1 or k < 0 or variance < 0:
        raise ValueError("Parameters n,d >= 1 and k,variance >= 0 must be true")

    w_true = np.zeros((d ,1))
    
    # j+1 is to account for zero-based indexing in python
    for j in range(k): w_true[j] = (j+1) / k

    np.random.seed(seed)
    X = np.random.normal(size = (n,d))
    errors = np.random.normal(scale = np.sqrt(variance), size=(n,))

    y = np.reshape(np.dot(w_true.T, X.T) + errors.T, (n,))
    return (X, y)



def min_null_lambda(X, y):
    '''
    Returns the smallest lambda value that generates a null solution
    (i.e. w is entirely zeros). Start with lambda at this value and
    decrease over time
    Assuming y is a column vector of responses and X is a matrix with each column a feature
    and each row a measurement (i.e. the data matrix)
    '''
    return 2*np.max(np.abs(np.dot(y.T - np.mean(y), X)))



def lasso_coordinate_descent(X, y, lam, delta = 10e-4):
    '''
    Runs the coordinate descent algorithm on the LASSO regression problem.

    X = data matrix with each measurement stored as a row (columns are features) n x features
    y = response variable : n x 1
    lam = lambda value used for regularization
    delta = stopping condition; if no element in w changes by more than delta in a single iteration, stop.

    '''

    n = X.shape[0]
    d = X.shape[1]
    prev_w = np.ones((d,1))
    w = np.zeros((d,))
    c = np.zeros((d,))
    #c3 = np.zeros((d,))

    # a_k is just summing over all the rows of the squared entries. Thus, we can calculate it
    # quickly by squaring every entry in X (elementwise) then summing along the rows
    a =  2*np.sum(np.square(X), axis = 0)

    # while we still have elements in w that changed by more than delta in last iteration
    while np.max(np.abs(w - prev_w)) > delta:

        prev_w = np.copy(w)

        wTXT = np.dot(w.T, X.T)
        b = 1/n * np.sum(y - wTXT)

        #c2 = 2*(np.dot(X.T, y-b-wTXT) + np.dot(wTXT, X))
        
        for k in range(d):

            # NOTE: Can use updated w_k values from SAME ROUND OF ITERATION piazza @199
            # Makes the matrix calculated used above for c2 invalid
            c[k] = 2*np.dot(X[:, k], y - (b + np.dot(w.T, X.T) - w[k]*X[:,k]))

            #for i in range(n):
            #    c3[k] += 2*X[i, k] * (y[i] - b - sum([w[j]*X[i,j] for j in range(d) if not j == k]))

            
            if c[k] < -1*lam:
                w[k] = (c[k] + lam) / a[k]
            elif c[k] > lam:
                w[k] = (c[k] - lam) / a[k]
            else:
                w[k] = 0

    return w
            

        


 # ===============================================================================

 # problem 8a)

n = 500
d = 1000
k = 100
variance = 1
(X, y) = generate_synthetic_data(n,d,k, variance, seed=45)

lam_max = min_null_lambda(X,y)
lam_ratio = 1.5 # ratio to decrease lambda by during each iteration
delta = 0.01 # threshold to stop iteration (search for better w)

current_lam = lam_max
lam_vals = [lam_max]

W = np.zeros((d,1)) # we will store the resulting w for each lambda as a column in this matrix
while np.count_nonzero(W[:, -1]) != d:
    current_lam = current_lam / lam_ratio
    lam_vals.append(current_lam)

    print("Running using lambda = ", current_lam)

    w_new = lasso_coordinate_descent(X,y, current_lam, delta)
    W = np.concatenate((W, np.expand_dims(w_new, axis = 1)), axis = 1)

plt.figure(1)
plt.semilogx(lam_vals, np.count_nonzero(W, axis=0), 'r-')
plt.xlabel('Lambda')
plt.ylabel('Nonzero Coefficients in w')

