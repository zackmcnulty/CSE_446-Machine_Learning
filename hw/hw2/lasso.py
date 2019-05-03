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


def generate_synthetic_data(n, d, k, variance, seed=123):
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



def lasso_coordinate_descent(X, y, lam, w_init = None, delta = 10e-4):
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
    if w_init is None:
        w = np.zeros((d,))
    else:
        w = w_init

    c = np.zeros((d,))

    # a_k is just summing over all the rows of the squared entries. Thus, we can calculate it
    # quickly by squaring every entry in X (elementwise) then summing along the rows
    a = 2*np.sum(np.square(X), axis=0)

    # while we still have elements in w that changed by more than delta in last iteration
    while np.max(np.abs(w - prev_w)) > delta:

        prev_w = np.copy(w)

        wTXT = np.dot(w.T, X.T)

        b = 1/n * np.sum(y - wTXT)

        for k in range(d):

            # NOTE: Can use updated w_k values from SAME ROUND OF ITERATION piazza @199
            #       For this reason, we have to recalculate wTXT at every iteration
            c[k] = 2*np.dot(X[:, k], y - (b + np.dot(w.T, X.T) - w[k]*X[:, k]))

            if c[k] < -1*lam:
                w[k] = (c[k] + lam) / a[k]
            elif c[k] > lam:
                w[k] = (c[k] - lam) / a[k]
            else:
                w[k] = 0

    return (w, b)


 # ===============================================================================

 # problem 7a)


n = 500
d = 1000
k = 100
variance = 1
(X, y) = generate_synthetic_data(n, d, k, variance, seed=3490853)

lam_max = min_null_lambda(X, y)
lam_ratio = 1.5 # ratio to decrease lambda by during each iteration
delta = 0.01 # threshold to stop iteration (search for better w)

current_lam = lam_max
lam_vals = [lam_max]

W = np.zeros((d, 1)) # we will store the resulting w for each lambda as a column in this matrix
while np.count_nonzero(W[:, -1]) != d:
    current_lam = current_lam / lam_ratio
    lam_vals.append(current_lam)

    print("Running using lambda = ", current_lam)

    (w_new, b) = lasso_coordinate_descent(X, y, current_lam, delta=delta)
    W = np.concatenate((W, np.expand_dims(w_new, axis=1)), axis=1)

plt.figure(1)
plt.semilogx(lam_vals, np.count_nonzero(W, axis=0), 'r-')
plt.xlabel('Lambda')
plt.ylabel('Nonzero Coefficients in w')
plt.title('7a: Nonzero Coefficients versus Lambda')
plt.show()

# Problem 7b)

# Based on the definition of w_true, only the first k entries in w are truly nonzero. To calculate the
# FDR rate all we have to do is count the nonzero entries in the other d-k slots as incorrect
# Here we skip the first column of W as it corresponds to the w found using lambda_max, which generates
# a w with all zeros. To avoid division by zero, we define FDR = 0 at this point
FDR = np.append([0], np.count_nonzero(W[k:, 1:], axis=0) / np.count_nonzero(W[:,1:], axis=0))

# Based on the definition of w_true only the first k entries in w are truly nonzero.
TPR = np.count_nonzero(W[:k, :], axis=0) / k

plt.figure(2)
plt.plot(FDR, TPR)
plt.title('7b: False Discoveries and True Positives')
plt.xlabel('FDR')
plt.ylabel('TPR')
plt.show()


# ========================================================================================
# Problem 8)


import pandas as pd
df_train = pd.read_table("data/crime-train.txt")
df_test = pd.read_table("data/crime-test.txt")

y_train = df_train["ViolentCrimesPerPop"].values
X_train = df_train.drop("ViolentCrimesPerPop", axis=1).values
y_test = df_test["ViolentCrimesPerPop"].values
X_test = df_test.drop("ViolentCrimesPerPop", axis=1).values


lam_max = min_null_lambda(X_train, y_train)
lam_ratio = 2 # factor to decrease lambda by after each iteration

current_lam = lam_max
lam_vals = []
delta = 0.01 # threshold used to determine when to stop searching for optimal w
prev_w = None

# matrix with each column the weights vector generated by the corresponding lambda
W = None
B = []
while current_lam >= 0.01:

    lam_vals.append(current_lam)

    (w, b) = lasso_coordinate_descent(X_train, y_train, current_lam, prev_w, delta)

    if W is None:
        W = np.expand_dims(w, axis=1)
    else:
        W = np.concatenate((W, np.expand_dims(w, axis=1)), axis=1)

    B.append(b)

    prev_w = w
    current_lam /= lam_ratio

# Part a: Number non-zero entries vs lambda
plt.figure(3)
plt.semilogx(lam_vals, np.count_nonzero(W, axis=0), 'r-')
plt.xlabel('Lambda')
plt.ylabel('Nonzero Coefficients in w')
plt.title('8a: Nonzero coefficients vs Lambda')
#plt.show()

# Part b: Regularization Paths: agePct12t29, pctWSocSec, pctUrban, agePct65up, householdsize

# find where these columns reside; - 1 to account for the fact that the first column of df_train is our response
# variable y and thus has no associated weight
i1 = np.where(df_train.columns == "agePct12t29")[0] - 1
i2 = np.where(df_train.columns == "pctWSocSec")[0] - 1
i3 = np.where(df_train.columns == "pctUrban")[0] - 1
i4 = np.where(df_train.columns == "agePct65up")[0] - 1
i5 = np.where(df_train.columns == "householdsize")[0] - 1


k = len(lam_vals)

plt.figure(4)
plt.semilogx(lam_vals, np.reshape(W[i1, :], (k, )), \
             lam_vals, np.reshape(W[i2, :], (k,)), \
             lam_vals, np.reshape(W[i3, :], (k,)), \
             lam_vals, np.reshape(W[i4, :], (k,)), \
             lam_vals, np.reshape(W[i5, :], (k,)))

plt.xlabel('Lambda')
plt.ylabel('Coefficient Value')
plt.title('8b: Regularization Paths')
plt.legend(["agePct12t29", "pctWSocSec", "pctUrban", "agePct65up", "householdsize"])
#plt.show()

# Part c: Squared Error on training/test data

y_pred_train = np.dot(W.T, X_train.T) + np.expand_dims(B, axis=1)
SSE_train = 1/X_train.shape[0] * np.sum(np.square(y_pred_train - y_train), axis=1)
y_pred_test = np.dot(W.T, X_test.T) + np.expand_dims(B, axis=1)
SSE_test = 1/X_test.shape[0] * np.sum(np.square(y_pred_test - y_test), axis=1)

plt.figure(5)
plt.semilogx(lam_vals, SSE_train, lam_vals, SSE_test)
plt.legend(["Training Error", "Testing Error"])
plt.xlabel('Lambda')
plt.ylabel('SSE / n')
plt.title('8c: Squared Error as a function of Lambda')
plt.show()

# Part d:

(w30, _ ) = lasso_coordinate_descent(X_train, y_train, lam=30)

var_names = df_train.columns[1:] # skip the first varname corresponding to response variable ViolentCrimesPerPop
nonzero_coeffs = {w30[i]:var_names[i] for i in range(len(w30)) if not w30[i] == 0}
#nonzero_coeffs = {w30[i]:i for i in range(len(w30)) if not w30[i] == 0}
max_coeff = max(list(nonzero_coeffs.keys())) 
min_coeff = min(list(nonzero_coeffs.keys())) 
print(nonzero_coeffs)
print("feature with largest coefficient: ", nonzero_coeffs[max_coeff], " , value: ", max_coeff)
print("feature with smallest coefficient: ", nonzero_coeffs[min_coeff], " , value: ", min_coeff)


# Part e:

# Correlation does not imply causation. This could be an example of a third-variable
# problem where the given variable has a high correlation with something that is
# actually influencing the variable of interest (crime rate)
