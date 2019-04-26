'''
lasso.py

implements the coordinate descent algorithm for LASSO regression.
LASSO promotes sparsity by regularizing the Least Squared Error 
with an L1 norm.

X --> rows are data measurements, columns are specific features

'''

import numpy as np
import matplotlib.pyplot as plt


def min_null_lambda(x_vals, y_vals):
    '''
    Returns the smallest lambda value that generates a null solution
    (i.e. w is entirely zeros). Start with lambda at this value and
    decrease over time
    '''
    y_mean = mean(y_vals)

    return 2*np.max()

