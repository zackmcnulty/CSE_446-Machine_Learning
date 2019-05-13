
import numpy as np
import matplotlib.pyplot as plt


def generate_data(n):
    '''
    Generates n random datapoints with the x values distributed uniformly on [0,1]. 
    y = f(x) + N(0,1)
    '''
    f = lambda x: 4*np.sin(np.pi*x)*np.cos(6*np.pi* x ** 2)
    x = np.random.uniform(low=0, high=1, size=(n,1))
    error = np.random.normal(size=(n,1))
    y = f(x) + error
    return (x,y)



