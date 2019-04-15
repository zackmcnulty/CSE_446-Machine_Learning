import numpy as np
import matplotlib.pyplot as plt
import math

n = 256
variance = 1
f_x = lambda x: 4*math.sin(math.pi * x)*math.cos(6*math.pi*x**2)

all_errors = []
all_biases = []
all_variances = []
all_total = []

m_vals = [1,2,4,6,8,16,32]

for m in m_vals:
    empirical_error = 0
    bias_squared = 0

    x_vals = [i/n for i in range(1,n+1)]
    y_vals = [f_x(x) + np.random.randn() for x in x_vals]
    
    # for each interval {1, .., m}, {m+1, ..., 2m}
    # calculate the value assigned to that interval fm
    for j in range(1, n//m +1):
        # indices differ due to python 0 based indexing
        fm = 1/m * sum(y_vals[j*m : (j+1)*m])
        fj = 1/m * sum([f_x(x) for x in x_vals[(j-1)*m: j*m]])
        empirical_error += sum((fm - f_x(x))**2 for x in x_vals[j*m : (j+1)*m])
        bias_squared += sum((fj - f_x(x))**2 for x in x_vals[(j-1)*m: j*m])
        

    average_empirical_error = 1/n * empirical_error
    all_errors.append(average_empirical_error)

    average_variance = variance / m
    all_variances.append(average_variance)

    average_bias_squared = 1/n*bias_squared
    all_biases.append(average_bias_squared)

    all_total.append(average_variance + average_empirical_error + average_bias_squared)


plt.plot(m_vals, all_biases)
plt.plot(m_vals, all_errors)
plt.plot(m_vals, all_variances)
plt.plot(m_vals, all_total)

plt.legend(['average bias-squared', 'average empirical error', 'average variance', 'average error'])
plt.show()
