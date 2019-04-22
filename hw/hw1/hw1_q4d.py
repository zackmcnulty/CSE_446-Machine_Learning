import numpy as np
import matplotlib.pyplot as plt
import time

n = 256
variance = 1
f_x = lambda x: 4*np.sin(np.pi * x)*np.cos(6*np.pi*x**2)

all_errors = []
all_biases = []
all_variances = []
all_total = []

m_vals = [1,2,4,8,16,32]

# generate the relevant random data
x_vals = [i/n for i in range(1,n+1)]
y_vals = [f_x(x) + np.random.randn() for x in x_vals] # can just use randn as variance == 1

for m in m_vals:
    empirical_error = 0
    bias_squared = 0

#    fm_vals = []    

    # for each interval {1, .., m}, {m+1, ..., 2m} --> {0, 1, ..., m-1}, {m, ..., 2m-1} due to pythons
    # zero based indexing
    # indices differ by one due to python 0 based indexing

    for j in range(1, n//m +1):

        # calculate the average value of the data, y_i,  over the jth interval
        fm = 1/m * sum(y_vals[(j-1)*m : j*m]) # wont include j*m

        # calculate the true average value of the function, f(x), over the jth interval
        fj = 1/m * sum([f_x(x) for x in x_vals[(j-1)*m: j*m]])

        empirical_error += sum([(fm - f_x(x))**2 for x in x_vals[(j-1)*m : j*m]])
        bias_squared += sum([(fj - f_x(x))**2 for x in x_vals[(j-1)*m: j*m]])

#        fm_vals.extend([fm]*m)
        

    average_empirical_error = 1/n * empirical_error
    all_errors.append(average_empirical_error)

    average_variance = variance / m
    all_variances.append(average_variance)

    average_bias_squared = 1/n*bias_squared
    all_biases.append(average_bias_squared)

    # plot sum of the average bias and average variance
    all_total.append(average_variance + average_bias_squared)

    # TESTING CODE =======================
    '''
    plt.plot(x_vals, y_vals, 'k.')
    plt.plot(x_vals, [f_x(x) for x in x_vals], 'b-')
    plt.plot(x_vals, fm_vals, 'r-')
    plt.legend(['data', 'true f(x)', 'Estimate f_16(x)'])
    plt.show()

    time.sleep(100)
    '''
    # ====================================


plt.plot(m_vals, all_biases, 'ro--')
plt.plot(m_vals, all_errors, 'bo--')
plt.plot(m_vals, all_variances, 'ko--')
plt.plot(m_vals, all_total, 'go--')
plt.xlabel('m')
plt.ylabel('Value')
plt.legend(['average bias-squared', 'average empirical error', 'average variance', 'average error = variance + bias-squared'])
plt.show()
