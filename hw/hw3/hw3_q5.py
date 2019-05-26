
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_data(n):
    '''
    Generates n random datapoints with the x values distributed uniformly on [0,1]. 
    y = f(x) + N(0,1)
    '''

    #np.random.seed(1234) # for reproducibility

    f = lambda x: 4*np.sin(np.pi*x)*np.cos(6*np.pi* x ** 2)
    x = np.random.uniform(low=0, high=1, size=(n, ))
    error = np.random.normal(size=(n, ))
    y = f(x) + error
    return (x,y)

# ================================================================================
# Problem 5a)




def find_K(kernel_function, x, n):
    '''
    Finds the kernel matrix K for the specified data (x) and kernel function
    '''
    return np.fromfunction(lambda i, j: kernel_function(x[i], x[j]), shape=(n,n), dtype=int)


def kernel_ridge_regress(x, y, kf, lam, n):
    '''

    Returns the function f obtained via kernelized ridge regression

    '''
    K = find_K(kernel_function=kf, x=x, n=n)

    alpha_hat = np.linalg.solve(K + lam * np.eye(n), y)

    f = lambda z: np.sum(np.dot(alpha_hat, kf(z, x)))
    return f

# Part a)
def error_loo_tunning(x, y, kernel_function, lam_vals, hyperparam_vals, n):
    '''

    Runs Leave One Out cross-validation to calculate the optimal hyperparameters for
    kernel ridge regression using the specified kernel function.

    :param x: input data
    :param y: input labels
    :param kf: kernel function
    :param lam_vals: lambda values to cross validate at
    :param hyperparam_vals: hyperparameter (d or gamma) values to cross validate at
    :param n: number data points
    :return: best parameter choice
    '''

    errors = np.empty((len(lam_vals), len(hyperparam_vals)))
    for row, lam in enumerate(lam_vals):
        for col, hyperparam in enumerate(hyperparam_vals):
            error_loo = 0
            for j in range(n):
                x_val = x[j]
                y_val = y[j]
                x_train = np.concatenate((x[:j], x[j+1:]))
                y_train = np.concatenate((y[:j], y[j+1:]))

                kf = lambda x, xprime: kernel_function(x, xprime, hyperparam)
                f = kernel_ridge_regress(x=x_train, y=y_train, kf=kf, lam=lam, n=n-1)

                error_loo += (y_val - f(x_val)) ** 2

            error_loo = error_loo / n
            errors[row, col] = error_loo


    #fig = plt.figure(1)
    #ax = fig.gca(projection='3d')

    #X, Y = np.meshgrid(lam_vals, hyperparam_vals)
    #ax.plot_surface(X, Y, errors.T)
    #plt.xlabel('Lambda Value')
    #plt.ylabel('hyperparameter value')
    #ax.set_zlabel('Error (leave one out)')
    #plt.show()

    lam_ind, h_ind = np.unravel_index(np.argmin(errors, axis=None), errors.shape)
    return (lam_vals[lam_ind], hyperparam_vals[h_ind])



# Part b)
def part_b(x,y, kf, lam, hyperparameter, n, kernel_name):
    '''
    Plots the original data (x,y) and the function f_hat (determined through kernel ridge regression) evaluated
    at each point in a fine grid. Also plots the true distribution.
    '''
    plt.figure()
    f_true = lambda x: 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)

    x_fine, y_kernel = find_fhat(x=x,y=y,kf=kf,hyperparameter=hyperparameter, lam=lam, n=n)
    plt.plot(x, y, 'ko')
    plt.plot(x_fine, f_true(x_fine))

    plt.plot(x_fine, y_kernel)
    plt.legend(['Data', 'True', 'Kernel Ridge Regression'])
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title('Problem 5b: Kernel Ridge Regression (' + kernel_name + ')')
    plt.show()

def find_fhat(x,y,kf, hyperparameter, lam, n, num_points=1000):
    '''
    Returns the values of f_hat evaluated at each point in the fine grid
    '''


    f_hat = kernel_ridge_regress(x=x,y=y, lam=lam, kf=lambda x, xprime: kf(x, xprime, hyperparameter), n=n)

    x_fine = np.linspace(0,1, num_points)
    y_kernel = [f_hat(xi) for xi in x_fine]
    return x_fine, y_kernel

def part_c(x,y, kf, hyperparameter, lam, n, num_points=1000, B=300):
    '''

    Run bootstrap to generate an approximate range (5th percentile and 95th percentile) for the predictions of
    f_hat at each point in the fine grid (between [0,1] with num_points linearly spaced points)

    :param x: input data
    :param y: input labels
    :param kf: kernel function
    :param lam: lambda value to use for kernel ridge regression
    :param hyperparam: hyperparameter (d or gamma) to use for kernel ridge regression
    :param n: number data points
    :param num_points: number of points to use in fine grid to plot f_hat
    :param B: number of bootstrap samples to use
    '''
    fhat = np.zeros((B, num_points))

    for b in range(B):

        print(b)

        # sample from the data with replacement
        indices = np.random.choice(list(range(n)), n, replace=True)
        x_b = x[indices]
        y_b = y[indices]

        _ , fhat[b, :] = find_fhat(x=x_b, y=y_b, kf=kf, hyperparameter=hyperparameter, lam=lam, n=n, num_points=num_points)

    percentile_5 = np.percentile(fhat, q=5, axis=0)
    percentile_95 = np.percentile(fhat, q=95, axis=0)

    f_true = lambda x: 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)

    x_fine, y_kernel = find_fhat(x,y,kf,hyperparameter, lam, n)

    plt.figure()
    plt.plot(x, y, 'ko')
    plt.plot(x_fine, f_true(x_fine))
    plt.plot(x_fine, y_kernel)
    plt.plot(x_fine, percentile_5, 'r-')
    plt.plot(x_fine, percentile_95, 'b-')
    plt.legend(['data', 'True', 'f_hat', '5th percentile', '95th percentile'])
    plt.title('Problem 5c')
    plt.show()


def ten_fold_CV(x, y, kernel_function, lam_vals, hyperparam_vals, n):
    '''

    Runs 10-fold cross-validation to estimate the parameters of the given kernel function that
    minimize the validation error

    :param x: input data
    :param y: input labels
    :param kf: kernel function
    :param lam_vals: lambda values to cross validate at
    :param hyperparam_vals: hyperparameter (d or gamma) values to cross validate at
    :param n: number data points
    :return: best parameter choice
    '''

    errors = np.empty((len(lam_vals), len(hyperparam_vals)))
    for row, lam in enumerate(lam_vals):
        for col, hyperparam in enumerate(hyperparam_vals):
            error_CV = 0
            for j in range(10):
                start_val = j* (n//10)
                end_val = (j+1) * (n //10)
                x_val = x[start_val:end_val]
                y_val = y[start_val:end_val]
                x_train = np.concatenate((x[:start_val], x[end_val:]))
                y_train = np.concatenate((y[:start_val], y[end_val:]))

                kf = lambda x, xprime: kernel_function(x, xprime, hyperparam)
                f = kernel_ridge_regress(x=x_train, y=y_train, kf=kf, lam=lam, n=(n - n//10) )

                error_CV += 1/x_val.size * np.sum(np.square((y_val - np.asarray([f(x) for x in x_val]))))

            errors[row, col] = error_CV


    #fig = plt.figure(1)
    #ax = fig.gca(projection='3d')

    #X, Y = np.meshgrid(lam_vals, hyperparam_vals)
    #ax.plot_surface(X, Y, errors.T)
    #plt.xlabel('Lambda Value')
    #plt.ylabel('hyperparameter value')
    #ax.set_zlabel('Error (leave one out)')
    #plt.show()

    lam_ind, h_ind = np.unravel_index(np.argmin(errors, axis=None), errors.shape)
    return (lam_vals[lam_ind], hyperparam_vals[h_ind])

# =================================================================================================
'''
# parts a,b,c)
# part a is mostly accomplished by the function error_loo_tuning
# part b,c are accomplished by the part_b(), part_c() functions repspectively.

# part i) Polynomial Kernel
n = 30
x, y = generate_data(n)

lam_vals = [10**k for k in np.linspace(-6, 1, 25)]  # all lambda values of interest
d_vals = list(range(1, 20, 1))
kf_p = lambda x, xprime, d: np.power((1 + x * xprime), d)
best_lam, best_d = error_loo_tunning(x=x, y=y, kernel_function=kf_p, lam_vals=lam_vals, hyperparam_vals=d_vals, n=n)

print("Polynomial Kernel Function \n")
print("best lambda: ", best_lam, " best d: ", best_d)

part_b(x=x, y=y, kf=kf_p, lam=best_lam, hyperparameter=best_d, n=n, kernel_name='poly')
part_c(x=x, y=y, kf=kf_p, lam=best_lam, hyperparameter=best_d, n=n, num_points=1000, B=300)


# part ii) RBF Kernel

gamma_ballpark = 1 / np.median([(x[i] - x[j]) ** 2 for i in range(n) for j in range(n)])
print('gamma ballpark: ', gamma_ballpark) # find a ballpark estimate for gamma.

# use lambda values from above
gamma_vals = np.linspace(1, 25, 20)
kf_rbf = lambda x, xprime, gamma: np.exp(-gamma * np.power(x - xprime, 2))
best_lam, best_gamma = error_loo_tunning(x=x, y=y, kernel_function=kf_rbf, lam_vals=lam_vals, hyperparam_vals=gamma_vals, n=n)

print("\n\nRBF Kernel Function \n")
print("best lambda: ", best_lam, " best gamma: ", best_gamma)

part_b(x=x, y=y, kf=kf_rbf, lam=best_lam, hyperparameter=best_gamma, n=n, kernel_name='RBF')
part_c(x=x, y=y, kf=kf_rbf, hyperparameter=best_gamma, lam=best_lam, n=n, num_points=1000, B=300)



# =========================================================================================
# part d)

'''
# Polynomial Kernel
n = 300
x, y = generate_data(n)

lam_vals = [10**k for k in np.linspace(-6, 1, 20)]  # all lambda values of interest
d_vals = list(range(1, 20, 1))
kf_p = lambda x, xprime, d: np.power((1 + x * xprime), d)

best_lam_poly, best_d = ten_fold_CV(x=x, y=y, kernel_function=kf_p, lam_vals=lam_vals, hyperparam_vals=d_vals, n=n)
print("Polynomial Kernel Function \n")
print("best lambda: ", best_lam_poly, " best d: ", best_d)

part_b(x=x, y=y, kf=kf_p, lam=best_lam_poly, hyperparameter=best_d, n=n, kernel_name='poly')
part_c(x=x, y=y, kf=kf_p, lam=best_lam_poly, hyperparameter=best_d, n=n, num_points=1000, B=300)



# RBF Kernel
# use lambda values from above
gamma_vals = np.linspace(1, 25, 20)
kf_rbf = lambda x, xprime, gamma: np.exp(-gamma * np.power(x - xprime, 2))

best_lam_rbf, best_gamma = ten_fold_CV(x=x, y=y, kernel_function=kf_rbf, lam_vals=lam_vals, hyperparam_vals=gamma_vals, n=n)
print("\n\nRBF Kernel Function \n")
print("best lambda: ", best_lam_rbf, " best gamma: ", best_gamma)
part_b(x=x, y=y, kf=kf_rbf, lam=best_lam_rbf, hyperparameter=best_gamma, n=n, kernel_name='RBF')
part_c(x=x, y=y, kf=kf_rbf, hyperparameter=best_gamma, lam=best_lam_rbf, n=n, num_points=1000, B=300)



#best_lam_poly = 1e-6
#best_lam_rbf = 1.274e-5
#best_d = 9
#best_gamma = 21.21
#kf_rbf = lambda x, xprime, gamma: np.exp(-gamma * np.power(x - xprime, 2))
#kf_p = lambda x, xprime, d: np.power((1 + x * xprime), d)

# ===========================================
# part e)
m = 1000
xm, ym = generate_data(m)

f_hat_poly = kernel_ridge_regress(x=x,y=y, lam=best_lam_poly, kf=lambda x, xprime: kf_p(x, xprime, best_d), n=n)
f_hat_rbf = kernel_ridge_regress(x=x,y=y, lam=best_lam_rbf, kf=lambda x, xprime: kf_rbf(x, xprime, best_gamma), n=n)

B = 300

bootstrap_values = np.zeros((B, ))

for b in range(B):
    indices = np.random.choice(list(range(m)), m, replace=True)
    x_b = xm[indices]
    y_b = ym[indices]

    poly_error = np.square(y_b - np.asarray([f_hat_poly(x) for x in x_b]))
    rbf_error = np.square(y_b - np.asarray([f_hat_rbf(x) for x in x_b]))
    bootstrap_values[b] = 1/m * np.sum(poly_error - rbf_error)


percentile_95 = np.percentile(bootstrap_values, q=95)
percentile_5 = np.percentile(bootstrap_values, q=5)
print('95th percentile: ', str(percentile_95))
print('5th Percentile: ', str(percentile_5))

