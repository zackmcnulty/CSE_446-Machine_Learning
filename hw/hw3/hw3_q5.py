
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_data(n):
    '''
    Generates n random datapoints with the x values distributed uniformly on [0,1]. 
    y = f(x) + N(0,1)
    '''
    f = lambda x: 4*np.sin(np.pi*x)*np.cos(6*np.pi* x ** 2)
    x = np.random.uniform(low=0, high=1, size=(n, ))
    error = np.random.normal(size=(n, ))
    y = f(x) + error
    return (x,y)

# ================================
# Problem 5a)




def find_K(kernel_function, x, n):
    return np.fromfunction(lambda i, j: kernel_function(x[i], x[j]), shape=(n,n), dtype=int)


def kernel_ridge_regress(x, y, kernel_function, lam_vals, hyperparam_vals, n):
    '''
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
                K = find_K(kernel_function=kf, x=x_train, n=n-1)

                alpha_hat = np.linalg.solve(K + lam * np.eye(n-1), y_train)

                f = lambda x: sum([alpha_hat[i] * kf(x, x_train[i]) for i in range(n-1)])
                error_loo += (y_val - f(x_val)) ** 2

            error_loo = error_loo / n
            errors[row, col] = error_loo

    print("done!")

    fig = plt.figure(1)
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(lam_vals, d_vals)
    ax.plot_surface(X, Y, errors.T)
    plt.xlabel('Lambda Value')
    plt.ylabel('d value')
    ax.set_zlabel('Error (leave one out)')
    plt.show()

    lam_ind, d_ind = np.unravel_index(np.argmin(errors, axis=None), errors.shape)
    return (lam_vals[lam_ind], d_vals[d_ind])


# part i) Polynomial Kernel
n = 30
x,y = generate_data(n)

lam_vals = list(range(1,75,5))  # all lambda values of interest
d_vals = list(range(1, 15, 1))
kf = lambda x, xprime, d: (1 + x * xprime)**d
best_lam, best_d = kernel_ridge_regress(x=x, y=y, kernel_function=kf, lam_vals=lam_vals, hyperparam_vals=d_vals, n=n)

print("Polynomial Kernel Function \n")
print("best lambda: ", best_lam, " best d: ", best_d)

# part ii) RBF Kernel

lam_vals = [1,2,3]  # all lambda values of interest
gamma_vals = [1,2,3]
kf = lambda x, xprime, gamma: np.exp(-gamma * (x - xprime)**2)
best_lam, best_gamma = kernel_ridge_regress(x=x, y=y, kernel_function=kf, lam_vals=lam_vals, hyperparam_vals=gamma_vals, n=n)

print("\n\nRBF Kernel Function \n")
print("best lambda: ", best_lam, " best d: ", best_gamma)
