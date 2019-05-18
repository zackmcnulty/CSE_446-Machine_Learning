
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
    return np.fromfunction(lambda i, j: kernel_function(x[i], x[j]), shape=(n,n), dtype=int)


def kernel_ridge_regress(x, y, kf, lam, n):
    K = find_K(kernel_function=kf, x=x, n=n)

    alpha_hat = np.linalg.solve(K + lam * np.eye(n), y)

    f = lambda z: sum([alpha_hat[i] * kf(z, x[i]) for i in range(n)])
    return f

# Part a)
def error_loo_tunning(x, y, kernel_function, lam_vals, hyperparam_vals, n):
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
                f = kernel_ridge_regress(x=x_train, y=y_train, kf=kf, lam=lam, n=n-1)

                error_loo += (y_val - f(x_val)) ** 2

            error_loo = error_loo / n
            errors[row, col] = error_loo


    fig = plt.figure(1)
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(lam_vals, hyperparam_vals)
    ax.plot_surface(X, Y, errors.T)
    plt.xlabel('Lambda Value')
    plt.ylabel('hyperparameter value')
    ax.set_zlabel('Error (leave one out)')
    plt.show()

    lam_ind, h_ind = np.unravel_index(np.argmin(errors, axis=None), errors.shape)
    return (lam_vals[lam_ind], hyperparam_vals[h_ind])



# Part b)
def part_b(x,y, kf, lam, hyperparameter, n, kernel_name):
    plt.figure()
    f_true = lambda x: 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)
    f_hat = kernel_ridge_regress(x=x,y=y, lam=lam, kf=lambda x, xprime: kf(x, xprime, hyperparameter), n=n)

    x_fine = np.linspace(0,1, 1000)

    plt.plot(x, y, 'ko')
    plt.plot(x_fine, f_true(x_fine))

    y_kernel = [f_hat(xi) for xi in x_fine]
    plt.plot(x_fine, y_kernel)
    plt.legend(['Data', 'True', 'Kernel Ridge Regression'])
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title('Problem 5b: Kernel Ridge Regression (' + kernel_name + ')')
    plt.show()



# part i) Polynomial Kernel
n = 30
x, y = generate_data(n)

lam_vals = [10**k for k in np.linspace(-6, 1, 50)]  # all lambda values of interest
d_vals = list(range(1, 15, 1))
kf_p = lambda x, xprime, d: (1 + x * xprime) ** d
best_lam, best_d = error_loo_tunning(x=x, y=y, kernel_function=kf_p, lam_vals=lam_vals, hyperparam_vals=d_vals, n=n)

print("Polynomial Kernel Function \n")
print("best lambda: ", best_lam, " best d: ", best_d)

part_b(x,y, kf_p, best_lam, best_d, n, 'poly')


# part ii) RBF Kernel

gamma_ballpark = np.median([(x[i] - x[j]) ** 2 for i in range(n) for j in range(n)])
print(gamma_ballpark)

# use lambda values from above
gamma_vals = np.linspace(1e-5, 1, 200)
kf_rbf = lambda x, xprime, gamma: np.exp(-gamma * (x - xprime)**2)
best_lam, best_gamma = error_loo_tunning(x=x, y=y, kernel_function=kf_rbf, lam_vals=lam_vals, hyperparam_vals=gamma_vals, n=n)

print("\n\nRBF Kernel Function \n")
print("best lambda: ", best_lam, " best gamma: ", best_gamma)

part_b(x,y, kf_rbf, best_lam, best_gamma, n, 'RBF')



