import matplotlib.pyplot as plt
import numpy as np

# Problem 7

def plot_mvg(mu, Sigma, n):
    '''
    plots n datapoints drawn from the multivariate gaussian
    specified by the parameters mu and Sigma
    '''
    L, Q = np.linalg.eig(Sigma)

    # L is returned as an array and NOT a diagonal matrix, so
    # just convert it to a matrix.
    L = np.diag(L)
    A = np.matmul(np.matmul(Q, np.sqrt(L)), Q.T)

    Z = np.random.randn(2, n)
    X =  np.matmul(A, Z) + mu

    plt.figure(1)
    plt.plot(X[0, :], X[1, :], '^')
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.title('Problem 7a')
    plt.show()
    
    return X


def calc_mean_and_cov(X, n):
    '''
    Calculates an estimate of mu and the covariance matrix for the given data
    '''

    mu_hat = 1/n * np.sum(X, axis=1)
    mu_hat = np.expand_dims(mu_hat, axis=1)
    

    temp = X - mu_hat
    Sigma_hat = 1/(n-1) * sum([np.outer(temp[:, i], temp[:, i]) for i in range(n)])

    L, Q = np.linalg.eig(Sigma_hat)

    start = mu_hat
    end1 = mu_hat + np.expand_dims(np.sqrt(L[0]) * Q[:, 0], axis=1)
    end2 = mu_hat + np.expand_dims(np.sqrt(L[1]) * Q[:, 1], axis=1)
    points1 = np.hstack((start, end1))
    points2 = np.hstack((start, end2))

    plt.figure(1)
    plt.plot(mu_hat[0], mu_hat[1], 'ro')
    plt.plot(points1[0, :], points1[1, :])
    plt.plot(points2[0, :], points2[1, :])

    r = max(np.amax(np.abs(points1)), np.amax(np.abs(points2))) + 1
    limits = [-r, r]
    plt.xlim(limits)
    plt.ylim(limits)
    plt.legend(['mu', 'Eigenvector 1 (Lambda = ' + str(L[0]) + ' )', 'Eigenvector 2 (Lambda = ' + str(L[1]) + ' )',])
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.title('Problem 7b')

    plt.show()


    return mu_hat, Sigma_hat, L, Q


def part_c(X, mu_hat, L, Q):
    row1 = 1/np.sqrt(L[0]) * np.dot(Q[:, 0], X - mu_hat)
    row2 = 1/np.sqrt(L[1]) * np.dot(Q[:, 1], X - mu_hat)

    r = 1 + max(np.amax(np.abs(row1)), np.amax(np.abs(row2)))

    plt.figure(3)
    plt.plot(row1, row2, 'ro')
    plt.xlim([-r, r])
    plt.ylim([-r, r])
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.title('Problem 7c')
    plt.show()


# Part a) and b) and c)
n = 100
mu1 = np.asarray([[1], [2]])
Sigma1 = np.asarray([[1, 0], [0, 2]])
X1 = plot_mvg(mu1, Sigma1, n)
mu1_hat, Sigma1_hat, L1, Q1 = calc_mean_and_cov(X1, n)
part_c(X1, mu1_hat, L1, Q1)


mu2 = np.asarray([[-1], [1]])
Sigma2 = np.asarray([[2, -1.8], [-1.8, 2]])
X2 = plot_mvg(mu2, Sigma2, n)
mu2_hat, Sigma2_hat, L2, Q2 = calc_mean_and_cov(X2, n)
part_c(X2, mu2_hat, L2, Q2)

mu3 = np.asarray([[2], [-2]])
Sigma3 = np.asarray([[3, 1], [1, 2]])
X3 = plot_mvg(mu3, Sigma3, n)
mu3_hat, Sigma3_hat, L3, Q3 = calc_mean_and_cov(X3, n)
part_c(X3, mu3_hat, L3, Q3)


