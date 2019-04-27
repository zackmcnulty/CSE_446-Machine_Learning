# Problem 5


# GIVEN CODE
import numpy as np
import matplotlib.pyplot as plt

num_trials = 30

lam_vals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100] # lambda values

# These will be 2D arrays with each row representing a given lambda value and
# each column a trial at that lambda value
all_train_errors = np.zeros((len(lam_vals), num_trials))
all_test_errors = np.zeros((len(lam_vals), num_trials))

for j in range(num_trials): # run 30 trials
    train_n = 100
    test_n = 1000
    d = 100
    X_train = np.random.normal(0,1, size=(train_n,d))
    a_true = np.random.normal(0,1, size=(d,1))
    y_train = X_train.dot(a_true) + np.random.normal(0,0.5,size=(train_n,1))
    X_test = np.random.normal(0,1, size=(test_n,d))
    y_test = X_test.dot(a_true) + np.random.normal(0,0.5,size=(test_n,1))


# ADDED CODE =========================================

    # lam = lambda values
    for i, lam in enumerate(lam_vals):
        Xt_X_plus_lambda = np.dot(X_train.T, X_train) + lam * np.eye(d,d) 
        Xt_y = np.dot(X_train.T, y_train) 
        w_hat = np.linalg.solve(Xt_X_plus_lambda, Xt_y)

        test_error = np.linalg.norm(np.dot(X_test, w_hat) - y_test) / np.linalg.norm(y_test)
        train_error = np.linalg.norm(np.dot(X_train, w_hat) - y_train) / np.linalg.norm(y_train)

        all_test_errors[i, j] = test_error
        all_train_errors[i,j] = train_error

# calculate the average along each row (each lambda value)
ave_train_errors = np.mean(all_train_errors, axis=1)
ave_test_errors = np.mean(all_test_errors, axis=1)

plt.semilogx(lam_vals, ave_train_errors)
plt.semilogx(lam_vals, ave_test_errors)
plt.legend(['Training error', 'Testing error'])
plt.ylabel('Normalized Error')
plt.xlabel('lambda')
plt.show()

