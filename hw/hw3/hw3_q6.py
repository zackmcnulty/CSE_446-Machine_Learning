'''
hw3_q3c.py

Given a dataset whose response variable is a class y = 1,2,3,..., k (i.e. a classification problem with k classes)
determine the optimal classifier according to the loss functions specified in the homework specification

'''

import numpy as np
import random
import matplotlib.pyplot as plt
from mnist.loader import MNIST


# Load MNIST Data
mndata = MNIST('./data')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())
X_train = X_train / 255.0
X_test = X_test / 255.0

n = X_train.shape[0]
d = X_train.shape[1] # 28 x 28 = 784

#convert training labels to one hot
# i.e. encode an i as [0, 0, ..., 0, 1, 0, ... 0] where only the ith entry is nonzero
Y_train = np.zeros((X_train.shape[0], 10))
for i,digit in enumerate(labels_train):
        Y_train[i, digit] = 1
# =====================================================================================================================================

def objective_function(X, partitions, MU):
    '''
        partitions is a dictionary mapping 1 through k to a set of indices (i.e. refering to datapoints )
        closest to MU_i (where i matches the key of the dictionary)
        MU is a matrix where each row is one of the k means
    '''
    cost = 0
    for i, part in enumerate(partitions):
        mu = MU[i, :]
        cost += np.sum(np.square(np.linalg.norm(X[partitions[part], :] - mu, axis=1)))

    return cost
    

def k_means(X, k):
    # randomly choose the starting points in [0,1] (the range of the data); each mu_i is a row in MU
    MU = np.random.uniform(low=0, high=1, size=(k, d))
    prev_MU = np.zeros((k,d))
    delta = 1e-3
    obj_delta = 1

    objective_values = []
    itr = 0

    while np.amax(np.abs(MU - prev_MU)) > delta and (len(objective_values) < 2 or abs(objective_values[-1] - objective_values[-2]) > obj_delta):
        itr += 1
        print("iteration number: ", itr)
        partitions = {}
        for i in range(n):
            x = X[i, :]
            part = np.argmin(np.linalg.norm(MU - x, axis=1))
            if part in partitions:
                partitions[part].append(i)
            else:
                partitions[part] = [i]

        # Choose the new centroids
        for i in range(k):
            if i in partitions:
                MU[i, :] = 1/len(partitions[i]) * np.sum(X[partitions[i], :], axis=0)

        
            # else just leave the current MU as is?
        objective_values.append(objective_function(X, partitions, MU))
        print(objective_values[-1])

    plt.figure(1)
    plt.plot(objective_values)
    plt.xlabel("iteration number")
    plt.ylabel("objective value")
    plt.show()


    plt.figure(2)
    for i in range(k):
        plt.subplot(5,2,i+1)
        plt.imshow(np.reshape(MU[i, :], (28, 28)))

    plt.show()

    return MU, partitions

# Problem 6a)

#k_means(X_train, 10)

# Problem 6b)
all_train_errors = []
all_test_errors = []
k_vals = [2,5,10,20,40,80,160,320,640,1280] 
for k in k_vals:
    MU_k, partitions_k = k_means(X_train, k)
    train_error = 1/n * objective_function(X_train, partitions_k, MU_k)

    # NOTE: partitions is made for X_train; have to many refind partition for X_test
    test_error = 1/X_test.shape[0] * objective_function(X_test, partitions_k, MU_k)
    all_train_errors.append(train_error)
    all_test_errors.append(test_error)


plt.figure(3)
plt.plot(k_vals, all_train_errors)
plt.plot(k_vals, all_test_errors)
plt.legend(['Train', 'Test'])
plt.show()
