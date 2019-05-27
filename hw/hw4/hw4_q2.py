'''

hw4_q2.py

Building a movie recommendation system.

'''

import csv
import numpy as np
import matplotlib.pyplot as plt

# LOAD DATA ===========================================================================
data = []
with open('./data/ml-100k/u.data') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        data.append([int(row[0])-1, int(row[1])-1, int(row[2])])

data = np.array(data)

n = len(data) # n= 100,000
num_users = max(data[:,0])+1 # num_users = 943, indexed 0,...,942
num_items = max(data[:,1])+1 # num_items = 1682 indexed 0,...,1681

np.random.seed(1)
num_train = int(0.8*n)
perm = np.random.permutation(data.shape[0])
train = data[perm[0:num_train],:]
test = data[perm[num_train::],:]

# COMMON FUNCTIONS =====================================================================

def find_error(X, R_hat):
    '''

    :param X: data matrix in sparse representation: rows are in form (user_index, movie_index, rating)
    :param R_hat: predicted movie preferences; this is a function that takes in (i,j), the user
                  index and the movie index, and outputs a number in [1, 5]
    :return: testing error
    '''

    n = X.shape[0]

    # rows of X are (user_index, movie_index, rating)
    # R_hat is a function that takes in (user_index, movie_index) and outputs predicted rating
    return 1/n * sum([(X[k, 2] - R_hat(X[k, 0], X[k, 1])) ** 2 for k in range(n)])





# Problem 2a =============================================================================

mu = np.zeros(shape=(num_items, ))

# counts the number of ratings for each movie
counts = np.zeros(shape=(num_items, ), dtype=int)
for i in range(train.shape[0]):
    next_rating = train[i, 2]
    next_movie = train[i, 1]
    mu[next_movie] += next_rating
    counts[next_movie] += 1


# TODO: Temporary fix( check piazza @538 for answer)
counts = np.where(counts == 0, 1, counts)

mu = mu / counts

R_hat = lambda user_index, movie_index: mu[movie_index]
print('Test Error Part a: ', find_error(test, R_hat))










# Problem 2b =============================================================================

matrix = np.zeros(shape=(num_items, num_users))
for entry in range(train.shape[0]):
    next_user = train[entry, 0]
    next_movie = train[entry, 1]
    next_rating = train[entry, 2]
    matrix[next_movie, next_user] = next_rating

[u, s, vh] = np.linalg.svd(matrix, full_matrices=False)

d_vals = [1,2,5,10,20,50]
all_train_errors = []
all_test_errors = []

for d in d_vals:

    # Calculate full matrix beforehand
    #R_lowrank = np.dot(u[:, :d] * s[:d], vh[:d, : ])
    #Rd_hat = lambda user_index, movie_index: R_lowrank[movie_index, user_index]

    # Consider R_ij as an inner product
    #Rd_hat2 = lambda user_index, movie_index: sum([u[movie_index, i]*s[i]*vh[i, user_index] for i in range(d)])

    # First, note that the lowrank (rank d) approximation can be acheived through the matrix
    # multiplication U[:, :d] * S[:d, :d] * Vh[:d, :] = U_d S_d Vh_d.
    # To find the (i,j) = (movie_index, user_index) entry in this low rank approximation of our matrix, we can multiply the
    # ith row of U_dS_d by the jth column of Vh_d
    Rd_hat = lambda user_index, movie_index: np.inner(u[movie_index, :d]*s[:d], vh[:d, user_index])

    train_error = find_error(train, Rd_hat)
    test_error = find_error(test, Rd_hat)
    all_test_errors.append(test_error)
    all_train_errors.append(train_error)

    #print('Test Error Part b (d = ', d, '): ', test_error)

plt.figure(1)
plt.plot(d_vals, all_train_errors)
plt.plot(d_vals, all_test_errors)
plt.title('Problem 2b: Error using Low Rank Approximations ')
plt.xlabel('Rank used for Approximation (d)')
plt.ylabel('Error')
plt.legend(['Training Error', 'Testing Error'])
plt.show()












# Problem 2c =============================================================================


matrix = np.zeros(shape=(num_items, num_users)) - 1 # negative entries will be our sign that no ratings are present
for entry in range(train.shape[0]):
    next_user = train[entry, 0]
    next_movie = train[entry, 1]
    next_rating = train[entry, 2]
    matrix[next_movie, next_user] = next_rating

# Replace missing entries with the average rating from other users.
# mu is calculated in part a) above (MOVE THIS DOWN HERE?)
for j in range(matrix.shape[1]):
    matrix[:, j] = np.where(matrix[:, j] == -1, mu, matrix[:, j])


[u, s, vh] = np.linalg.svd(matrix, full_matrices=False)

d_vals = [1,2,5,10,20,50]
all_train_errors = []
all_test_errors = []

for d in d_vals:

    # Calculate full matrix beforehand
    #R_lowrank = np.dot(u[:, :d] * s[:d], vh[:d, : ])
    #Rd_hat = lambda user_index, movie_index: R_lowrank[movie_index, user_index]

    # Consider R_ij as an inner product
    #Rd_hat2 = lambda user_index, movie_index: sum([u[movie_index, i]*s[i]*vh[i, user_index] for i in range(d)])

    # First, note that the lowrank (rank d) approximation can be acheived through the matrix
    # multiplication U[:, :d] * S[:d, :d] * Vh[:d, :] = U_d S_d Vh_d.
    # To find the (i,j) = (movie_index, user_index) entry in this low rank approximation of our matrix, we can multiply the
    # ith row of U_dS_d by the jth column of Vh_d
    Rd_hat = lambda user_index, movie_index: np.inner(u[movie_index, :d]*s[:d], vh[:d, user_index])

    train_error = find_error(train, Rd_hat)
    test_error = find_error(test, Rd_hat)
    all_test_errors.append(test_error)
    all_train_errors.append(train_error)

    #print('Test Error Part b (d = ', d, '): ', test_error)

plt.figure(2)
plt.plot(d_vals, all_train_errors)
plt.plot(d_vals, all_test_errors)
plt.title('Problem 2c: Error using Low Rank Approximations ')
plt.xlabel('Rank used for Approximation (d)')
plt.ylabel('Error')
plt.legend(['Training Error', 'Testing Error'])
plt.show()















# Problem 2d =============================================================================










# Problem 2e =============================================================================
