import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt


# Problem 6b)

def train(X, Y, lam):
    '''
    Given training data X and its associated labels Y, performs ridge regression
    using the given regularization parameter lambda = lam. Returns the weights matrix
    W_hat generated by this procedure.
    '''
    d = X.shape[1]

    Xt_X_plus_lambda = np.dot(X.T, X) + lam * np.eye(d,d)
    Xt_Y = np.dot(X.T, Y)

    return np.linalg.solve(Xt_X_plus_lambda, Xt_Y)

def predict(W, X_prime):
    '''
    Given a trained mapping W from pixelspace to digitspace and new observations X, makes predictions
    about the digit present in each image in X. Returns these labels (as digits)
    '''
    return np.argmax(np.dot(X_prime, W), axis = 1)

def check_error(predictions, actual):
   '''
   Given two label vectors (as digits not one-hot), true labels and predicted labels, calculates the error rate.
   '''
   if len(predictions) != len(actual):
       raise ValueError("Two label vectors must be the same length")

   return sum([1 for i in range(len(predictions)) if predictions[i] != actual[i] ]) / len(predictions)


# ======= LOAD MNIST DATA ==================================
mndata = MNIST('./python-mnist/data/')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())
X_train = X_train/255.0
X_test = X_test/255.0

# Convert training labels to one hot 
# i.e. encode an i as [0, 0, ..., 0, 1, 0, ... 0] where only the ith entry is nonzero
Y_train = np.zeros((X_train.shape[0], 10))
for i,digit in enumerate(labels_train):
    Y_train[i, digit] = 1


num_training_images = X_train.shape[0]
num_test_images = X_test.shape[0]
d = X_train.shape[1]
# ==========================================================

'''
W = train(X_train, Y_train, 10**(-4))
train_predictions = predict(W, X_train)
test_predictions = predict(W, X_test)

test_error = check_error(test_predictions, labels_test)
train_error = check_error(train_predictions, labels_train)

print("Training error: ", train_error, "\nTest error: ", test_error)
'''


# ========================================================================
# ========================================================================
# ========================================================================

# Problem 6c)
variance = 0.1
lam = 0.01#10**(-4) # lambda for ridge regression
fraction_train = 0.8

#p_vals = [500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 20000]
p_vals = [1000*i for i in range(1, 6)]
all_train_errors = []
all_validation_errors = []

# choose which images/indices will be train/validation data 
shuffled_indices = np.arange(num_training_images)
np.random.shuffle(shuffled_indices)
train_indices = shuffled_indices[0:int(fraction_train * num_training_images)]
validation_indices = shuffled_indices[int(fraction_train * num_training_images) : ]

# generate the corresponding labels based on the shuffled indices
Yp_train = Y_train[train_indices, :]
labels_train_p = labels_train[train_indices]
labels_validate_p = labels_train[validation_indices]

# Used for determining the best p_value to choose
min_validation_error = 10**8

for p in p_vals:

    print("Running using p = ", p)
    # calculate G and b
    G = np.random.normal(0, np.sqrt(variance), size = (p,d))
    b = np.random.uniform(low=0, high=2*np.pi, size=(p,1)) 

    # transform each row x_i of X by cos(Gx_i + b) --> new_x_i = cos(x_i^TG^T + b^T) --> X_new = cos(XG^T + b^T)
    X_train_transformed = np.cos(np.dot(X_train, G.T) + b.T)
    

    # selected out the appropriate images for training/validation from pre-selected
    # indices
    Xp_train = X_train_transformed[train_indices, :]
    Xp_validate = X_train_transformed[validation_indices, :]

    Wp = train(Xp_train, Yp_train, lam)
    train_predictions = predict(Wp, Xp_train)
    validation_predictions = predict(Wp, Xp_validate)

    train_error = check_error(train_predictions, labels_train_p)
    all_train_errors.append(train_error)
    validation_error = check_error(validation_predictions, labels_validate_p)
    all_validation_errors.append(validation_error)

    print("training error: ", train_error)
    print("validation error: ", validation_error)

    # for saving the best p-value and corresponding Wp, G, b for part 6d)
    if min_validation_error > validation_error:
        min_validation_error = validation_error
        best_p = p
        best_b = b
        best_G = G
        Wp_best = Wp


# Plot the results!
plt.plot(p_vals, all_train_errors)
plt.plot(p_vals, all_validation_errors)
plt.legend(["training errors", "validation errors"])
plt.xlabel('p')
plt.ylabel('error rate')
plt.show()


# =================================================

# Problem 6d)

# \hat{epsilon}(\hat{f})

X_test_transformed = np.cos(np.dot(X_test, G.T) + b.T)
test_error = check_error(predict(Wp_best,X_test_transformed), labels_test)
N_test = num_test_images
square_root_part = np.sqrt(np.log(40) /  (2*N_test))

print("N_test: ", N_test)
print("best_p: ", best_p)
print("test error: ", test_error)
print("Square root part: ", square_root_part)


print("True classification error 95% confidence interval: (", \
        test_error - square_root_part, ", ", test_error + square_root_part, ")")
