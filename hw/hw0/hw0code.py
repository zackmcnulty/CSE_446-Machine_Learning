import numpy as np
import matplotlib.pyplot as plt


# Defining variables
A = np.array([[0,2,4], [2,4,2], [3,3,1]])
b = np.array([-2, -2, -4])
c = np.array([1, 1, 1])

#print("\nA:\n", A)
#print("\nb: \n", b)
#print("\nc: \n", c)


# Problem 11

# part a)
Ainv = np.linalg.inv(A)

print("\n A^{-1} : \n", Ainv)
#print("\nA * A^{-1}: \n", np.dot(A, Ainv))

# part b)

Ainvb = np.dot(Ainv,b)
Ac = np.dot(A,c)

print("\n A^{-1} b: \n", Ainvb)
print("\n Ac : \n", Ac)




# Problem 12

# part a)
n = 40000
Z = np.random.randn(n)
plt.step(sorted(Z), np.arange(1, n+1)/float(n))

# part b)
for k in [1, 8, 64, 512]:
    Yk = np.sum(np.sign(np.random.randn(n,k)) * np.sqrt(1./ k), axis = 1)
    plt.step(sorted(Yk), np.arange(1, n+1)/float(n))


# plotting results
plt.legend(['Gaussian', '1', '8', '64', '512'])
plt.ylim((0,1))
plt.xlim((-3,3))
plt.ylabel('Probability')
plt.xlabel('Observations')
plt.ioff()
plt.show()

