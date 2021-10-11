from functions import *
import numpy as np
import matplotlib.pyplot as plt

max_degree = 20
N = 200
noise = 0.2 
testsize = 0.2

x_y = np.random.rand(N,2)
x = x_y[:,0]
y = x_y[:,1] 
z = FrankeFunction_noise(x, y, noise)

lambda_candidates = np.logspace(-10,5,100)

MSE_train_array = np.zeros(max_degree)
MSE_test_array = np.zeros(max_degree)

for degree in range(1, max_degree + 1): 
    X = design_matrix(degree, x, y)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=testsize)
    X_train, X_test = scale(X_train, X_test)

    z_tilde_train, z_tilde_test, Beta_Ridge, optimalLambda, MSE_lamb = Ridge(X_train, X_test, z_train, z_test, lambda_candidates)

    MSE_train_array[degree-1] = MSE(z_train,z_tilde_train)
    MSE_test_array[degree-1] = MSE(z_test,z_tilde_test)

polydegs = np.arange(1, degree + 1)
plt.figure()
plt.plot(polydegs,MSE_train_array, label="Training")
plt.plot(polydegs,MSE_test_array, label="Test")
plt.xlabel("Polynomial degree", fontsize="large")
plt.ylabel("Mean Squared Error (MSE)", fontsize="large")
plt.title("N = %i, test size = %.1f%%, noise = %.2f\nRidge Shrinkage Method"% (N,testsize*100,noise), fontsize="x-large")
plt.grid()
plt.legend()
plt.semilogy()
plt.show()
