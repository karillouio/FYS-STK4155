from functions import *
import numpy as np
import matplotlib.pyplot as plt

n = 3 #Number of bootstraps
max_degree = 12 #Run for all polynomial degrees up to this
N = 200 #Number of data points
noise = 1 #Factor of noise in data
testsize = 0.2 #Test size

x_y = np.random.rand(N,2) #Create both the function parameters and values
x = x_y[:,0]
y = x_y[:,1]
z = FrankeFunction_noise(x, y, noise)

MSE_a = np.zeros(max_degree)
bias_a = np.zeros(max_degree)
var_a = np.zeros(max_degree)

#Perform bootstrap for all polynomial degrees
for degree in range(1, max_degree + 1): 
    X = design_matrix(max_degree, x, y)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=testsize) #Create / Split / Scale design matrices
    X_train, X_test = scale(X_train, X_test)

    MSE_a[degree-1], bias_a[degree-1], var_a[degree-1] = bootstrap(X_train, X_test, z_train, z_test, n, 'OLS')

#plot
degree_array = np.arange(1, max_degree + 1)
plt.plot(degree_array, bias_a, "--",label = "Bias")
plt.plot(degree_array, var_a, "--",label = "Variance")
plt.plot(degree_array, MSE_a, label = "MSE")
plt.xlabel("Degree of Polynomial", fontsize="large")
plt.ylabel("Error score (MSE/bias/variance)", fontsize="large")
plt.title("Bias-variance tradeoff. N = %i, noise = %.2f, %i bootstraps"%(N, noise, n), fontsize="x-large")
plt.grid()
plt.legend()
plt.semilogy()
plt.show()