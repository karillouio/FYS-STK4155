from functions import *
import numpy as np
import matplotlib.pyplot as plt


K = 5 ##number of folds
n = 50 #number of bootstraps
N = 200 #number of data points
noise = 1
max_degree = 12 
testsize = 0.25

x_y = np.random.rand(N,2)
x = x_y[:,0]; y = x_y[:,1] #Create both function parameters and function values
z = FrankeFunction_noise(x, y, noise)

MSE_crossval_array = np.zeros(max_degree) #Create empty arrays to be filled with interesting data
MSE_bootstrap_array = np.zeros(max_degree)

for degree in range(1, max_degree + 1): #Run through all degrees of polynomials
    X = design_matrix(degree, x, y)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=testsize)
    X_train, X_test = scale(X_train, X_test)

    MSE_crossval_array[degree-1] = cross_validation(degree, X, z, K, 'OLS')
    MSE_bootstrap_array[degree-1] = bootstrap(X_train, X_test, z_train, z_test, n, "OLS")[0]

degrees = np.arange(1, max_degree + 1) 

#Display the results together
plt.plot(degrees, MSE_crossval_array, label = "Cross-validation")
plt.plot(degrees, MSE_bootstrap_array, label = "Bootstrap")
plt.xlabel("Degree of polynomial", fontsize="large")
plt.ylabel("Error score (MSE)", fontsize="large")
plt.title("Cross-validation and bootstrap.\n N = %i, noise = %.2f, %i bootstraps"%(N,noise,n), fontsize="x-large")
plt.grid()
plt.legend()
plt.semilogy()
plt.show()