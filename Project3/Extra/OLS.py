from functions import *
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(272)

n = 25 #Number of bootstraps
N = 2000
noise = 0.2
#max_degree = 25
max_degree = 15

testsize = 0.2

x_y = np.random.rand(N,2)
x = x_y[:,0]; y = x_y[:,1] #Create both function parameters and function values
z = FrankeFunction_noise(x, y, noise)

MSE_ = np.zeros(max_degree)#Create empty array to be filled with interesting data
bias_ = np.zeros(max_degree)#Create empty array to be filled with interesting data
var_ = np.zeros(max_degree)#Create empty array to be filled with interesting data

for degree in range(1, max_degree + 1): #Run through all degrees of polynomials
    X = design_matrix(degree, x, y)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=testsize)
    X_train, X_test = scale(X_train, X_test)
    #MSE_[degree-1], bias_[degree-1], var_[degree-1] = bootstrap(X_train, X_test, z_train, z_test, n, 'OLS')
    MSE_[degree-1], bias_[degree-1], var_[degree-1] = bootstrap(X_train, X_test, z_train, z_test, n, 'Ridge')

degrees = np.arange(1, max_degree + 1) 

#Display the results
plt.plot(degrees, MSE_, label = 'MSE')
plt.plot(degrees, bias_, label = 'Bias')
plt.plot(degrees, var_, label = 'Variance')
plt.xlabel("Degree of polynomial", fontsize="large")
plt.ylabel("Error", fontsize="large")
#plt.title("Bias-variance trade-off, ordinary least squares regression", fontsize="x-large")
plt.title("Bias-variance trade-off, Ridge regression", fontsize="x-large")
plt.grid()
plt.legend()
plt.semilogy()
plt.show()