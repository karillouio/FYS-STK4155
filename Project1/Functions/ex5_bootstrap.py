from functions import *
import numpy as np
import matplotlib.pyplot as plt

n = 20 #Number of bootstraps
max_degree = 7 #Max polynomial degree
N = 200 
noise = 1 #Noise factor
testsize = 0.2 

x_y = np.random.rand(N ,2)
x = x_y[:,0]
y = x_y[:,1] 
z = FrankeFunction_noise(x, y, noise)

MSE_a = np.zeros(max_degree)
bias_a = np.zeros(max_degree)
var_a = np.zeros(max_degree)

for degree in range(1, max_degree + 1):
    X = design_matrix(degree, x, y)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=testsize) 
    X_train, X_test = scale(X_train, X_test)
    
    MSE_a[degree-1], bias_a[degree-1], var_a[degree-1] = bootstrap(X_train, X_test, z_train, z_test, n, 'Lasso')
    
polydeg_array = np.arange(1, max_degree + 1) #Plot arrays with MSE, bias and variance together.
plt.plot(polydeg_array, bias_a, "--", label="Bias")
plt.plot(polydeg_array, var_a, "--", label="Variance")
plt.plot(polydeg_array, MSE_a, label="MSE")
plt.xlabel("Degree of polynomial)",fontsize="large")
plt.ylabel("Error (MSE/bias/variance)",fontsize="large")
plt.title("Bias-variance tradeoff. N = %i, noise = %.2f, %i bootstraps"%(N, noise, n), fontsize="large")
plt.grid()
plt.legend()
plt.semilogy()
plt.show()
