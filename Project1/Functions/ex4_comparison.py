from functions import *
import numpy as np
import matplotlib.pyplot as plt

n = 50 #Number of bootstraps
max_degree = 8 #Max polynomial degree
N = 200 
noise = 1 #Noise factor
testsize = 0.25
K = 5

x_y = np.random.rand(N ,2)
x = x_y[:,0]
y = x_y[:,1] 
z = FrankeFunction_noise(x, y, noise)

MSE_boot = np.zeros(max_degree)
MSE_crossval = np.zeros(max_degree)
bias_dummy = np.zeros(max_degree)
var_dummy = np.zeros(max_degree)

for degree in range(1, max_degree + 1):
    X = design_matrix(degree, x, y)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=testsize) 
    X_train, X_test = scale(X_train, X_test)

    MSE_boot[degree-1], bias_dummy[degree-1], var_dummy[degree-1] = bootstrap(X_train, X_test, z_train, z_test, n, 'Ridge')
    MSE_crossval[degree-1] = cross_validation(degree, X, z, K, 'Ridge')
    
polydeg_array = np.arange(1, max_degree + 1) #Plot arrays with MSE, bias and variance together.
plt.plot(polydeg_array, MSE_boot, label="Bootstrap")
plt.plot(polydeg_array, MSE_crossval, label="Cross-validation")
plt.xlabel("Degree of polynomial",fontsize="large")
plt.ylabel("Mean squared error",fontsize="large")
plt.title("Ridge regression - bootstrap vs. cross-validation")
plt.grid()
plt.legend()
plt.semilogy()
plt.show()
