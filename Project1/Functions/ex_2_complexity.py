from functions import *
import numpy as np
import matplotlib.pyplot as plt

max_degree = 12 #Test all polynomial degrees up to this one
N = 500 #Number of data points
noise = 1 #Factor of noise in data
testsize = 0.2 #Test size

x_y = np.random.rand(N, 2) #Generate function parameteres and values
x = x_y[:,0]
y = x_y[:,1]
z = FrankeFunction_noise(x, y, noise)

MSE_train_array = np.zeros(max_degree)
MSE_test_array = np.zeros(max_degree)

#Perform OLS for all polynomial degrees
for degree in range(1, max_degree + 1): 
    
    X = design_matrix(degree, x, y)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=testsize)
    X_train, X_test = scale(X_train, X_test)

    z_tilde_train, z_tilde_test, beta_optimal = OLS(X_train, X_test, z_train, z_test) #Calculates OLS values

    MSE_train_array[degree-1] = MSE(z_train,z_tilde_train) #Fills the arrays to be plotted
    MSE_test_array[degree-1] = MSE(z_test,z_tilde_test)

polydeg_array = np.arange(1, max_degree + 1) #Plot the MSE results against each other
plt.plot(polydeg_array,MSE_train_array,label="Training")
plt.plot(polydeg_array,MSE_test_array,label="Testing")
plt.xlabel("Polynomial degree of model", fontsize="large")
plt.ylabel("Mean Squared Error", fontsize="large")
plt.title("Bias-Variance tradeoff", fontsize="x-large")
plt.legend()
plt.grid()
plt.semilogy()
plt.show()