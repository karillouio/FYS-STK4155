from functions import *
import numpy as np
import matplotlib.pyplot as plt


K = 5
n = 50 #Bootstraps
N = 200
noise = 1
max_degree = 20
testsize = 0.2

x_y = np.random.rand(N ,2)
x = x_y[:,0]
y = x_y[:,1] 
z = FrankeFunction_noise(x, y, noise)

MSE_crossval = np.zeros(max_degree) 
MSE_bootstrap = np.zeros(max_degree)

for degree in range(1, max_degree + 1):
    X = design_matrix(degree,x,y)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=testsize)
    X_train, X_test = scale(X_train, X_test)

    MSE_crossval[degree-1] = cross_validation(degree, X, z, K, 'Ridge')
    MSE_bootstrap[degree-1] = bootstrap(X_train, X_test, z_train, z_test, n, 'Ridge')[0]

degrees = np.arange(1, max_degree + 1) #Plot arrays together in the same figure.
plt.plot(degrees, MSE_crossval, label="Cross-validation")
plt.plot(degrees, MSE_bootstrap, label="Bootstrap")
plt.xlabel("Degree of polynomial)", fontsize="large")
plt.ylabel("Error score (MSE)",fontsize="large")
plt.title("Cross-Validation vs. Bootstrap.\n N = %i, noise = %.2f, bootstraps=%i"%(N,noise,n),fontsize="x-large")
plt.grid(); plt.legend(); plt.semilogy()
plt.show()