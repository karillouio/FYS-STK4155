from functions import *
import numpy as np
import matplotlib.pyplot as plt

degree = 5 #seems to be a good choice
N = 200
noise = 0.2 
testsize = 0.2

x_y = np.random.rand(N,2)
x = x_y[:,0]
y = x_y[:,1] 
z = FrankeFunction_noise(x, y, noise)

lambda_candidates = np.logspace(-10, 3, 100)


X = design_matrix(degree, x, y)
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=testsize)
X_train, X_test = scale(X_train, X_test)

z_tilde_train, z_tilde_test, Beta_Ridge, optimalLambda, MSE_lambdas = Ridge(X_train, X_test, z_train, z_test, lambda_candidates)



plt.plot(lambda_candidates, MSE_lambdas)
plt.xlabel("Value of lambda", fontsize="large")
plt.ylabel("Mean squared error", fontsize="large")
plt.title("Ridge regression with varying lambda values", fontsize="x-large")
plt.legend()
plt.grid()
plt.semilogx()
plt.show()
