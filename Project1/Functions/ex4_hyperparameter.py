from functions import *
import numpy as np
import matplotlib.pyplot as plt

degree = 5
N = 200 
noise = 1

x_y = np.random.rand(N,2)
x = x_y[:,0]
y = x_y[:,1] 
z = FrankeFunction_noise(x, y, noise)
X = design_matrix(degree, x, y) 

lambda_candidates = np.logspace(-10, 5, 200)

X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2) 
X_train_scaled, X_test_scaled = scale(X_train, X_test)

z_tilde_train, z_tilde_test, BetaRidge, OptLamb, MSE_lamb = Ridge(X_train, X_test, z_train, z_test, lambda_candidates)

#Plot MSE vs hyperparameter
plt.plot(lambda_candidates, MSE_lamb, label="Test data MSE", color = 'g')
plt.axvline(OptLamb,label="$\lambda = $%e"%OptLamb,color = 'r')
plt.semilogx();
plt.grid()
plt.xlabel("Value for hyperparameter $\lambda$", fontsize="large")
plt.ylabel("Mean Squared Error (MSE)", fontsize="large")
plt.title("Ridge Hyperparameter fit for $\lambda$", fontsize="x-large")
plt.legend()
plt.show()