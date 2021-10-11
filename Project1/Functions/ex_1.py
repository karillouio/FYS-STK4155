from functions import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

N = 200 #number of data points
degree = 5 #polynomial degree
noise = 0.2 #factor of noise in data

#Create random function parameters
x_y = np.random.rand(N, 2) 
x = x_y[:,0]
y = x_y[:,1]


X = design_matrix(degree, x, y) #Create design matrix
z = FrankeFunction_noise(x, y, noise) #Franke function with noise

#Split the data into a training set and a test set using an SKLearn function
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.25)
X_train, X_test = scale(X_train, X_test) #Scale the data

z_tilde_train, z_tilde_test, beta = OLS(X_train, X_test, z_train, z_test)

MSE_train = MSE(z_train, z_tilde_train)
R2_train = R2(z_train, z_tilde_train)

MSE_test = MSE(z_test, z_tilde_test)
R2_test = R2(z_test, z_tilde_test)

#Confidence intervals
var_Z = variance_estimator(degree, z_train, z_tilde_train)
var_beta = np.diag(np.linalg.pinv(X_train.T @ X_train))*var_Z
CI_beta_L, CI_beta_U = confidence_interval(beta, var_beta, 0.05)
CI_beta_df = pd.DataFrame(np.transpose(np.array([CI_beta_L, beta, CI_beta_U])),columns=['Lower CI', 'beta', 'Upper CI'])

#Output
print("Training data, R2 score: %e" % R2_train)
print(" ")
print("Test data, R2 score: %e" % R2_test)
print(" ")
print("Training data, MSE score: %e" % MSE_train)
print(" ")
print("Test data, MSE score %e" % MSE_test)
print("\nConfidence interval score\n")

print(CI_beta_df)





