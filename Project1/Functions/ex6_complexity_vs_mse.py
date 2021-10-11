from functions import *
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import sys
import scipy.stats as st

from sklearn.model_selection import train_test_split, cross_val_score
import sklearn.metrics as metric
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import sklearn.linear_model as skl

terrain_var = imread('SRTM_data_Norway_1.tif') #Read image file

N = 1000 #Number of random points to extract
x = np.random.randint(0, terrain_var.shape[1], size=N) #Random coordinates
y = np.random.randint(0, terrain_var.shape[0], size=N)
z = terrain_var[y,x] #Extract corresponding value of map to the random indexes above

max_degree = 20 #Max polynomial size of runs
testsize = 0.2 #Test size to be used

MSE_testOLS_array = np.zeros(max_degree)
MSE_trainOLS_array = np.zeros(max_degree)
MSE_testridge_array = np.zeros(max_degree)
MSE_testlasso_array = np.zeros(max_degree)

degrees = np.arange(1, max_degree + 1)

for degree in range(1, max_degree + 1):
    lambda_candidates = np.logspace(-10,5,100) 

    X = design_matrix(degree,x,y)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=testsize) #Create / split / scale design matrices
    X_train, X_test = scale(X_train, X_test)

    z_tilde_train, z_tilde_test, beta = OLS(X_train, X_test, z_train, z_test) #Run OLS

    MSE_testOLS_array[degree-1] = MSE(z_test, z_tilde_test)
    MSE_trainOLS_array[degree-1] = MSE(z_train, z_tilde_train) #Fill OLS arrays with MSE values

    z_tilde_test_ridge = Ridge(X_train, X_test, z_train, z_test, lambda_candidates)[1] #Run Ridge

    MSE_testridge_array[degree-1] = MSE(z_test, z_tilde_test_ridge) #Fill ridge array with MSE values

    MSE_temp_lasso = np.zeros(len(lambda_candidates))
    for i, this_lambda in enumerate(lambda_candidates): #Run Lasso
        clf = Lasso(alpha = this_lambda).fit(X_train, z_train)
        z_tilde_test_temp = clf.predict(X_test)
        MSE_temp_lasso[i] = MSE(z_test, z_tilde_test_temp)

    MSE_testlasso_array[degree-1] = np.min(MSE_temp_lasso)
    print(degree) 

#Plot training and test data for OLS
plt.figure() 
plt.plot(degrees, MSE_testOLS_array, label="Test OLS")
plt.plot(degrees, MSE_trainOLS_array, label="Train OLS")
plt.xlabel("Polynomial degree", fontsize="large")
plt.ylabel("Mean squared error ", fontsize="large")
plt.title("N = %i, test size = %.1f%%,\nOLS on terrain data"% (N, testsize*100), fontsize="x-large")
plt.grid()
plt.legend()
plt.semilogy()

#Plot MSE values for test data on all three methods
plt.figure() 
plt.plot(degrees, MSE_testOLS_array, label="OLS")
plt.plot(degrees, MSE_testridge_array, label="Ridge")
plt.plot(degrees, MSE_testlasso_array, label="Lasso")
plt.xlabel("Polynomial degree", fontsize="large")
plt.ylabel("Mean Squared Error",fontsize="large")
plt.title("N = %i, test size = %.1f%%,\nMSE of test data on terrain data"% (N, testsize*100),fontsize="x-large")
plt.grid()
plt.legend()
plt.semilogy()

plt.show()