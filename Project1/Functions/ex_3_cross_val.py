from functions import *
import numpy as np

import pandas as pd

degree = 5

N = 200
noise = 1
K = np.array([2,5,8,10]) # of K-folds

x_y = np.random.rand(N, 2)
x = x_y[:,0]
y = x_y[:,1] 
z = FrankeFunction_noise(x, y, noise)

X = design_matrix(degree, x, y)

MSE_crossval = np.zeros(K.shape[0])

for i, K_val in enumerate(K):
    MSE_crossval[i] = cross_validation(degree, X, z, K_val, 'OLS')

crossval_data = {'# of folds (K)': K, 'Mean squared error': MSE_crossval}
crossval_df = pd.DataFrame(data=crossval_data) #Organize data with a data frame

print(crossval_df) #Print the data frame
