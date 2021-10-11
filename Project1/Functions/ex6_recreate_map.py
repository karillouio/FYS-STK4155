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

terrain_var = imread('SRTM_data_Norway_1.tif') #Read initial map

N = 10000 #Number of data points
x = np.random.randint(0, terrain_var.shape[1], size=N)
y = np.random.randint(0, terrain_var.shape[0], size=N)
TrainingData = terrain_var[y,x] #Terrain value in a random position

#Options: 'OLS', 'Ridge', 'Lasso'
method = 'Ridge'

if method == 'OLS':

    degree = 20

    X = design_matrix(degree, x, y)
    X_train, X_test, z_train, z_test = train_test_split(X, TrainingData, test_size=0.2)
    X_train, X_test = scale(X_train, X_test)

    z_tilde_train, z_tilde_test, beta = OLS(X_train, X_test, z_train, z_test) #Run OLS

    MSE_train = MSE(z_train, z_tilde_train)
    MSE_test = MSE(z_test, z_tilde_test)

    image_approx = np.zeros(terrain_var.shape) #Empty array for terrain data

    #Try to recreate the rows of the map
    for y_index in range(terrain_var.shape[0]): 
        print(y_index) #keeps track of how close it is to finishing
        X_temp = design_matrix(degree, np.arange(terrain_var.shape[1]), y_index*np.ones(terrain_var.shape[1])) 
        X_temp = scale(X_train, X_temp)[1]
        image_approx[y_index] = X_temp @ beta 
        del X_temp 

    #Approximated map
    plt.figure() 
    plt.title("Approximate map\nPolynomial degree: %i , MSE value: %e"%(degree, MSE_test), fontsize="x-large")
    plt.imshow(image_approx, cmap='gray')
    plt.xlabel("<- West - East ->", fontsize="large")
    plt.ylabel("<- South - North ->", fontsize="large")
    plt.xticks([])
    plt.yticks([])


    #Original map
    plt.figure() 
    plt.title("Actual map", fontsize="x-large")
    plt.imshow(terrain_var, cmap='gray')
    plt.xlabel("<- West - East ->", fontsize="large")
    plt.ylabel("<- South - North ->", fontsize="large")
    plt.xticks([]);plt.yticks([])
    plt.show()

elif method == 'Ridge':
    degree = 20 

    X = design_matrix(degree, x, y)
    X_train, X_test, z_train, z_test = train_test_split(X, TrainingData, test_size=0.2)
    X_train, X_test = scale(X_train, X_test)

    lambda_candidates = np.logspace(-10, 3, 100) #Possible lambda values
    z_tilde_train, z_tilde_test, beta = Ridge(X_train, X_test, z_train, z_test, lambda_candidates)[:3]

    MSE_train = MSE(z_train, z_tilde_train)
    MSE_test = MSE(z_test, z_tilde_test)

    image_approx = np.zeros(terrain_var.shape) #Empty array for terrain data

    #Try to recreate the rows in the map
    for y_index in range(terrain_var.shape[0]): 
        print(y_index) #keeps track of how close it is to finishing
        X_temp = design_matrix(degree, np.arange(terrain_var.shape[1]), y_index*np.ones(terrain_var.shape[1])) 
        X_temp = scale(X_train, X_temp)[1] 
        image_approx[y_index] = X_temp @ beta 
        del X_temp 

#Approximated map
    plt.figure() 
    plt.title("Approximate map using Ridge \nPolynomial Degree: %i , MSE value: %e" % (degree, MSE_test), fontsize="x-large")
    plt.imshow(image_approx, cmap='gray')
    plt.xlabel("<- West - East ->",fontsize="large")
    plt.ylabel("<- South - North ->",fontsize="large")
    plt.xticks([])
    plt.yticks([])

#Original map
    plt.figure() 
    plt.title("Actual map",fontsize="x-large")
    plt.imshow(terrain_var, cmap='gray')
    plt.xlabel("<- West - East ->",fontsize="large")
    plt.ylabel("<- South - North -->",fontsize="large")
    plt.xticks([])
    plt.yticks([])
    plt.show()

elif method == 'Lasso':
     degree = 20

     X = design_matrix(degree,x,y)
     X_train, X_test, z_train, z_test = train_test_split(X,TrainingData,test_size=0.2)
     X_train, X_test = scale(X_train, X_test)

     lambda_candidates = np.logspace(-10,3,100)

     MSE_temp_array = np.zeros(lambda_candidates.shape[0])
     beta_temp_array = np.zeros([lambda_candidates.shape[0],X.shape[1]])

     for i, this_lambda in enumerate(lambda_candidates):
         clf = Lasso(alpha = this_lambda).fit(X_train,z_train) 
         beta_temp_array[i] = clf.coef_  
         MSE_temp_array[i] = MSE(z_test,clf.predict(X_test))

     MSE_test = np.min(MSE_temp_array) #Find the lowest MSE
     beta = beta_temp_array[np.argmin(MSE_test)] #Optimal beta

     image_approx = np.zeros(terrain_var.shape) #Empty array for terrain data

    #Row by row
     for y_index in range(terrain_var.shape[0]): 
         print(y_index) #keeps track of how close it is to finishing
         X_temp = design_matrix(degree, np.arange(terrain_var.shape[1]), y_index*np.ones(terrain_var.shape[1])) 
         X_temp = scale(X_train, X_temp)[1] #Scale the design matrix
         image_approx[y_index] = X_temp @ beta #Recreate map for each row
         del X_temp 

     #Approximated map
     plt.figure()
     plt.title("Approximate map using Lasso \nPolynomial Degree: %i , MSE value: %e" % (degree, MSE_test),fontsize="x-large")
     plt.imshow(image_approx, cmap='gray')
     plt.xlabel("<- West - East ->",fontsize="large")
     plt.ylabel("<- South - North ->",fontsize="large")
     plt.xticks([])
     plt.yticks([])

     #Original map
     plt.figure() #Plot the actual map
     plt.title("Actual map", fontsize="x-large")
     plt.imshow(terrain_var, cmap='gray')
     plt.xlabel("<- West - East ->",fontsize="large")
     plt.ylabel("<- South - North ->",fontsize="large")
     plt.xticks([])
     plt.yticks([])
     plt.show()