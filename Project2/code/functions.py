import numpy as np
import sys
import scipy.stats as st

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, SGDRegressor
from sklearn.model_selection import train_test_split
import sklearn.metrics as metric

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Calculate the confidence interval of a data set with a normal distri
# mean, variance: self-explanatory
# alpha: significance level (confidence level = 1 - alpha)
# returns the lower and upper confidence boundaries
def confidence_interval(mean, variance, alpha):

    sigma = np.sqrt(variance) #standard deviation
    Z = st.t.ppf(1-alpha/2, variance.shape[0]-1)
    
    lower = mean - Z*sigma
    upper = mean + Z*sigma
    return lower, upper


#Calculate the estimator of an unknown variance
#p: degree of the polynomial
#y: actual data points
#y_tilde: predicted data points
def variance_estimator(p, y, y_tilde):
    n = len(y)

    var = np.sum((y-y_tilde)**2)/(n-p-1)
    return var


#Calculate the coefficient of determination
#y: actual data points
#y_tilde: modelled data points
def R2(y, y_tilde):

    n = len(y)
    y_mean = np.mean(y)

    SS_res = 0 #sum of squares of residuals
    SS_tot = 0 #total sum of squares
    for i in range(n):
        SS_res += (y[i] - y_tilde[i])**2
        SS_tot += (y[i] - y_mean)**2
    R2_val = 1 - (SS_res/SS_tot)
    return R2_val

#Calculate the mean squared error
#y: actual data points
#y_tilde: modelled data points
def MSE(y, y_tilde):
    sum = 0
    n = len(y)
    for i in range(n):
        sum += (y[i] - y_tilde[i])**2
    mean_squared_error = sum/n
    return mean_squared_error

#Calculate the mean error
#y: actual data points
#y_tilde: predicted data points
def mean_error(y, y_tilde):
    
    n = len(y)
    sum = 0
    for i in range(n):
        sum += np.abs((y[i]-y_tilde[i]))
    output = sum/n
    return output


def FrankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def FrankeFunction_noise(x, y, noise):

    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    n = len(x)
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 - term4 + noise*np.random.randn(n)


# Create a 1D design matrix
# degree: the degree of the polynomial
# x : x-values
def design_matrix_1d(degree, x):

    if len(x.shape) > 1:
        x = np.ravel(x)

    n = len(x)
    X = np.zeros([n, degree])
    X[:,0] = 1
    
    for i in range(1, degree):
        X[:,i] = (x**i)
    return X


# Create a 2D design matrix
# degree: the degree of the polynomial
# x and y: x-values and y-values
def design_matrix_2d(degree, x, y):

    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    n = len(x)
    terms = int(((degree+1)*(degree+2))/2) #number of terms

    X = np.zeros([n, terms])
    X[:,0] = 1
    column = 1
    for i in range(1, degree+1):
        for j in range(i+1):
            X[:,column] = (x**j)*(y**(i-j))
            column += 1
    return X

#Ordinary Least Squares on a data set
#X_train: training design matrix
#X_test: testing design matrix
#y_train: data point sets corresponding to X_train
#y_test: data point sets corresponding to X_test

#y_tilde_train: approximated values corresponding to training data
#y_tilde_test: approximated values corresponding to testing data
#beta: coefficients of the polynomial that is the best fit
def OLS(X_train, X_test, y_train, y_test):

    beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
    y_tilde_train = X_train @ beta
    y_tilde_test = X_test @ beta

    return y_tilde_train, y_tilde_test, beta


#Ridge regression on a data set, find the optimal lambda
#X_train: training design matrix
#X_test: testing design matrix
#y_train: data point sets corresponding to X_train
#y_test: data point sets corresponding to X_test
#lambda_candidates: potential lambda values to test
#testsize: proportion of data set aside for testing

#y_tilde_train: approximated values corresponding to training data
#y_tilde_test: approximated values corresponding to testing data
#beta: coefficients of the polynomial that is the best fit
#best_lambda: the optimal lambda, i.e. the one that yields the lowest MSE
#lambda_MSEs: array containing the MSEs associated with lambdas

def Ridge(X_train, X_test, y_train, y_test, lambda_candidates, testsize = 0.25):

    beta = np.zeros((len(lambda_candidates),X_train.shape[1]))
    lambda_MSEs = np.zeros(len(lambda_candidates))

    X_training, X_validate, y_training, y_validate = train_test_split(X_train, y_train, test_size = testsize)

    for i, lambda_val in enumerate(lambda_candidates):
        beta[i,:] = np.linalg.pinv(X_training.T @ X_training + lambda_val * np.identity((X_training.T @ X_training).shape[0])) @ X_training.T @ y_training
        y_tilde_validate = X_validate @ beta[i]

        lambda_MSEs[i] = MSE(y_validate, y_tilde_validate)

    best_lambda = lambda_candidates[np.argmin(lambda_MSEs)]
    beta = beta[np.argmin(lambda_MSEs)]

    y_tilde_train = X_train @ beta
    y_tilde_test = X_test @ beta

    return y_tilde_train, y_tilde_test, beta, best_lambda, lambda_MSEs

#Scale design matrices using with the SKLearn StandardScaler using the first input
def scale(x_train, x_test):

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)    
    x_train_scaled[:,0] = 1
    
    x_test_scaled = scaler.transform(x_test)
    x_test_scaled[:,0] = 1

    return x_train_scaled, x_test_scaled

#Bootstrap resampling

#X_train: training design matrix
#X_test: testing design matrix
#y_train: data point sets corresponding to X_train
#y_test: data point sets corresponding to X_test
#n: bootstrap iterations
#method: linear regression method (can be OLS, Ridge or Lasso)

#boot_MSE: mean value of the calculated MSE values of all bootstraps
#boot_bias: mean value of the calculated bias values of all bootstraps
#boot_variance: mean value of the calculated variance values of all bootstraps

def bootstrap(X_train, X_test, y_train, y_test, n, method):

    y_tilde_test = np.empty((y_test.shape[0], n))

    for i in range(n):
        random_indexes = np.random.randint(0, len(X_train), len(X_train))
        X = X_train[random_indexes]
        Y = y_train[random_indexes]
        if method == 'OLS':
            y_tilde_test[:,i] = OLS(X, X_test, Y, y_test)[1]

        elif method == 'Ridge':
            lambda_values = np.logspace(-3, 5, 200)
            y_tilde_test[:,i] = Ridge(X, X_test, Y, y_test, lambda_values)[1]

        elif method == 'Lasso':
            lambda_values = np.logspace(-10, 5, 100)
            MSE_test_array = np.zeros(len(lambda_values))
            Y_tilde_test_array = np.zeros([len(lambda_values), X_test.shape[0]])

            for j, lambda_val in enumerate(lambda_values):
                clf = Lasso(alpha=lambda_val).fit(X, Y)
                Y_tilde_test_array[j] = clf.predict(X_test)
                MSE_test_array[j] = MSE(y_test,Y_tilde_test_array[i])

            y_tilde_test[:,i] = Y_tilde_test_array[np.argmin(MSE_test_array)]

        else:
            print("Not a valid regression method")
            sys.exit(0)

    y_test = y_test[:,np.newaxis]


    boot_MSE = np.mean(np.mean((y_test-y_tilde_test)**2, axis=1, keepdims=True))
    boot_bias = np.mean((y_test-np.mean(y_tilde_test, axis=1, keepdims=True))**2)
    boot_variance = np.mean(np.var(y_tilde_test, axis=1, keepdims=True))

    return boot_MSE, boot_bias, boot_variance

#Performs cross-validation
#degree: the degree of the polynomial
#X: design matrix
#y: corresponding data
#K: number of folds to cross-validate
#method: linear regression method (can be OLS, Ridge or Lasso)

#MSE_mean: mean of the calculated MSE values for each fold

def cross_validation(degree, X, y, K, method):

    beta_len = int((2 + degree)*(1 + degree)/2)
    X_split = np.array(np.array_split(X, K))
    Y_split = np.array(np.array_split(y, K))

    MSEs = np.zeros(K)

    for i in range(K): #Runs through every single fold
        X_test = X_split[i]
        Y_test = Y_split[i]
        X_train = np.concatenate((X_split[:i], X_split[(i+1):]))
        Y_train = np.concatenate((Y_split[:i], Y_split[(i+1):])).ravel()
        X_train = X_train.reshape(-1, beta_len)
        X_train, X_test = scale(X_train, X_test)
        if method == 'OLS':
            y_pred = OLS(X_train, X_test, Y_train, Y_test)[0]

        elif method == 'Ridge':
            lambda_vals = np.logspace(-10,5,100)
            y_pred = Ridge(X_train,X_test,Y_train,Y_test,lambda_vals)[0]

        elif method == 'Lasso':
            lambda_vals = np.logspace(-10,5,100)

            MSE_test_array = np.zeros(len(lambda_vals))
            Y_tilde_test_array = np.zeros([len(lambda_vals),X_test.shape[0]])

            for j, lambda_val in enumerate(lambda_vals):
                clf = Lasso(alpha=lambda_val).fit(X_train, Y_train)
                Y_tilde_test_array[j] = clf.predict(X_test)
                MSE_test_array[j] = MSE(Y_test,Y_tilde_test_array[j])

            y_pred = Y_tilde_test_array[np.argmin(MSE_test_array)]

        else:
            print("Not a valid regression method")
            sys.exit(0)

        MSEs[i] = MSE(Y_test, y_pred)

    return np.mean(MSEs)

# t: epochs*m+i
def learning_schedule(t, t0=5, t1=50):
    return t0/(t+t1)

# Stochastic Gradient Descent
# X: design matrix
# n: number of data points
# M: mini batch size
# epochs: number of iterations over mini batches

# Code based on the code in chapter 4 of Geron's textbook
# Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow

def SGD(X, y, n, M, epochs, cost_function = 'OLS', lambda_val = 0, gamma = 0, classes = 2):
    
    m = int(n/M)
    if cost_function == 'OLS':
        theta = np.random.randn(X.shape[1]) #Random initialization
        for epoch in range(epochs):
            for j in range(m):
                k = np.random.randint(n-M) #Index to pick random bin
                X_k = X[k:k+M]
                y_k = y[k:k+M]
                gradient = (2/n)*X_k.T@((X_k@theta)-y_k) #Derivative of (MSE) cost function
                gamma = learning_schedule(epoch * m + j)
                theta = theta - gamma * gradient
        return theta
    elif cost_function == 'Ridge':
        theta = np.random.randn(X.shape[1]) #Random initialization
        for epoch in range(epochs):
            for j in range(m):
                k = np.random.randint(n-M)
                X_k = X[k:k+M]
                y_k = y[k:k+M]
                gradient = (2/n)*X_k.T@((X_k@theta)-y_k) + 2*lambda_val*theta
                #gamma = learning_schedule(epoch * m + j)
                theta = theta - gamma * gradient
        return theta
    elif cost_function == 'Logistic':
        theta = np.random.randn(X.shape[1],classes) #Random initialization
        for epoch in range(epochs):
            print(epoch)
            gradient = np.zeros(classes)
            for j in range(n):
                likely = X[j]@theta
                for class_val in range(classes):
                    likelyhood = likely[class_val]/np.sum(likely)
                    gradient[class_val] += X[j, class_val]*((y[j]==class_val)-likelyhood)
            theta = theta - gamma * gradient
        return theta
    
def logistic_sigmoid(x):
    return np.exp(x)/(1+np.exp(x))

# x: function input
# w: weights
# b: bias
def activation_function(x, w, b):
        z = x @ w + b
        return logistic_sigmoid(z)


#x: function input
#N_n: nodes in layer
#N_l: number of layers
#w: weights
#b: bias

def FFNN(x, N_n, N_l, *args):

    if len(args) == 2:
        w = args[0]
        b = args[1]

    if (args) == ():
        w = np.random.rand(len(x),N_l)
        b = np.random.rand(N_l)

    for i in range(N_l):
        Z = activation_function(x, w, b)

    predict = np.argmax(Z,axis = 1)
    y = 1
    return y

# Backward propagation
def backward_propagation(X_train, Y_train, w, b):
    z = np.dot(w.T, X_train) + b
    Y_head = logistic_sigmoid(z)
    loss = - Y_train * np.log(Y_head) - (1 - Y_train) * np.log(1 - Y_head)
    
    cost = (np.sum(loss)) / X_train.shape[1] # x_train.shape[1] for scaling
  
    # backward propagation - gradients
    d_w = (np.dot(X_train, (
        (Y_head - Y_train).T))) / X_train.shape[1] 
    d_b = np.sum(
        Y_head-Y_train) / X_train.shape[1]
    return cost, d_w, d_b

def accuracy_score(y, t):
    n = len(t)
    sum = 0
    for i in range(n):
        sum += y[i] == t[i]
    return sum/n


#Used for debugging, though the code for debugging has since been removed.
if __name__ == '__main__':
    print("\nYou accidentally ran the function file instead of an actual file.")