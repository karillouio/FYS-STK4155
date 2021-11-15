from functions import * 

#This is the solution to part a) of project 2

n_poly = 40
MSE_OLS_array = np.zeros(n_poly)
MSE_SGD_array = np.zeros(n_poly)
MSE_SGD_sklearn_array = np.zeros(n_poly)
poly_degs = np.arange(1,n_poly+1)

for poly_deg in poly_degs:
    print(poly_deg)
    N = 2000 #Number of data points
    noise = 0.2 #Factor of noise in data

    x_y = np.random.rand(N, 2) #Create random function parameters
    x = x_y[:,0]
    y = x_y[:,1]


    X = design_matrix_2d(poly_deg, x, y) #Create design matrix

    z = FrankeFunction_noise(x,y,noise) #Corresponding Franke Function val w/ noise

    X_train, X_test, z_Train, z_Test = train_test_split(X, z, test_size=0.2) #Split data into training and testing set
    X_train, X_test = scale(X_train, X_test) #Properly scale the data

    #Theta from our OLS
    z_tilde_train, z_tilde_test, theta_ols = OLS(X_train, X_test, z_Train, z_Test)


    #SGD from SKLearn
    sgd_reg = SGDRegressor(max_iter=200, penalty='l2', eta0=0.1, loss="squared_loss")
    #sgd_reg.fit(x, y.ravel())
    skl_sgd_model = sgd_reg.fit(X_train, z_Train)
    #print('Theta from sklearn SGD: ', a.coef_,'\n')

    #Our SGD
    M = 2 #Minibatch size
    epochs = 500

    theta_own_SGD = SGD(X_train, z_Train, N, M,  epochs)

    #print('Theta from own SGD: ',theta_own_SGD)

    MSE_OLS = metric.mean_squared_error(z_Test, z_tilde_test)
    MSE_SGD_own = metric.mean_squared_error(z_Test, X_test@theta_own_SGD)
    MSE_SGD_SKLearn = metric.mean_squared_error(z_Test, skl_sgd_model.predict(X_test))

    MSE_OLS_array[poly_deg-1] = MSE_OLS
    MSE_SGD_array[poly_deg-1] = MSE_SGD_own
    MSE_SGD_sklearn_array[poly_deg-1] = MSE_SGD_SKLearn


plt.plot(poly_degs, MSE_OLS_array, label = "OLS")
plt.plot(poly_degs, MSE_SGD_array, label = "SGD")
#plt.plot(polydegs, MSE_SGD_sklearn_array, label = "SKLearn")
plt.grid();
plt.legend();
plt.semilogy()
plt.xlabel("Complexity of model", fontsize = "large")
plt.ylabel("MSE", fontsize="large")
plt.title("MSE vs. complexity of model", fontsize = "x-large")
plt.show()