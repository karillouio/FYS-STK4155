from functions import *
import time

n_poly = 40
MSE_OLS_array = np.zeros(n_poly)
t_OLS_array = np.zeros(n_poly)
MSE_SGD_array = np.zeros(n_poly)
t_SGD_array = np.zeros(n_poly)

polydegs = np.arange(20, n_poly+1)

for polydeg in polydegs:
    print(polydeg)
    N = 100 #Number of data points
    noise = 0.2 #Factor of noise in data

    x_y = np.random.rand(N,2) #Create random function parameters
    x = x_y[:,0]; y = x_y[:,1]


    X = design_matrix_2d(polydeg, x, y) #Create design matrix

    z = FrankeFunction_noise(x, y, noise) #Corresponding Franke Function val w/ noise

    X_train, X_test, z_Train, z_Test = train_test_split(X, z, test_size=0.2) #Split data into training and testing set
    X_train, X_test = scale(X_train, X_test) #Properly scale the data

    #Theta from our OLS
    t0 = time.time_ns()
    z_tilde_test = OLS(X_train, X_test, z_Train, z_Test)[1]
    t_OLS_array[polydeg-1] = (time.time_ns() - t0)/1e9
    print(t_OLS_array[polydeg-1])

    #Our SGD
    M = 2 #Minibatch size
    epochs = 10*X.shape[1]
    t1 = time.process_time()
    theta_own_SGD = SGD(X_train, z_Train, N, M, epochs, cost_function="Ridge", lambda_val=0.0001, gamma=0.001) #chosen after the experiment
    t_SGD_array[polydeg-1] = time.process_time() - t1

    MSE_OLS = metric.mean_squared_error(z_Test, z_tilde_test)
    MSE_SGD_own = metric.mean_squared_error(z_Test, X_test@theta_own_SGD)

    MSE_OLS_array[polydeg-1] = MSE_OLS
    MSE_SGD_array[polydeg-1] = MSE_SGD_own

plt.figure()
plt.scatter(np.log(t_OLS_array), np.log(MSE_OLS_array), label="OLS")
plt.scatter(np.log(t_SGD_array), np.log(MSE_SGD_array), label="SGD (Ridge version)")
plt.grid(); plt.legend();
plt.xlabel("Time spent calculating [Log(s)]", fontsize='large')
plt.ylabel("Log(MSE)", fontsize='large')
plt.title("Time elapsed calculating vs MSE", fontsize="x-large")
plt.show()