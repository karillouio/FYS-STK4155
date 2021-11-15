from functions import *

lambda_vals = np.logspace(-8,-1,8)
learning_rates = np.logspace(-3,0,4)

polydeg = 15 #Degree of polynomial fit
N = 2000 #Number of data points
noise = 0.2 #Noise in data

x_y = np.random.rand(N,2) #Create random function parameters
x = x_y[:,0]; y = x_y[:,1]

X = design_matrix_2d(polydeg,x,y) #Create design matrix
z = FrankeFunction_noise(x,y,noise) #Corresponding Franke Function val w/ noise

X_train, X_test, z_Train, z_Test = train_test_split(X,z,test_size=0.2) #Split data into training and testing set
X_train, X_test = scale(X_train, X_test) #Properly scale the data

matrixplot = np.zeros([lambda_vals.shape[0], learning_rates.shape[0]])

M = 2 #Minibatch size
epochs = 200

for lambda_index, lambda_ in enumerate(lambda_vals):
    print(lambda_index)
    for learnindex,learning_rate in enumerate(learning_rates):
        #print(learnindex)
        theta_own_SGD_Ridge = SGD(X_train, z_Train, N, M, epochs, cost_function="Ridge", lambda_val=lambda_, gamma=learning_rate)
        MSE_val = metric.mean_squared_error(z_Test, X_test@theta_own_SGD_Ridge)
        matrixplot[lambda_index, learnindex] = MSE_val

plt.matshow(matrixplot,cmap='gray',vmax=1)
plt.colorbar()
plt.xlabel("Learning rates",fontsize="x-large")
plt.ylabel("$\lambda$",fontsize="x-large")
plt.title("MSE of SGD (Ridge) for different learning rates\nand different hyperparameter $\lambda$\n", fontsize="x-large")
plt.yticks(np.arange(lambda_vals.shape[0]), lambda_vals)
plt.xticks(np.arange(learning_rates.shape[0]), learning_rates, rotation=90)
plt.show()