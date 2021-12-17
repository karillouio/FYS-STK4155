from sklearn.model_selection import train_test_split
from functions import *
import matplotlib.pyplot as plt
from sklearn import tree
np.random.seed(272)

N = 2000
noise = 0.2 #Factor of noise in data

x_y = np.random.rand(N,2) #Create random function parameters
x = x_y[:,0]; y = x_y[:,1]

z = FrankeFunction_noise(x,y,noise) #Create the data

# Creating the inputs for the decision tree
def D2_desmat(x, y):
    n = len(x)
    matrix=np.zeros((n,2))
    for i in range(n):
        matrix[i,0] = x[i]
        matrix[i,1] = y[i]
    return matrix

X = D2_desmat(x,y)

X_train, X_test, z_train, z_test = train_test_split(X, z, train_size=0.8)


this_max_depth = 30
MSE_ = np.zeros(this_max_depth)#Create empty array to be filled with interesting data
bias_ = np.zeros(this_max_depth)#Create empty array to be filled with interesting data
var_ = np.zeros(this_max_depth)#Create empty array to be filled with interesting data
boots = 30

#Bootstrap function for decision tree
def bootstrap_tree(X_train, X_test, y_train, y_test, n, this_depth):

    y_tilde_test = np.empty((y_test.shape[0], n))

    for i in range(n):
        random_indexes = np.random.randint(0, len(X_train), len(X_train))
        
        X = X_train[random_indexes]
        Y = y_train[random_indexes]
        clf = tree.DecisionTreeRegressor(max_depth = this_depth + 1)
        clf = clf.fit(X, Y)
        
        y_tilde_test[:,i] = clf.predict(X_test)


    y_test = y_test[:,np.newaxis]


    boot_MSE = np.mean(np.mean((y_test-y_tilde_test)**2, axis=1, keepdims=True))
    boot_bias = np.mean((y_test-np.mean(y_tilde_test, axis=1, keepdims=True))**2)
    boot_variance = np.mean(np.var(y_tilde_test, axis=1, keepdims=True))

    return boot_MSE, boot_bias, boot_variance


for depth in range(this_max_depth):
    MSE_[depth], bias_[depth], var_[depth] = bootstrap_tree(X_train, X_test, z_train, z_test, boots, depth + 1)

plt.plot(range(this_max_depth) + 1*np.ones(this_max_depth), MSE_, label = 'MSE')
plt.plot(range(this_max_depth) + 1*np.ones(this_max_depth), bias_, label = 'Bias')
plt.plot(range(this_max_depth) + 1*np.ones(this_max_depth), var_, label = 'Variance')
plt.title('Bias-variance trade-off, decision tree', fontsize='x-large')
plt.ylabel('Error', fontsize='large')
plt.xlabel('Max depth', fontsize='large')
plt.show()

