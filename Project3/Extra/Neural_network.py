from sklearn.model_selection import train_test_split
from functions import *
import matplotlib.pyplot as plt
from NeuralNetwork_class import FFNeuralNetwork
np.random.seed(272)

N = 2000

noise = 0.2 #Factor of noise in data

x_y = np.random.rand(N,2) #Create random function parameters
x = x_y[:,0]; y = x_y[:,1]

z = FrankeFunction_noise(x,y,noise) #Create the data

# Creating the inputs for the neural network
def D2_desmat(x, y):
    n = len(x)
    matrix=np.zeros((n,2))
    for i in range(n):
        matrix[i,0] = x[i]
        matrix[i,1] = y[i]
    return matrix

X = D2_desmat(x,y)

# Choosing hyperparameters
hidden_neurons = 125
hidden_layers = 2
epochs = 10000
batch_size = 10
confusion_matrix = np.zeros([5,11])
gamma = 0.01
lambda_val = 0
boots = 3

MSE_ = np.zeros(hidden_layers)#Create empty array to be filled with interesting data
bias_ = np.zeros(hidden_layers)#Create empty array to be filled with interesting data
var_ = np.zeros(hidden_layers)#Create empty array to be filled with interesting data

X_train, X_test, z_train, z_test = train_test_split(X, z, train_size=0.8)
X_train, X_test = scale(X_train, X_test)

for i in range(hidden_layers):
    
    y_tilde_test = np.zeros((z_test.shape[0], boots))
    
    for j in range(boots):
        random_indexes = np.random.randint(0, len(X_train), len(X_train))
        X = X_train[random_indexes]
        Y = z_train[random_indexes]
        
        FFNN = FFNeuralNetwork(X, Y, hidden_neurons, i+2, epochs, batch_size, gamma, lambda_val, out_func='Sigmoid', n_outputs=1)
        FFNN.train() # training the network
        print(y_tilde_test[:,j])
        y_tilde_test[:,j] = FFNN.predict(X_test).ravel()
        print(y_tilde_test[:,j])
    
    z_test = z_test[:,np.newaxis]
    
    MSE_[i] = np.mean(np.mean((z_test-y_tilde_test)**2, axis=1, keepdims=True))
    bias_[i] = np.mean((z_test-np.mean(y_tilde_test, axis=1, keepdims=True))**2)
    var_[i] = np.mean(np.var(y_tilde_test, axis=1, keepdims=True))
    

plt.plot(range(hidden_layers)+2*np.ones(hidden_layers), MSE_, label='MSE')
plt.plot(range(hidden_layers)+2*np.ones(hidden_layers), bias_, label='Bias')
plt.plot(range(hidden_layers)+2*np.ones(hidden_layers), var_, label='Variance')
plt.title('Bias-variance trade-off, neural network', fontsize='x-large')
plt.ylabel('Error', fontsize='large')
plt.xlabel('Hidden layers', fontsize='large')
plt.show()