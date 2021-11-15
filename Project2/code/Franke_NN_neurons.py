from sklearn.model_selection import train_test_split
from functions import *
import matplotlib.pyplot as plt
from NeuralNetwork import FFNeuralNetwork
np.random.seed(5513)

N = 100

noise = 0.1 #Factor of noise in data

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
neuron_test_number = 8
hidden_neurons = np.linspace(25, 200, neuron_test_number)
hidden_layers = 2
epochs = 10000
batch_size = 10
confusion_matrix = np.zeros([5,11])
gamma = 0.01
lambda_val = 0

X_train, X_test, z_train, z_test = train_test_split(X, z, train_size=0.8)

MSE_ = np.zeros(neuron_test_number)

for i in range(neuron_test_number):
    for j in range(1):
        FFNN = FFNeuralNetwork(X_train, z_train, hidden_neurons[i], hidden_layers, epochs, batch_size, gamma, lambda_val, out_func='Sigmoid', n_outputs=1)
        FFNN.train() # training the network
        z_pred = FFNN.predict(X_test) # predicting the test data
        z_predict = FFNN.predict(X_train[:2])
        #confusion_matrix[i,j] = MSE(z_test, z_pred)
        print(MSE(z_test, z_pred))
        MSE_[i] = MSE(z_test, z_pred)

FFNN = FFNeuralNetwork(X_train, z_train, hidden_neurons=25, hidden_layers=2, epochs=1000, batch_size=25, gamma=0.001, lmbd=0.1, out_func='Leaky_RELU', n_outputs=1)
z_prev = FFNN.predict(X_train[:2])
FFNN.train()
z_pred = FFNN.predict(X_train[:2])
#z_predict = FFNN.predict(X_test)
#print(z_train[:2], FrankeFunction_noise(X_train[:2,0], X_train[:2,1], 0))
#print(z_pred)
#print(z_prev)

#print(MSE(z_train[:2], z_pred))

#print(MSE_)
#print(hidden_layers)

plt.plot(hidden_neurons, MSE_, 'r-')
plt.title('MSE vs. number of hidden neurons', fontsize='x-large')
plt.ylabel('MSE', fontsize='large')
plt.xlabel('Hidden neurons', fontsize='large')
plt.show()