import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from NeuralNetwork import FFNeuralNetwork

# ensure the same random numbers appear every time
np.random.seed(271)

tumors = datasets.load_breast_cancer()

info = tumors.data; targets = tumors.target

info_train, info_test, targets_train, targets_test = train_test_split(info, targets, train_size=0.8)

#finding min and maximum values of each column in the test dataset
minmax = []
for index in range(30):
    colvals = [] #holds all column values
    for val in info_train:
        colvals.append(val[index])
    max1 = max(colvals) #max value of each column
    min1 = min(colvals) # min value of each column
    listcreated = []
    listcreated.append([min1 ,max1]) #holds all min and max values for each column
    minmax.append(listcreated[0]) 

    
#normalizing data sets to hold values between 0 and 1
for values in info_train:
    length_req = len(values)
    for index in range(length_req):
        firstsub = values[index] - minmax[index][0]
        secondsub = minmax[index][1] - minmax[index][0]
        values[index] = (firstsub) / (secondsub)

#finding min and maximum values of each column in the test dataset
minmax = []
for index in range(30):
    colvals = [] #holds all column values
    for val in info_test:
        colvals.append(val[index])
    max1 = max(colvals) #max value of each column
    min1 = min(colvals) # min value of each column
    listcreated = []
    listcreated.append([min1 ,max1]) #holds all min and max values for each column
    minmax.append(listcreated[0]) 

    
#normalizing data sets to hold values between 0 and 1
for values in info_test:
    length_req = len(values)
    for index in range(length_req):
        firstsub = values[index] - minmax[index][0]
        secondsub = minmax[index][1] - minmax[index][0]
        values[index] = (firstsub) / (secondsub)


n = len(targets_train)

    
def accuracy_score_np(y, t):
    acc = 0
    for i in range(len(t)):
        if(y[i] == t[i]):
            acc = acc + 1

    return acc/len(y)

epochs = 100
batch_size = 100
gamma = np.logspace(-6,-1,6)
lambda_vals = np.logspace(-6,-1,6)
lambda_vals = np.append([0], lambda_vals)
hidden_neurons = 100
hidden_layers = 2

#Code for finding a good combination of gamma and lambda

for i, gamma_val in enumerate(gamma):
    for j, lambda_val in enumerate(lambda_vals):
        FFNN = FFNeuralNetwork(info_train, targets_train, hidden_neurons, hidden_layers, epochs, batch_size, gamma=gamma_val, lmbd=lambda_val, out_func='Sigmoid', n_outputs=1)
        targets_prev = FFNN.classification(info_test)
        FFNN.train()
        targets_predict = FFNN.classification(info_test)
        print('learning rate: ', gamma_val)
        print('Lambda: ', lambda_val)
        print("Accuracy score on test set: ", accuracy_score_np(targets_predict, targets_test))
        print('\n')


gamma_val = 0.01
lambda_val = 0.0001


FFNN = FFNeuralNetwork(info_test, targets_test, hidden_neurons, hidden_layers, epochs, batch_size, gamma=gamma_val, lmbd=lambda_val, out_func='Sigmoid', n_outputs=1)
targets_prev = FFNN.classification(info_test)
FFNN.train()
targets_predict = FFNN.classification(info_test)
confusion_matrix_nr = np.zeros((2,2))

print("Accuracy score on test set: ", accuracy_score_np(targets_predict, targets_test))

print(targets_predict)
print(targets_test)

for i in range(len(targets_predict)):
    if(targets_predict[i] == 0):
        if(targets_test[i] == 0):
            confusion_matrix_nr[0, 0] += 1
        else:
            confusion_matrix_nr[0, 1] += 1
    else:
        if(targets_test[i] == 0):
            confusion_matrix_nr[1, 0] += 1
        else:
            confusion_matrix_nr[1, 1] += 1
            

plt.matshow(confusion_matrix_nr, cmap='gray')
plt.xlabel('Correct result (benign / malignant)', fontsize='large')
plt.ylabel('Predicted result (benign / malignant)', fontsize='large')
plt.colorbar()
plt.show()
plt.title("Confusion matrix for neural network", fontsize="x-large")