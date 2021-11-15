# reading in all the import statements
from sklearn import datasets
import numpy as np
import math
from sklearn.model_selection import train_test_split
from functions import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ensure the same random numbers appear every time
np.random.seed(271)

#Download the breast cancer dataset from SKLibrary
tumors = datasets.load_breast_cancer()

#splitting the feature matrix as X and target matrix as Y
X = tumors.data
Y = tumors.target

#setting the learning rate
initial_learning_rate = 0.0002

#setting the # of iterations
iterations = 150

#using train test split to get values of x_train,x_test,y_train,y_test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=325) 

#finding min and maximum values of each column
minmax = []

for index in range(30):
    col_vals = [] #holds the column values
    for val in X_train:
        col_vals.append(val[index])
    max1 = max(col_vals) # max value of each column
    min1 = min(col_vals) # min value of each column
    list_ = []
    list_.append([min1,max1]) #appends min and max values of each column data
    minmax.append(list_[0])

#normalizing data set to hold values between 0 and 1
for every_val in X_train:
    length_req = len(every_val)
    for index in range(length_req):
        firstsub = every_val[index] - minmax[index][0] #first subtraction in formula for normalizing
        secondsub = minmax[index][1] - minmax[index][0] #second subtraction in formula for normalizing
        every_val[index] = (firstsub) / (secondsub) #dividing both values

#initiliazing empty weights array
theta = [] #creating a weights array
for index in range(31):
    theta.append(0)
 

#stochastic gradient descent update
def update(theta, x, y):
    mul1 = np.append(x,1) #add 1 to the created array
    predicted_y = theta @ mul1 #matrix multiplication function  
    predicted_y = logistic_sigmoid(predicted_y) #sigmoid function 
    sub1 = predicted_y - y
    theta = theta - initial_learning_rate * sub1 * mul1 #stochastic gradient formula
    return theta


for index in range(iterations):
    for dummy in range(X_train.shape[0]):
        theta = update(theta, X_train[index,:], Y_train[index]) #update all theta values

    
#finding min and maximum values of each column in the test dataset
minmax = []
for index in range(30):
    colvals = [] #holds all column values
    for val in X_test:
        colvals.append(val[index])
    max1 = max(colvals) #max value of each column
    min1 = min(colvals) # min value of each column
    listcreated = []
    listcreated.append([min1 ,max1]) #holds all min and max values for each column
    minmax.append(listcreated[0]) 

    
#normalizing data sets to hold values between 0 and 1
for values in X_test:
    length_req = len(values)
    for index in range(length_req):
        firstsub = values[index] - minmax[index][0]
        secondsub = minmax[index][1] - minmax[index][0]
        values[index] = (firstsub) / (secondsub)
        
        
#making predictions
predictions_test = []
for values in X_test:
    mul1 = np.append(values, 1)
    prediction = theta @ mul1 
    prediction = logistic_sigmoid(prediction)
    predictions_test.append(prediction)
    
predictions_train = []
for values in X_train:
    mul1 = np.append(values, 1)
    prediction = theta @ mul1 
    prediction = logistic_sigmoid(prediction)
    predictions_train.append(prediction)
   
   
#print(X_train)
#print(X_test)
#rounding each value to be exactly 0 or 1
for index in range(len(predictions_test)):
    predictions_test[index] = round(predictions_test[index])

#rounding each value to be exactly 0 or 1
for index in range(len(predictions_train)):
    predictions_train[index] = round(predictions_train[index])

accuracy_test = []
lenreq = len(Y_test)
for index in range(lenreq):
    if(Y_test[index] == predictions_test[index]):
        accuracy_test.append(1)
        
        
accuracy_train = []
lenreq = len(Y_train)
for index in range(lenreq):
    if(Y_train[index] == predictions_train[index]):
        accuracy_train.append(1)
    

accuracy_tr = sum(accuracy_train)/len(Y_train) 
print("Training accuracy: ", accuracy_tr*100,"%")    
    
accuracy_te = sum(accuracy_test)/len(Y_test) 
print("Test accuracy: ", accuracy_te*100,"%")

# compare to SKLearn's logistic regression
SKLearn_lr = LogisticRegression()
SKLearn_lr.fit(X_train, Y_train)

SKLearn_predict = SKLearn_lr.predict(X_test)


print("\nSKLearn accuracy: %f" %(accuracy_score(Y_test, SKLearn_predict) * 100))

confusion_matrix_nr = np.zeros((2,2))

for i in range(len(predictions_test)):
    if(predictions_test[i] == 0):
        if(Y_test[i] == 0):
            confusion_matrix_nr[0, 0] += 1
        else:
            confusion_matrix_nr[0, 1] += 1
    else:
        if(Y_test[i] == 0):
            confusion_matrix_nr[1, 0] += 1
        else:
            confusion_matrix_nr[1, 1] += 1


plt.matshow(confusion_matrix_nr, cmap='gray')
plt.xlabel('Correct result (benign / malignant)', fontsize='large')
plt.ylabel('Predicted result (benign / malignant)', fontsize='large')
plt.colorbar()
plt.show()
plt.title("Confusion matrix for logistic regression", fontsize="x-large")

