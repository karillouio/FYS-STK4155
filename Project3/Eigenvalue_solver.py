from Neural_Network_Classes import Neural_Network_eigenvalue
import autograd.numpy as np
import matplotlib.pyplot as plt
import numpy.linalg
plt.style.use('seaborn')


init_array = np.random.rand(6)

hidden_neurons = [25,25]
num_iter = 20 #number of iterations
lambda_val = 0.01

#generate a symmetric matrix by adding a random matrix to its transpose
random_matrix = np.random.randn(init_array.shape[0], init_array.shape[0])
symmetric_matrix = 0.5*(random_matrix.T + random_matrix)

#solve using numpy's method
print(numpy.linalg.eig(symmetric_matrix))

solver = Neural_Network_eigenvalue(init_array, symmetric_matrix, hidden_neurons, num_iter, lambda_val)
solver.train()
eigenvector = solver.output_function(0)
eigenvalue = (eigenvector.T@symmetric_matrix@eigenvector)/(eigenvector.T@eigenvector)

