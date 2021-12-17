from Neural_Network_Classes import Neural_Network_PDE
import autograd.numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

#Data points (x,t)
Nx = 20; Nt = 20
x = np.linspace(0, 1 ,Nx)
t = np.linspace(0, 1, Nt)

#Parameters / design of NN
hidden_neurons = [100, 100]
num_iter = 200
lambda_val = 0.01


Solver = Neural_Network_PDE(x, t, hidden_neurons, num_iter, lambda_val) #Set up NN
Solver.train() #Train NN

#Store results
neural_approx = np.zeros((x.shape[0],t.shape[0]))
analytic_solution = np.zeros((Nx, Nt))
for i,x_point in enumerate(x):
    for j, t_point in enumerate(t):
        point = np.array([x_point, t_point])
        neural_approx[i,j] = Solver.output_function(point)
        analytic_solution[i,j] = Solver.analytic_function(point)

abs_diff = np.abs(neural_approx-analytic_solution)

plt.figure()
plt.title("Neural network solution, depth = %i" % np.shape(hidden_neurons)[0], fontsize='x-large')
plt.ylabel("Time (t) [Relative units]", fontsize='x-large')
plt.xlabel("Position (x) [Relative units]", fontsize='x-large')
plt.contourf(x, t, neural_approx, levels=50, cmap='inferno')
plt.colorbar().set_label("$u(x,t)$",fontsize='x-large')

plt.figure()
plt.title("Absolute difference between analytic solution and neural network", fontsize='x-large')
plt.ylabel("Time (t) [Relative units]",fontsize='x-large')
plt.xlabel("Position (x) [Relative units]",fontsize='x-large')
plt.contourf(x, t, abs_diff, levels=50, cmap='inferno')
plt.colorbar().set_label("Absolute difference",fontsize='x-large')

plt.show()