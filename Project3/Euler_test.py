import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
plt.style.use('seaborn')


def initial_condition(x):
    return np.sin(np.pi*x)

#List of free variables
sim_length = 0.25 #Final value for t in the simulation
delta_x = 1/100
delta_t = 1/20000 #Delta t, must be less than twice the square of delta_x
bc1 = 0 #Boundary condition for x = 0
bc2 = 0 #Boundary condition for x = 1


#Check stability criterion
alpha = delta_t/(delta_x**2)
if alpha > 0.5:
    print("Warning: the stability criterion is not fulfilled, so the program will stop.")
    print("%g/(%g)^2 = %.2f > 0.5" % (delta_t, delta_x, delta_t/(delta_x**2)))
    sys.exit(0)

t_steps = np.arange(0 , sim_length + delta_t , delta_t) #t values
x_points = np.arange(0 , 1 + delta_x , delta_x) #x values
frames = np.zeros([t_steps.shape[0],x_points.shape[0]]) #2D array for the calculated values

function_val = initial_condition(x_points)
frames[0] = function_val #Initial condition

print("\n")

prev_val = np.copy(function_val)
for j,t in enumerate(tqdm(t_steps[:-1])): #Loop over all times t, calculate u(x,t) at each one
    new_val = np.zeros(x_points.shape[0]) #Create empty array for the u(x) values at time t
    new_val[0] = bc1 #Boundary conditions
    new_val[-1] = bc2
    for i in range(1, x_points.shape[0]-1): #Loop over all positions x
        new_val[i] = prev_val[i] + alpha*(prev_val[i+1]+prev_val[i-1]-2*prev_val[i])
    frames[j+1] = prev_val
    prev_val = np.copy(new_val)


points_x, points_t = np.meshgrid(x_points, t_steps) #Calculate the exact value at each point (x,t)
true_val = np.sin(np.pi*points_x)*np.exp(-np.pi**2*points_t)

plt.figure() #Plot u(x,t) at different times t
plt.plot(x_points, frames[0], label="Initial")
plt.plot(x_points, frames[500], label="Middle")
plt.plot(x_points, frames[-1], label="Final")
plt.legend(fontsize='x-large')
plt.xlabel("Position (x) [Relative units]",fontsize='x-large')
plt.ylabel("u(x,t) at fixed times (t)",fontsize='x-large')
plt.title("Initial, middle and final state for simulation of length $t = %.2f$\n $\Delta x = \\frac{1}{%i}$" % (sim_length, delta_x**(-1)),fontsize='x-large')

plt.figure() #Contour plot for all points (x,t)
plt.contourf(x_points,t_steps,frames,levels=50,cmap='inferno')
plt.xlabel("Position (x) [Relative units]",fontsize='x-large')
plt.ylabel("Time (t) [Relative units]",fontsize='x-large')
plt.title("$u(x,t)$ in 1D space and time\n$\Delta x = \\frac{1}{%i}$" % (delta_x**(-1)),fontsize='x-large')
plt.colorbar().set_label("$u(x,t)$",fontsize='x-large')

plt.figure() #Plot absolute difference
plt.contourf(points_x ,points_t, np.abs(true_val-frames), levels=50, cmap='inferno')
plt.xlabel("Position (x) [Relative units]", fontsize='x-large')
plt.ylabel("Time (t) [Relative units]", fontsize='x-large')
plt.title("Absolute difference between analytic and numerical solution\n$\Delta x = \\frac{1}{%i}$" % (delta_x**(-1)), fontsize='x-large')
plt.colorbar().set_label("Absolute difference", fontsize='x-large')

plt.show()