import autograd.numpy as np
from autograd import jacobian, hessian, grad
from tqdm import tqdm

class Neural_Network_PDE: #Neural Network for PDE Solving
    def __init__(self, X, t, num_neurons, epochs, lambda_val):
        #initializing the solver class
        self.X = X
        self.t = t
        self.lambda_val = lambda_val
        self.num_neurons = num_neurons
        self.N_neurons = np.size(self.num_neurons)
        self.N_outputs = 1
        self.iter = epochs

        self.initialize_deep_param()

    def initialize_deep_param(self):
        #initialize random values for the deep parameters
        self.N_hidden = np.size(self.num_neurons)

        self.P = [None]*(self.N_hidden+1)
        self.P[0] = np.random.randn(self.num_neurons[0], 2+1)
        for i in range(1, self.N_hidden):
            self.P[i] = np.random.randn(self.num_neurons[i], self.num_neurons[i-1] + 1)

        self.P[-1] = np.random.randn(1, self.num_neurons[-1] + 1)
        
    #activation function
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def initial_function(self, x):
        return np.sin(np.pi*x)

    def trial_function(self, point, P):
        x,t = point
        return (1-t)*self.initial_function(x) + x*(1-x)*t*self.feed_forward(P,point)

    def cost_function(self, P, X, T):
        cost_sum = 0

        gt_jacobi_function = jacobian(self.trial_function)
        gt_hessian_function = hessian(self.trial_function)

        for x in X:
            for t in T:
                point = np.array([x, t])

                #gt = self.trial_function(point, P)
                gt_jacobi = gt_jacobi_function(point, P)
                gt_hessian = gt_hessian_function(point, P)

                gt_dt = gt_jacobi[1]
                gt_dx2 = gt_hessian[0][0]

                err = (gt_dt - gt_dx2)**2
                cost_sum += err
        return cost_sum /( np.size(X)*np.size(T) )

    def feed_forward(self, P, point):
        num_coords = np.size(point,0)
        x = point.reshape(num_coords, -1)

        num_values = np.size(x,1)

        x_input = x
        x_prev = x_input

        for l in range(self.N_hidden):
            w_hidden = P[l]

            x_prev = np.concatenate((np.ones((1, num_values)), x_prev), axis=0)

            z_h = np.matmul(w_hidden, x_prev)
            x_h = self.sigmoid(z_h)

            x_prev = x_h

        w_out = P[-1]

        x_prev = np.concatenate((np.ones((1,num_values)), x_prev), axis=0)

        z_out = np.matmul(w_out, x_prev)
        x_out = z_out

        return x_out[0][0]

    def output_function(self, point):
        u = self.trial_function(point, self.P)
        return u

    def analytic_function(self, point):
        x, t = point
        return np.sin(np.pi*x)*np.exp(-np.pi**2*t)

    #back propagation 
    #find the gradient of the cost function and update the deep parameters
    def train(self):
        cost_function_grad = grad(self.cost_function, 0)
        for j in tqdm(range(self.iter)):
            cost_grad = cost_function_grad(self.P, self.X, self.t)

            for l in range(self.N_hidden+1):
                self.P[l] = self.P[l] - self.lambda_val*cost_grad[l]

class Neural_Network_eigenvalue:
    def __init__(self, X, matrix, num_neurons, epochs, lambda_val):
        #initializing the solver class
        self.X0 = X
        self.X = X
        self.lambda_val = lambda_val
        self.num_neurons = num_neurons
        self.N_neurons = np.size(self.num_neurons)
        self.N_outputs = self.X0.shape[0]
        self.iter = epochs
        self.matrix = matrix

        self.initialize_deep_param()

    def initialize_deep_param(self):
        #initialize random values for the deep parameters
        self.N_hidden = np.size(self.num_neurons)

        self.P = [None]*(self.N_hidden+1)
        self.P[0] = np.random.randn(self.num_neurons[0], 2+1)
        for i in range(1, self.N_hidden):
            self.P[i] = np.random.randn(self.num_neurons[i], self.num_neurons[i-1] + 1)

        self.P[-1] = np.random.randn(self.N_outputs, self.num_neurons[-1] + 1)

    #activation function
    def sigmoid(self, z):        
        return 1/(1 + np.exp(-z))

    def initial_function(self, x):
        return np.sin(np.pi*x)

    def trial_function(self, t, P):
        return np.exp(-t)*self.X0+(1-np.exp(-t))@self.feed_forward(P,self.X)

    def f(self,x):
        return (x.T @ x * self.matrix + (1- x.T@self.matrix@x)*np.identity(x.shape[0]))@x

    def cost_function(self, P, T):
        cost_sum = 0
        jacob = jacobian(self.trial_function)
        for t_point in T:
            gt = self.trial_function(t_point, P)
            gt_jacobi = jacob(t_point, P)
            gt_dt = gt_jacobi

            err = (gt_dt + gt - self.f(self.X))**2
            cost_sum += err
        return cost_sum /(np.size(T))

    def feed_forward(self, P, t):

        num_values = np.size(t, 1)

        x_input = t
        x_prev = x_input

        for l in range(self.N_hidden):
            w_hidden = P[l]

            x_prev = np.concatenate((np.ones((1, num_values)), x_prev), axis=0)

            z_h = np.matmul(w_hidden, x_prev)
            x_h = self.sigmoid(z_h)

            x_prev = x_h

        w_out = P[-1]

        x_prev = np.concatenate((np.ones((1, num_values)), x_prev), axis=0)

        z_out = np.matmul(w_out, x_prev)
        x_out = z_out

        return x_out[0][0]

    def output_function(self, t):
        return self.trial_function(t, self.P)

    #back propagation 
    #find the gradient of the cost function and update the deep parameters
    def train(self):
        cost_function_grad = grad(self.cost_function, 0)
        for j in tqdm(range(self.iter)):
            cost_grad = cost_function_grad(self.P, self.X)
            print(self.cost_function)

            for l in range(self.N_hidden + 1):
                self.P[l] = self.P[l] - self.lambda_val*cost_grad[l]
            self.X = self.output_function(self.X)