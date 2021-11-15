import numpy as np

class FFNeuralNetwork:
    def __init__(self, X, Y, hidden_neurons, hidden_layers, epochs, batch_size, gamma, lmbd, out_func, n_outputs):
        self.X_data_full = X
        self.Y_data_full = Y

        self.inputs = X.shape[0]
        self.features = X.shape[1]
        self.hidden_neurons = int(hidden_neurons)
        self.hidden_layers = int(hidden_layers)
        self.n_outputs = n_outputs

        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.iterations = self.inputs // self.batch_size
        self.gamma = gamma
        self.lmbd = lmbd
        self.out_func = out_func
        self.activ_func = 'Sigmoid'

        self.create_bias_and_weights()

    def create_bias_and_weights(self):
        self.weights_i = np.random.randn(self.features, self.hidden_neurons)*2/np.sqrt(self.features+self.hidden_neurons)
        self.bias_i = np.zeros(self.hidden_neurons) + 0.01

        self.weights_h = np.random.randn(self.hidden_layers-1, self.hidden_neurons, self.hidden_neurons)*2/np.sqrt(2*self.hidden_neurons)
        self.bias_h = np.zeros((self.hidden_layers-1, self.hidden_neurons)) + 0.01

        self.weights_o = np.random.randn(self.hidden_neurons, self.n_outputs)*2/np.sqrt(self.hidden_neurons+self.n_outputs)
        self.bias_o = np.zeros(self.n_outputs) + 0.01

    def activation_function(self, z, function):
        results = np.zeros(z.shape)

        if function == 'Sigmoid':
            return(1/(1+np.exp(-z)))
        if function == 'RELU':
            for j in range(len(z)):
                for i in range(len(z[0])):
                    results[j,i] = max(0.0, z[j,i])
            return results
        if function == 'Leaky_RELU':
            self.alpha = 0.01
            for j in range(len(z)):
                for i in range(len(z[0])):
                    if z[j,i] < 0.0:
                        results[j,i] = self.alpha*z[j,i]
                    else:
                        results[j,i] = z[j,i]
            return results
        if function == 'SoftMax':
            exp_term = np.exp(z)
            return exp_term/np.sum(exp_term, axis=1, keepdims=True)

    def derivatives(self,z, function):
        results = np.zeros(z.shape)
        if function == 'Sigmoid':
            return self.activation_function(z, function)*(1-self.activation_function(z, function))
        if function == 'RELU':
            for j in range(len(z)):
                for i in range(len(z[0])):
                    if z[j,i] > 0:
                        results[j,i] = 1
            return results
        if function == 'Leaky_RELU':
            self.alpha = 0.01
            for j in range(len(z)):
                for i in range(len(z[0])):
                    if z[j,i] < 0.0:
                        results[j,i] = self.alpha
                    else:
                        results[j,i] = 1
            return results
        if function == 'SoftMax':
            return self.activation_function(z, function)*(1-self.activation_function(z,function))

    def feed_forward(self):
        self.a_h = np.zeros((self.hidden_layers, self.input_train, self.hidden_neurons))
        self.z_h = np.zeros((self.hidden_layers-1, self.input_train, self.hidden_neurons))

        self.z_i = self.X_data@self.weights_i + self.bias_i
        self.a_h[0] = self.activation_function(self.z_i, self.activ_func)

        for i in range(self.hidden_layers-1):
            self.z_h[i] = self.a_h[i]@self.weights_h[i] + self.bias_h[i]
            self.a_h[i+1] = self.activation_function(self.z_h[i], self.activ_func)

        self.z_o = self.a_h[-1]@self.weights_o + self.bias_o

        self.outputs = self.activation_function(self.z_o, self.out_func)

    def feed_forward_out(self, X):
        z_i = X@self.weights_i + self.bias_i
        a_h = self.activation_function(z_i, self.activ_func)

        for i in range(self.hidden_layers-1):
            z_h = a_h@self.weights_h[i] + self.bias_h[i]
            a_h = self.activation_function(z_h, self.activ_func)

        z_o = a_h@self.weights_o + self.bias_o

        outputs = self.activation_function(z_o, self.out_func)

        return outputs

    def back_propagation(self):
        error_o = 2/self.n_outputs*(self.outputs-self.Y_data)*self.derivatives(self.z_o, function=self.out_func)
        error_h = error_o@self.weights_o.T * self.derivatives(self.z_h[-1], function=self.activ_func)

        self.weight_gradient_o = self.a_h[-1].T@error_o
        self.bias_gradient_o = np.sum(error_o, axis=0)

        self.weight_gradient_h = self.a_h[-2].T@error_h
        self.bias_gradient_h = np.sum(error_h, axis=0)

        if self.lmbd > 0.0:
            self.weight_gradient_o += self.lmbd*self.weights_o
            self.weight_gradient_h += self.lmbd*self.weights_h[-1]

        self.weights_o -= self.gamma*self.weight_gradient_o
        self.bias_o -= self.gamma*self.bias_gradient_o
        self.weights_h[-1] -= self.gamma*self.weight_gradient_h
        self.bias_h[-1] -= self.gamma*self.bias_gradient_h

        for j in range(self.hidden_layers-2, 0, -1):
            error_new = error_h@self.weights_h[j].T*self.derivatives(self.z_h[j-1], function=self.activ_func)#*self.a_h[j]*(1-self.a_h[j])

            self.weight_gradient_h = self.a_h[j-1].T@error_new
            self.bias_gradient_h = np.sum(error_h, axis=0)

            if self.lmbd > 0.0:
                self.weight_gradient_h += self.lmbd*self.weights_h[j-1]

            self.weights_h[j-1] -= self.gamma*self.weight_gradient_h
            self.bias_h[j-1] -= self.gamma*self.bias_gradient_h
            error_h = error_new

        error_i = error_h@self.weights_h[0].T*self.derivatives(self.z_i, function=self.activ_func)

        self.weight_gradient_i = self.X_data.T@error_i
        self.bias_gradient_i = np.sum(error_i, axis=0)

        if self.lmbd > 0.0:
            self.weight_gradient_i += self.lmbd*self.weights_i

        self.weights_i -= self.gamma*self.weight_gradient_i
        self.bias_i -= self.gamma*self.bias_gradient_i

    def predict(self,X):
        y = self.feed_forward_out(X)
        return y

    def classification(self,X):
        y = self.feed_forward_out(X)
        #print(y)
        #print(y.shape)
        #print(np.round(y))
        return (np.round(y))

    def train(self):
        indexes = np.arange(self.inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                random_indexes = np.random.choice(indexes, size=self.batch_size, replace=False)

                self.X_data = self.X_data_full[random_indexes]
                self.Y_data = self.Y_data_full[random_indexes]

                if self.Y_data.ndim < 2:
                    self.Y_data = np.expand_dims(self.Y_data, axis=1)

                self.input_train = self.X_data.shape[0]

                self.feed_forward()
                self.back_propagation()