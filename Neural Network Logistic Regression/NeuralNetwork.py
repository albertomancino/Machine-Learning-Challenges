import numpy as np
from math import log
import Preprocessing
from math import sqrt

def create_network(hidden_layer_neurons, input_layer_neurons, output_layer_neurons):

    layers = len(hidden_layer_neurons) + 1  # +1 perchè bisogna includere il livello iniziale delle feature

    network = list()

    network.append(2 * np.random.random((hidden_layer_neurons[0], input_layer_neurons))-1)  # prima matrice di teta, tra il layer 1 e il 2

    for index, neurons in enumerate(hidden_layer_neurons[:-1]):
        start_layer_neurons = hidden_layer_neurons[index+1]
        end_layer_neurons = hidden_layer_neurons[index]

        network.append(2 * np.random.random(((start_layer_neurons, end_layer_neurons)))-1)

    network.append(2 * np.random.random((output_layer_neurons, hidden_layer_neurons[-1]))-1)  # prima matrice di teta, tra il layer 1 e il 2

    return network


class NeuralNetwork():

    def __init__(self, X, Y, neurons, alfa, _lambda):

        self.Y = np.array(Y)
        self.X = np.array(X)
        self.neurons = neurons
        self.layers = len(neurons)+1
        self.network = create_network(neurons, len(self.X[0]), len(self.Y[0]))
        self.bias = np.ones((self.layers, len(self.X)))
        self.bias_ = 1
        self.bias_weights = list()
        self._lambda = _lambda
        self.alfa = alfa

        for net in self.network:
            self.bias_weights.append(2*np.random.random((net.shape[0],1))-1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivation(self, x):
        return x*(1-x)

    def activation_value(self, layer_minus_one, previous_layer):


        previous_a = previous_layer

        theta_layer_matrix = self.network[layer_minus_one]

        z = theta_layer_matrix.dot(previous_a)

        # ---- aggiunta bias -----
        bias_contribution = self.bias_weights[layer_minus_one] * self.bias_
        z += bias_contribution

        a = self.sigmoid(z)
        return a

    def activation_output(self, a):

        return self.network[-1].dot(a)


    def forward_propagation(self, x = None):

        activation_values = list()

        if (x == None):
            a = self.X.T
        else:
            a = np.matrix(x).T

        activation_values.append(a)

        for layer in range(len(self.network)-1):

            a = self.activation_value(layer,a)
            activation_values.append(a)

        activation_values.append(self.activation_output(a))

        return activation_values

    def backward_propagation(self):

        activation_values = self.forward_propagation()

        deltas = list()
        delta = activation_values[-1] - self.Y.T
        deltas.append(delta)

        for layer in range(self.layers - 1,0,-1):
            delta = np.dot(self.network[layer].T,delta)
            delta = np.multiply(delta, activation_values[layer], 1 - activation_values[layer])
            deltas.append(delta)

        deltas.reverse()

        for index, net in enumerate(self.network):

            update = np.dot(deltas[index], activation_values[index].T)
            update = (update + self._lambda * net) / len(self.Y)

            bias = np.matrix(self.bias[index])
            bias_matrix = np.ones((deltas[index].shape[1],1))*self.bias_
            bias_update = np.dot(deltas[index], bias_matrix) / len(self.Y)

            self.network[index] -= self.alfa * update
            self.bias_weights[index] -= self.alfa * bias_update

    def regularization(self, _lambda):

        regularization = 0

        for matrix in self.network:

            squared = np.square(matrix)
            squared = np.sum(squared)
            regularization += squared

        return regularization * _lambda / 2*len(self.Y)

    def cost_function(self):

        a = self.forward_propagation()
        a = a[-1]
        J = 0

        for row in range(len(self.Y.T)):
            for col in range(len(self.Y.T[0])):
                J += (a[row][col] - self.Y.T[row][col])**2
        J = J / (2*len(self.Y))
        return J + self.regularization(self._lambda)



    def print_network(self):

        for net in self.network:
            print('\n', net)

    def print_cost(self):

        print(self.cost_function())

    def predict(self, X, means, std_devs):

        X = Preprocessing.zscore_norm_prediction(X, means, std_devs)
        solution = self.forward_propagation(X)[-1]

        print('----- SOLUTION -----')
        print(solution.item((0,0)))

    def fit(self, iterations):

        for it in range(iterations):

            self.backward_propagation()

    def hypothesis(self):

        return self.forward_propagation()[-1]

    def MAE(self):
        error = self.forward_propagation()[-1] - self.Y
        absolute_error = np.dot(error.transpose(),np.ones(len(error)))
        return (absolute_error / (2 * len(self.Y)))[0]


    def MSE(self):
        error = self.forward_propagation()[-1] - self.Y
        squared_error = np.dot(error.transpose(),error)
        return (squared_error / (2 * len(self.Y)))[0][0]

    def RMSE(self):
        return sqrt(self.MSE())

    def devT(self):

        Y = self.Y.transpose()
        ymean = Preprocessing.average(Y[0])
        deviation = Y - ymean
        deviationT = np.transpose(deviation)
        total_deviation = np.dot(deviation,deviationT)
        return total_deviation

    def devR(self):

        Y = self.Y.transpose()
        ymean = Preprocessing.average(Y[0])
        hypothesis = self.hypothesis()
        deviation = hypothesis - ymean
        deviationT = np.transpose(deviation)
        regression_deviation = np.dot(deviation,deviationT)
        return regression_deviation

    def squaredR(self):

        return self.devR()/self.devT()

