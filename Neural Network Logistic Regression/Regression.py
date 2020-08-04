import pandas as pd
import numpy as np
from math import e
from math import log
from math import sqrt
import Preprocessing as pre

class Regression():

    def __init__(self, X: pd.DataFrame, Y: pd.DataFrame):

        self.Y = np.array(Y)
        self.X = np.array(X)
        self.X = (np.c_[ np.ones((len(Y),1)) ,self.X])  #aggiunta x0
        self.thetas = np.zeros((1, len(self.X[0])))

    def hypothesis(self, start_row = 0, end_row = None):

        if end_row == None:
            end_row = len(self.Y)
        return np.dot((self.X)[start_row:end_row],self.thetas.transpose())

    def MAE(self):
        error = self.hypothesis() - self.Y
        absolute_error = np.dot(error.transpose(),np.ones(len(error)))
        return (absolute_error / (2 * len(self.Y)))[0]


    def MSE(self):
        error = self.hypothesis() - self.Y
        squared_error = np.dot(error.transpose(),error)
        return (squared_error / (2 * len(self.Y)))[0][0]

    def RMSE(self):
        return sqrt(self.MSE())


    def L2Norm(self, _lambda):

        L2Norm = 0
        for theta in self.thetas[1::]:
            L2Norm += theta**2
        L2Norm *= (_lambda / (2 * len(self.Y)))
        return L2Norm

    def cost_function(self, _lambda = 0):

        L2Norm = 0
        for theta in self.thetas[0][1:]:
            L2Norm += theta**2
        L2Norm *= (_lambda / (2 * len(self.Y)))
        return (self.MSE() + L2Norm)

    def new_thetas(self, alfa, _lambda = 0, start_row = 0, end_row = None):

        if end_row is None:
            end_row = len(self.Y)
        error = self.hypothesis(start_row, end_row) - self.Y[start_row:end_row]
        X = self.X[start_row: end_row]
        delta_gradient = np.dot(error.transpose(), X)
        update = delta_gradient * (- alfa) / (end_row - start_row)

        regularization = np.ones(self.thetas.shape)
        regularization -= (alfa * _lambda / (end_row - start_row))
        return np.multiply(self.thetas, regularization) + update

    def batch_gradient_descent(self, alfa, _lambda, iterations):

        for iter in range(iterations):
            #print('J: ', self.cost_function(_lambda))
            self.thetas = self.new_thetas(alfa, _lambda)
        #print('After ',iterations,' iterations J = ', self.cost_function(_lambda))
        return self.thetas

    def stochastic_gradient_descent(self, alfa, _lambda, iterations):

        for iter in range(iterations):
            #print('J: ', self.cost_function(_lambda))

            for row in range(len(self.Y)):
                self.thetas = self.new_thetas(alfa, _lambda, row, row+1)

        #print('After ',iterations,' iterations J = ', self.cost_function(_lambda))
        return self.thetas

    def minibatch_gradient_descent(self, alfa, _lambda, iterations, b):

        for iter in range(iterations):
            #print('J: ', self.cost_function(_lambda))

            row = 0
            while row < len(self.Y):
                end_row = row + b
                if end_row <= len(self.Y):
                    self.thetas = self.new_thetas(alfa, _lambda, row, end_row)
                else:
                    end_row = len(self.Y)
                    self.thetas = self.new_thetas(alfa, _lambda, row, end_row)
                row = end_row

        #print('After ',iterations,' iterations J = ', self.cost_function(_lambda))
        return self.thetas

    def predict(self, X:list):

        localX = X.copy()
        localX.insert(0,1)
        X = np.array(localX)
        return np.dot(self.thetas, localX)

    def print_solution(self, thetas):

        print('\nSOLUTION THETAS: ')
        for index in range(len(thetas.tolist()[0])):
            print('θ', index, '=', thetas.tolist()[0][index])

    def devT(self):

        Y = self.Y.transpose()
        #Y = [[1,-1,2,-2,4,-4,7]]
        ymean = pre.average(Y[0])
        deviation = Y - ymean
        deviationT = np.transpose(deviation)
        total_deviation = np.dot(deviation,deviationT)
        return total_deviation

    def devR(self):

        Y = self.Y.transpose()
        #Y = [[1,-1,2,-2,4,-4,7]]
        ymean = pre.average(Y[0])
        hypothesis = self.hypothesis().T
        deviation = hypothesis - ymean
        deviationT = np.transpose(deviation)
        regression_deviation = np.dot(deviation,deviationT)
        return regression_deviation

    def squaredR(self):

        return self.devR()/self.devT()

class LogisticRegression(Regression):

    def __init__(self, X: pd.DataFrame, Y: pd.DataFrame):
        super().__init__(X, Y)

    def value_to_predict(self, value):

        for index in range(len(self.Y)):
            if self.Y[index] == value:
                self.Y[index] = 1
            else:
                self.Y[index] = 0

    def logistic_function(self, X):

        X = np.matrix([X])
        exp = -(self.thetas.dot(X.transpose()))
        #print(exp)
        num = 1
        den = 1 + e ** (exp.item((0,0)))
        #print(num/den)
        #input()
        return num / den

    def hypothesis(self, start_row = 0, end_row = None):

        hypothesis = list()
        if end_row == None:
            end_row = len(self.Y)

        for x in self.X[start_row:end_row]:
            hypothesis.append(self.logistic_function(x))

        hypothesis = np.matrix([hypothesis])
        return hypothesis.transpose()

    def threshold(self, start_row = 0, end_row = None):

        h = self.hypothesis()
        predictions = list()
        threshold = 0.5

        for hypo in h:

            if hypo.item((0,0)) > threshold:
                predictions.append(1.0)
            else:
                predictions.append(0.0)

        return predictions

    def cost_function(self, _lambda = 0):

        J = 0
        for index, h in enumerate(self.hypothesis()):
            if self.Y[index] == 0:
                J += log(1-h)
            else:
                J += log(h)
        return -J/self.Y.shape[0] + self.L2Norm(_lambda)


    def predict(self, X:list):

        localX = X.copy()
        print('\n\n VALUE TO PREDICT: ',localX)
        localX.insert(0,1)

        return self.logistic_function(localX)

    def confusion_matrix(self):

        predictions = self.threshold()
        Y = self.Y.T[0].tolist()

        print(Y)
        print(predictions)

        confusionmatrix = [[0,0],[0,0]]

        for index in range(len(Y)):

            y = Y[index]
            y_star = predictions[index]

            if y == 1:
                if y_star == 1:
                    confusionmatrix[0][0] += 1 #TRUE POSITIVE
                else:
                    confusionmatrix[0][1] += 1 #FALSE NEGATIVE
            else:
                if y_star == 1:
                    confusionmatrix[1][0] += 1 #FALSE POSITIVE
                else:
                    confusionmatrix[1][1] += 1 #TRUE NEGATIVE
        return confusionmatrix

    def confusion_indeces(self):

        confusion_matrix = self.confusion_matrix()

        TP = confusion_matrix[0][0]
        FN = confusion_matrix[0][1]
        FP = confusion_matrix[1][0]
        TN = confusion_matrix[1][1]

        accuracy = (TP+TN)/(TP+TN+FP+FN)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        TNR = TN/(FP+TN)
        error_rate = 1-accuracy
        F_measure = 2*precision*recall/(precision+recall)
        FPR = 1- TNR

        print('----- LOGISTIC REGRESSION\'S INDECES -----')
        print('Accuracy = ',accuracy)
        print('Precision = ',precision)
        print('Recall = ',recall)
        print('Specifity',TNR)
        print('Error rate = ',error_rate)
        print('F-measure = ',F_measure)
        print('False Positive Rate = ',FPR)