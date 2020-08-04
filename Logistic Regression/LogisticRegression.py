import random
from math import e
import math
import Preprocessing as PRE
import numpy as np

class LogisticRegression:

    def __init__(self, observations):
        self.observations = observations
        self.Y = observations[0]
        self.X = observations[1]
        self.THETAS = list()
        for _ in self.X:
            #self.THETAS.append(random.randint(-1,1))
            self.THETAS.append(0)
        self.pre_processedX = self.X.copy()


    def prediction(self, X):

        prediction = 0

        for index, theta in enumerate(self.THETAS):

            # ---------- THETA*X ----------
            prediction += theta * X[index]

        return prediction

    def LogisticFunction(self, X):

        num = 1
        den = 1 + e ** (-(self.prediction(X)))
        return num / den

    def RowX(self, row):

        X = list()

        for x in self.X:
            X.append(x[row])

        return X


    def Probability_row(self, row):            # preleva i valori di x per la specifica riga

        cost = 0
        X = self.RowX(row)
        y = self.Y[row]

        if y == 1:
            cost = self.LogisticFunction(X)
        if y == 0:
            cost = 1 - self.LogisticFunction(X)

        return cost


    def Probability(self):

        cost = 1

        for row, y in enumerate(self.Y):

            cost *= self.Probability_row(row)

        return cost


    def CostFunction_row(self, row):

        y = self.Y[row]
        cost = 0

        if y == 1:
            cost = math.log(self.Probability_row(row))

        elif y == 0:
            cost = 1 - math.log(self.Probability_row(row))
        return cost

    def CostFunction(self, _labda = 0):

        cost = 0

        for row in range(len(self.Y)):

            cost += self.CostFunction_row(row)

        cost = -cost / len(self.Y)
        
        cost += (_labda / 2 * len(self.Y)) * self.regularization_term()

        return cost

    def prediction_error_row(self, row):

        X = self.RowX(row)
        logistic = self.LogisticFunction(X)
        error = logistic - self.Y[row]

        return error


    def j_gradient_row(self, row):  # restituisce una lista dei gradienti di J calcolati per ogni teta

        update = list()

        prediction_error = self.prediction_error_row(row)

        # ---------- update i = gradient * xi ----------
        for index, X in enumerate(self.X):
            gradient = prediction_error * X[row]
            update.append(gradient)

        return update

    def j_gradient(self, start_row = 0, end_row = None):

        if end_row == None:

            end_row = len(self.Y)

        gradient = list()

        # ---------- inizializzo a zero il valore dell'gradient per ogni teta ----------

        for _ in range(len(self.THETAS)):

            gradient.append(0)

        for row in range(start_row, end_row):            # per tutte le righe del dataset da start row a end row

            update = self.j_gradient_row(row)

            gradient = [x + y for x, y in zip(gradient, update)]  # somma tra elementi corrispondenti di liste di interi

        return gradient


    def new_thetas(self, alfa,  _lambda = 0, start_row = 0, end_row = None):           # rows è il numero di righe del dataset su cui siamo lavorando, di default è tutto il dataset

        if end_row == None:
            end_row = len(self.Y)

        theta_new = list()
        rows = end_row - start_row

        update = self.j_gradient(start_row, end_row)

        for index, theta in enumerate(self.THETAS):

            _theta_new = alfa * update[index] / rows
            
            if index == 0:
                _theta_new = theta - _theta_new
                
            else:
                _theta_new = theta * (1 - ((alfa * _lambda) / rows)) - _theta_new

            theta_new.append(_theta_new)

        return theta_new

    def batchGD(self, alfa, iterations, _lambda = 0):

        for _ in range(iterations):
            new_thetas = self.new_thetas(alfa, _lambda)
            self.THETAS = new_thetas

        print('DOPO ',iterations,' ITERAZIONI: J = ', self.CostFunction())

        return self.THETAS

    def predict_M(self, X):

        return np.dot(X,self.THETAS)

    def solution_zscore(self, solution):

        new_solution = list()
        new_x = 0
        for index, x in enumerate(solution):
            new_x = 0
            new_x = (x - PRE.average(self.pre_processedX[index+1])) / (PRE.standard_deviation(self.pre_processedX[index+1]))
            new_solution.append(new_x)

        return new_solution

    def regularization_term(self):

        sum = 0
        for theta in self.THETAS[1::]:
            sum += theta ** 2

        return sum
