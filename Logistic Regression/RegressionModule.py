import Preprocessing as PRE
import numpy as np

class Regression:

    def __init__(self, observations):
        self.observations = observations
        self.Y = observations[0]
        self.X = observations[1]

        self.THETAS = list()
        for _ in self.X:
            self.THETAS.append(0)

        self.pre_processedY = self.Y
        self.pre_processedX = self.X.copy()

    def MeanSquaredError(self, _lambda = 0):

        error = 0

        for row, observation in enumerate(self.Y):
            hypotesis = 0

            for col, teta in enumerate(self.THETAS):

                feature = self.X[col][row]
                hypotesis += (teta * feature)

            single_error = (hypotesis - observation) ** 2
            error += single_error
            
        error += self.regularization_term() * _lambda

        error = error / (2 * len(self.Y))


        return error


    def prediction_error_row(self, row):

        error = 0

        # ---------- h (xi) : predizione sulla data riga ----------
        for index, theta in enumerate(self.THETAS):

            error += theta * self.X[index][row]

        # ----------  h * xi - yi : errore di predizione sulla data riga ----------
        error -= self.Y[row]

        return error


    def j_gradient_row(self, row):  # restituisce una lista dei gradienti di J calcolati per ogni teta

        gradient = 0
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


    def new_thetas(self, alfa, start_row = 0, end_row = None, _lambda = 0):           # rows è il numero di righe del dataset su cui siamo lavorando, di default è tutto il dataset

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

    def batchGD(self, alfa, iterations, _lambda=0):

        for _ in range(iterations):
            new_thetas = self.new_thetas(alfa, _lambda)
            self.THETAS = new_thetas

        print('DOPO ', iterations, ' ITERAZIONI: J = ', self.CostFunction())

        return self.THETAS

    def predict(self, X):

        value = 0

        for index, coeff in enumerate(self.THETAS):

            if index == 0:
                value += round(coeff, 8)
            else:
                value += coeff * X[index-1]
                print(coeff)
        return value

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
    
    def CostFunction(self):
        
        return self.MeanSquaredError()
    
    def regularization_term(self):
        
        sum = 0
        for theta in self.THETAS[1::]:
            
            sum += theta ** 2
        
        return sum

