import RegressionModule as REG
import LogisticRegression as LG
import Preprocessing as PRE
import PlotPrint as PP
import numpy as np


class UnivariateRegressionModel():
    
    def __init__(self, csv, variables_indeces, classification_values=None):
        # ---------- ESTRAPOLA DATI ----------
        values = csv.giveme_values_indeces(variables_indeces)

        # ---------- MODELLO ----------

        self.model = REG.Regression(values)          # REGRESSIONE
        self.thetas = None


    def Normalization(self, flag):

        if flag == 'Y':
    
            for index, x in enumerate(self.model.X):
                if index != 0:
                    print('NORMALIZZAZIONE X', index)
                    self.model.X[index] = PRE.zscore_norm(x)
                
                
    def fit(self, alfa, iterations, _lambda = 0):

        print('J ALLA PRIMA ITERAZIONE = ', self.model.CostFunction())
        self.thetas = self.model.batchGD(alfa, iterations, _lambda)


    def print_solution(self):
        print('TETA DEL MODELLO: ', self.thetas)

        PP.print_graph(self.model.Y, self.model.X[1])
        PP.Polynomial(self.thetas, (min(self.model.X[1]) * 0.5), (max(self.model.X[1]) * 1.5))
        PP.show()
    
    def prediction(self, input):

        predict = self.model.solution_zscore(input)
        predict = np.insert(predict, 0, 1)

        print('VALORE PREDETTO = ', self.model.predict_M(predict))


class MultivariateRegressionModel():

    def __init__(self, csv, variables_indeces, classification_values=None):
        # ---------- ESTRAPOLA DATI ----------
        values = csv.giveme_values_indeces(variables_indeces)
        
        # ---------- MODELLO ----------

        self.model = REG.Regression(values)  # REGRESSIONE
        self.thetas = None

    def Normalization(self, flag):

        if flag == 'Y':

            for index, x in enumerate(self.model.X):
                if index != 0:
                    print('NORMALIZZAZIONE X', index)
                    self.model.X[index] = PRE.zscore_norm(x)

    def fit(self, alfa, iterations, _lambda = 0):

        print('J ALLA PRIMA ITERAZIONE = ', self.model.CostFunction())
        self.thetas = self.model.batchGD(alfa, iterations, _lambda)

    def print_solution(self):
        print('TETA DEL MODELLO: ', self.thetas)

    def prediction(self, input):

        predict = self.model.solution_zscore(input)
        predict = np.insert(predict, 0, 1)

        print('VALORE PREDETTO  = ', self.model.predict_M(predict))


class LogisticRegression:

    def __init__(self, csv, variables_indeces, classification_values):

        # ---------- ESTRAPOLA DATI ----------
        self.values = csv.giveme_values_indeces(variables_indeces, 0)
        self.filterY(classification_values[0])
        # ---------- MODELLO ----------
        self.classification_values = classification_values
        self.model = LG.LogisticRegression(self.values)  # REGRESSIONE
        self.thetas = None

    def filterY(self, target_value):

        filteredY = list()

        for y in self.values[0]:
            if y == target_value:
                filteredY.append(1)
            else:
                filteredY.append(0)

            self.values[0] = filteredY

    def Normalization(self):

        # CALCOLA Z SCORE NORM SULLE FEATURE

        for index, x in enumerate(self.model.X):
            if index != 0:
                #print('NORMALIZZAZIONE X', index)
                self.model.X[index] = PRE.zscore_norm(x)

    def fit(self, alfa, iterations, _lambda = 0):

        print('J ALLA PRIMA ITERAZIONE = ', self.model.CostFunction())
        self.thetas = self.model.batchGD(alfa, iterations, _lambda)


    def print_solution(self):
        print('TETA DEL MODELLO: ', self.thetas)

    def prediction(self, input, target):

        predict = self.model.solution_zscore(input)
        predict = np.insert(predict, 0, 1)

        print('PROBABILITÀ = ', self.model.LogisticFunction(predict))
        print('PREDETTO COME: ', end='')
        if self.model.LogisticFunction(predict) > 0.5:
            print(target)
        else:
            print('NON', target)