import csv

class CSVManager:


    def __init__(self, file_name):

        self.file_name = file_name


        # ---------- RESTITUISCE UNA LISTA FORMATA DAL VETTORE Y E DALLA MATRICE X: OUTPUT = [Y, X] ----------
    def giveme_values_indeces(self, indeces, regression = 1):

        # variables_list è la lista dei nomi delle variabili. Il primo è la variabile target, i successivi le variabli feature
        features_index = indeces[1]
        target_index = indeces[0]
        polinomials_indeces = indeces[2]

        X = list()  # è la matrice X dei valori delle feature
        Y = list()  # è il vettore Y dei valori della feature
        Z = list()

        output = list()

        # ---------- SETTO LA MATRICE X ----------
        for _ in range(len(features_index) + 1):  # IL +1 SERVE A CONSIDERARE LA FEATURE FITTIZIA x0

            X.append([])

        # ---------- SETTO LA MATRICE Z ----------
        for _ in range(len(polinomials_indeces)):
            Z.append([])

        # ---------- PRELEVA DAL DATASET I VALORI DI Y E X ----------
        with open(self.file_name, 'r', encoding='utf-8-sig') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            print('FILE = ', self.file_name)
            #next(csv_reader)

            for line in csv_reader:
                value = line[target_index]

                if regression == 1:
                    value = float(line[target_index])
                    Y.append(value)
                else:
                    Y.append(value)

                for index in range(len(X)):

                    if index == 0:
                        X[0].append(1)
                    else:
                        value = float(line[features_index[index - 1]])
                        X[index].append(value)

                for index, poli in enumerate(polinomials_indeces):

                    value = 1

                    for factor in poli:
                        value *= float(line[factor])
                    Z[index].append(value)

        for z in Z:
            X.append(z)
        output.append(Y)
        output.append(X)
        return output

