import CSVManager as CSVM
import Models
import csv
import Preprocessing as PRE
import numpy as np

print('\n----- ESEGUO RICHIESTA (1) -----')
# ---------- PRENDI IN INPUT IL TRAINING SET ----------
file1 = 'Dataset/candy.csv'
csv_file = CSVM.CSVManager(file1)

# ---------- INDICA L'INDICE DEL TARGET ----------
target_index = 10
# ---------- INDICA GLI INDICI DELLE FEATURE ----------
features_indeces = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12]

# ---------- INDICA LE FEATURE MULTICLASSE----------
'''
NOTA: vanno inserite ciascuna come lista degli indici che le compongono.
Tutte queste liste vanno inseite all'interno della polinomials_indeces
Ad esempio: [ [ 1, 3, 4] , [2, 2] , [3 , 3, 1 ] ]
'''
polinomials_indeces = []

# ---------- INPUT STATICO NOTI GLI INDICI----------
variables_indeces = [target_index, features_indeces, polinomials_indeces]
classification_values = ['1']  # VALORE DA CLASSIFICARE

# ---------- PRINT NORMALIZED CSV RICHIESTA (1)----------
# LEGGO DATI DAL DATASET
values = csv_file.giveme_values_indeces(variables_indeces)
zScored = list()
minmax = list()

# LEGGO LE FEATURE E LE NORMALIZZO
for index, feature in enumerate(values[1][1:11:]):
    zScored.append(PRE.zscore_norm(feature))

for index, feature in enumerate(values[1][1:11:]):
    minmax.append(PRE.min_max_norm(feature, 0, 5))

# AGGIUNGO LE ULTIME TRE COLONNE NON NORMALIZZATE
zScored.append(values[1][11])
zScored.append(values[1][12])
zScored.append(values[1][13])

minmax.append(values[1][11])
minmax.append(values[1][12])
minmax.append(values[1][13])

zScored = np.transpose(zScored)
minmax = np.transpose(minmax)

#min max (1.1)
with open('Dataset/Zscored.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"')
    csv_writer.writerows(zScored)
#Zscore (1.2)
with open('Dataset/Minmax.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"')
    csv_writer.writerows(minmax)

# ---------- MODELLO ----------
print('\n----- ESEGUO RICHIESTA (2) (3) -----')

csv_file = CSVM.CSVManager(file1)
features_indeces = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
variables_indeces = [target_index, features_indeces, polinomials_indeces]

print('INDICI: \nTARGET = ', target_index, '\nFEATURE = ', features_indeces, '\nVARIABILI POLINOMIALI: ',polinomials_indeces)

model = Models.LogisticRegression(csv_file, variables_indeces, classification_values)

# ---------- NORMALIZZAZIONE ----------
model.Normalization()

# ---------- FITTING ----------
alfa = 0.25
iterations = 1000
_lambda = 0

model.fit(alfa, iterations, _lambda)
model.print_solution()

# ---------- VALORE DA PREDIRE ----------
predict = [0, 0, 0, 1, 0, 0, 1, 0.87199998, 0.84799999, 49.524113]
# ---------- PREDIZIONE RICHIESTA (3) ----------
model.prediction(predict, 'CIOCCOLATO')

# ---------- PREDIZIONE RICHIESTA (4) (5) ----------
print('\n----- ESEGUO RICHIESTA (4) (5) -----')
#Non avendo previsto nella nostra libreria una classificazione su più y,
#viene rilanciato 3 volte il modello cambiando la target var
features_indeces = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
predict = [1, 0, 0, 0, 1, 0, 0, 0.186, 0.26699999, 41.904308]

print('PROBABILITà CHE SIA UN CIOCCOLATO')
target_index = 10
variables_indeces = [target_index, features_indeces, polinomials_indeces]
classification_values = ['1']  # VALORE DA CLASSIFICARE
print('INDICI: \nTARGET = ', target_index, '\nFEATURE = ', features_indeces, '\nVARIABILI POLINOMIALI: ',polinomials_indeces)
model = Models.LogisticRegression(csv_file, variables_indeces, classification_values)
model.Normalization()
model.fit(alfa, iterations, _lambda)
model.print_solution()
model.prediction(predict, 'CIOCCOLATO')

print('\nPROBABILITà CHE SIA UNA FRUTTA')
target_index = 11
variables_indeces = [target_index, features_indeces, polinomials_indeces]
classification_values = ['1']  # VALORE DA CLASSIFICARE
print('INDICI: \nTARGET = ', target_index, '\nFEATURE = ', features_indeces, '\nVARIABILI POLINOMIALI: ',polinomials_indeces)
model = Models.LogisticRegression(csv_file, variables_indeces, classification_values)
model.Normalization()
model.fit(alfa, iterations, _lambda)
model.print_solution()
model.prediction(predict, 'FRUTTA')

print('\nPROBABILITà CHE SIA UN DOLCETTO')
target_index = 12
variables_indeces = [target_index, features_indeces, polinomials_indeces]
classification_values = ['1']  # VALORE DA CLASSIFICARE
print('INDICI: \nTARGET = ', target_index, '\nFEATURE = ', features_indeces, '\nVARIABILI POLINOMIALI: ',polinomials_indeces)
model = Models.LogisticRegression(csv_file, variables_indeces, classification_values)
model.Normalization()
model.fit(alfa, iterations, _lambda)
model.print_solution()
model.prediction(predict, 'DOLCETTO')
