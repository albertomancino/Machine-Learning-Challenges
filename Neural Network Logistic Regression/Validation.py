from math import ceil
from Regression import Regression
import Preprocessing
import Input
import pandas as pd



def _Holdout_validation(training_set, validation_set, X_indeces, Y_indeces, alfa, iterations, _lambda):

    trainingX = Input.giveme_cols(training_set, X_indeces)
    trainingY = Input.giveme_cols(training_set, Y_indeces)

    validationX = Input.giveme_cols(validation_set, X_indeces)
    validationY = Input.giveme_cols(validation_set, Y_indeces)

    training_model = Regression(trainingX,trainingY)
    validation_model = Regression(validationX,validationY)

    validation_model.thetas = training_model.batch_gradient_descent(alfa,_lambda,iterations)

    return [(training_model.cost_function(_lambda),validation_model.cost_function(_lambda)), training_model.thetas]




def _KFolds_Splitting(dataset, X_indeces, Y_indeces, k):

    folder_dim = ceil(len(dataset)/k)
    folder_boundaries = list()
    start = 0

    # calcolo le righe di inizio e fine di ogni validation folder

    while start < len(dataset):
        end = start + folder_dim

        if end > len(dataset):
            end = len(dataset)

        folder_boundaries.append((start,end))
        start = start + folder_dim

    folders = list()

    datasets = list()

    for bound in folder_boundaries:
        folders.append(dataset.iloc[bound[0]:bound[1]])

    # dal dataset rimuovo le righe della validation folder, la parte restante comporrà il training set
    for folder in folders:
        newdataset = dataset.copy()
        indeces = folder.index
        newdataset = newdataset.drop(indeces)
        datasets.append((newdataset,folder))

    return datasets



def learning_curves_values(dataset, Xindeces, Yindex):

    trainingJ = list()
    validationJ = list()

    split = Preprocessing.split_sets(dataset,0.2)
    trainingset = split[0]
    fixed_validationset = split[1]

    print(trainingset)

    for index in range(15,len(dataset)):

        newtraining = trainingset[:index]
        result = _Holdout_validation(newtraining,fixed_validationset,Xindeces,Yindex,0.1,1000,0)

        trainingJ.append((index, result[0][0]))
        validationJ.append((index, result[0][1]))






