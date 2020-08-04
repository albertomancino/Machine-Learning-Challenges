from Input import Dataset
import Input
import Preprocessing
from Validation import _KFolds_Splitting

from NeuralNetwork import NeuralNetwork
import pandas as pd


if __name__ == '__main__':

    # ----- DATASET -----

    file = "Dataset/wine.csv"
    data = Dataset(file)

    # ----- FEATURES -----

    X_indeces = [0,1,2,3,4,5,6,7,8,9,10]

    # ----- TARGET -----

    Y_indeces = [11]

    # ----- PREPROCESSING -----

    Preprocessing.show_statistics(data.dataset, X_indeces)

    # ----------------------------------------------------------------
    # RICHIESTA 1) NORMALIZZAZIONE Z-SCORE

    # ----- Z-SCORE  -----

    normalization = Preprocessing.dataset_z_score(data, X_indeces)
    means = normalization[1]
    std_devs = normalization[2]

    # ----- NORMALIZED DATASET -----

    norm_file = file.replace(".csv", "") + "_ZSCORED.csv"
    data = Dataset(norm_file)




    # ----------------------------------------------------------------
    # RICHIESTA 2) SUDDIVISIONE DEL DATASET IN K-FOLDS CON CROSS VALIDATION

    datasets = _KFolds_Splitting(data.dataset, X_indeces, Y_indeces, 5)

    trainingsets = list()
    validationsets = list()

    for index, dataset in enumerate(datasets):
        print('\n\n')
        print(index+1, ') TRAINING SET - VALIDATION SET ----\n')
        print('TRAINING SET')
        print(dataset[0], '\n')
        trainingsets.append(dataset[0])
        print('VALIDATION FOLDER')
        print(dataset[1], '\n')
        validationsets.append(dataset[1])


    # ----------------------------------------------------------------
    # RICHIESTA 3) TROVARE LA MIGLIOR STIMA PER LAMBDA, ALFA E NUMERO DI NEURONI

    _lambdas = [0.0, 0.0000001, 0.0000003, 0.0000005, 0.0000007]
    #_lambda = 0.0000001
    alfas = [0.01, 0.05, 0.07, 0.001]
    #alfa = 0.01
    #hidden_layers_neurons = [20,20]
    neurons = [[3,3],[5,5],[10,10],[20,20]]
    iterations = 100

    for _lambda in _lambdas:
        for alfa in alfas:
            for hidden_layers_neurons in neurons:

                total_RMSE = 0
                total_MSE = 0
                total_MAE = 0
                total_squaredR = 0

                for index in range(len(datasets)):

                    X_training = Input.giveme_cols(trainingsets[index], X_indeces)
                    Y_training = Input.giveme_cols(trainingsets[index], Y_indeces)

                    X_validation = Input.giveme_cols(validationsets[index], X_indeces)
                    Y_validation = Input.giveme_cols(validationsets[index], Y_indeces)

                    training_model = NeuralNetwork(X_training,Y_training,hidden_layers_neurons, alfa, _lambda)
                    training_model.fit(iterations)

                    validation_model = NeuralNetwork(X_validation,Y_validation,hidden_layers_neurons, alfa, _lambda)
                    validation_model.network = training_model.network
                    validation_model.bias_weights = training_model.bias_weights

                    total_RMSE += validation_model.RMSE()
                    total_MSE += validation_model.MSE()
                    total_MAE += validation_model.MAE()
                    total_squaredR += validation_model.squaredR()


                total_RMSE = total_RMSE / len(datasets)
                total_MSE = total_MSE / len(datasets)
                total_MAE = total_MAE / len(datasets)
                total_squaredR = total_squaredR / len(datasets)

                print('lambda', _lambda, 'alfa', alfa, 'nuerons', hidden_layers_neurons)
                print('RMSE ',total_RMSE)
                print('MSE ',total_MSE )
                print('MAE ',total_MAE)
                print('total_squaredR', total_squaredR.item((0,0)))
                print('\n\n')


    input('finito')


    # ----- SPLIT DATASET AND TEST SET -----

    # ----- DATA FROM DATASET  -----

    X = Input.giveme_cols(data.dataset, X_indeces)
    Y = Input.giveme_cols(data.dataset, Y_indeces)


    '''
    model = NeuralNetwork(X,Y,hidden_layers_neurons, alfa, _lambda)
    model.backward_propagation()
    model.print_cost()

    for it in range(1000):

        model.backward_propagation()
        
    model.print_cost()

    prediction = [6.0, 0.31, 0.47, 3.6, 0.067, 18.0, 42.0, 0.99549, 3.39, 0.66, 11.0]
    prediction = [7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4]
    model.predict(prediction,means,std_devs)
    '''