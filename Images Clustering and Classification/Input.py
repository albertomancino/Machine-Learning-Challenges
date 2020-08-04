import pandas as pd
import numpy as np

class Dataset:

    def __init__(self, file_path):
        try:
            self.dataset = pd.read_csv(file_path, sep=',', quotechar='"', encoding='utf8', header=None, skiprows=[0])
        except:
            print('ACCESSO AL FILE NON RIUSCITO')
            self.dataset = None

        self.file_path = file_path

    def info(self):
        print(self.dataset.columns)

    def giveme_cols(self, index):
        return self.dataset[index]


def giveme_cols(dataset, index):
    return dataset[index]


def multiply_cols(dataset: pd.DataFrame, indeces):
    # given a list of indeces returns a new feature added to the dataset that is the multiplication of the column
    # indexed in the list of indeces

    result = dataset[indeces[0]] * dataset[indeces[1]]

    for index in indeces[2:]:
        result = result * dataset[index]

    last = dataset.columns[-1]
    dataset[last+1] = result


def from_multiples_columns_to_one(X: np.array):

    new_X = list()

    for row in range(len(X)):
        index = 0
        actual_row = X[row]
        found = False
        for element in actual_row:
            if element.item() > 1 and found is False:
                new_X.append(index)
                found = True
            index += 1
        if found is False:
            new_X.append(-1)

    return np.array(new_X)