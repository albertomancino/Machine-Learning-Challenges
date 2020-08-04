import pandas as pd


class Dataset:

    def __init__(self, file_path, means = 0, standard_dev = 0):
        try:
            self.dataset = pd.read_csv(file_path, sep=',', quotechar='"', encoding='utf8', header=None)
        except:
            print('ACCESSO AL FILE NON RIUSCITO')
            self.dataset = None
        self.file_path = file_path
        self.means = means
        self.standard_dev = standard_dev

    def info(self):
        print(self.dataset.columns)

def giveme_cols(dataset ,index):
    return dataset[index]

def multiply_cols(dataset : pd.DataFrame, indeces):    # given a list of indeces returns a new feature added to the dataset that is the multiplication of the column indexed in the list of indeces

    result = dataset[indeces[0]] * dataset[indeces[1]]

    print(dataset[indeces[0]])
    print(dataset[indeces[1]])

    for index in indeces[2:]:
        result = result * dataset[index]
        print(dataset[index])

    print(result)
    print(type(dataset))
    last = dataset.columns[-1]
    dataset[last+1]=result
