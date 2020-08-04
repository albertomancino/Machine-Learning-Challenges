import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import pickle
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant

print('importo modello')

dataset_file_path = "Model_Trained/v2w_trained_1e"

# IMPORT DATASET CON PICKLE
file = open(dataset_file_path,'rb')
model = pickle.load(file)
file.close()

'''Preparo i dati che voglio predire'''
test_sample1 = "alberto"
test_sample2 = "ciao mi chiamo"
test_sample3 = "questo e il mio dataset"
test_sample4 = "Alberto"
test_sample5 = "Ciao mi chiamo"
test_sample6 = "questo e il mio dataset"
test_sample7 = "fundies should find Josh McDowell"



#test_samples = [test_sample1, test_sample2, test_sample3, test_sample4, test_sample5, test_sample6, test_sample7]
test_samples = [test_sample7]

tokenizer = Tokenizer()
#test_samples_token = tokenizer_obj.texts_to_sequences(test_samples)
test_samples_token = tokenizer.texts_to_sequences(test_samples)
test_samples_token_pad = pad_sequences(test_samples_token, maxlen=1000)

predict = model.predict(x=test_samples_token_pad)

print(predict)






