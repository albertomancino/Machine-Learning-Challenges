import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout
import pickle
from sklearn.model_selection import train_test_split



#dataset_path = "Datasets/candy.csv"
#dataset_path = "Datasets/mnist.csv"
#dataset_path = "Datasets/mnist2.csv"
dataset_path = "Datasets/mnist_new.csv"

dataset_file_path = "Pickle/dataset"

print('importo dataset')

# IMPORT DATASET CON PICKLE
file = open(dataset_file_path,'rb')
dataset = pickle.load(file)
file.close()

print('dataset importato con dimensione', dataset.dataset.shape)

# TAKING FEATURES AND TARGETS
x_indeces = list(range(1,785))
features: pd.DataFrame = dataset.giveme_cols(x_indeces)
print('converto le features np.array')
features = np.array(features)

y_index = [0]
target = dataset.giveme_cols(y_index)
print('converto i target in np.array')
target = np.array(target)

features = features.astype('float32')
features /= 255

features = features.reshape(features.shape[0], 28, 28, 1)

seed = 1
np.random.seed(seed)

# SPLITTING TEST SET E TRAINING SET
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=seed)


#CREATE MODEL
input_shape = (28, 28, 1)

model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history = model.fit(x=features, y=target, epochs=10, validation_split=0.2)

print(history.history.keys())

#Saving model in a pickle
filepickle = open('model_trained_1e','wb')
pickle.dump(model,filepickle)
filepickle.close()
input('Pickle Saved!')


#Prediction
pred = model.predict(features[1].reshape(1,28,28,1))
print(pred.argmax(), pred)