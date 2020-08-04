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
from sklearn.metrics import classification_report, confusion_matrix
import Input
import seaborn as sns


#dataset_path = "Datasets/mnist_new.csv"

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


history = model.fit(x=features, y=target, epochs=1, validation_split=0.2)

print(history.history.keys())

#Saving model in a pickle
filepickle = open('model_trained_1e','wb')
pickle.dump(model,filepickle)
filepickle.close()
print('Pickle Saved!')


#Prediction
Y_pred = model.predict(features[1].reshape(1,28,28,1))
print(Y_pred.argmax(), Y_pred)

Y_pred_cm = model.predict_classes(X_test) # for the confusion matrix

#Confution Matrix and Classification Report
print('Confusion Matrix')
#print(confusion_matrix(y_test.argmax(axis=1), Y_pred.argmax(axis=1)))


''''''
test_lables = ["0","1","2","3","4","5","6","7","8","9"]
classes = [0,1,2,3,4,5,6,7,8,9]

con_mat = tf.math.confusion_matrix(labels=y_test, predictions=Y_pred_cm).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm,index = classes,columns = classes)

figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Greens)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()