import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn import preprocessing


from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout
from sklearn.metrics import mean_squared_log_error

train_dataset_path = "Texts/train.tsv"
test_dataset_path = "Texts/test.tsv"


training  = pd.read_csv(train_dataset_path, sep='\t', quotechar='"', encoding='utf8', skiprows=1, header=None, low_memory=False)
test = pd.read_csv(test_dataset_path, sep='\t', quotechar='"', encoding='utf8', skiprows=1, header=None, low_memory=False)

features_indeces = [1,2,3,4,6]
target_index = [8]

training_features = training[features_indeces]
training_target = training[target_index]

test_features = test[features_indeces]
test_target = test[target_index]

training_features = np.array(training_features)
training_target = np.array(training_target)

test_features = np.array(test_features)
test_target = np.array(test_target)

# lower case della feature 0,1,4
for t in training_features:
    t[0] = t[0].lower()
    t[1] = t[1].lower()
    t[4] = t[4].lower()

# lower case della feature 0,1,4
for t in test_features:
    t[0] = t[0].lower()
    t[1] = t[1].lower()
    t[4] = t[4].lower()

# removing "(books)" from col: 4
for t in training_features:
    t[4] = t[4].replace('(books)','')

# removing "(books)" from col: 4
for t in test_features:
    t[4] = t[4].replace('(books)','')

# removing "customer reviews" or "customer review" from col: 3
for t in training_features:
    t[3] = t[3].replace('customer reviews', '')
    t[3] = t[3].replace('customer review', '')
    t[3] = t[3].replace(',', '.')

    t[3] = float(t[3])


# removing "customer reviews" or "customer review" from col: 3
for t in test_features:
    t[3] = t[3].replace('customer reviews', '')
    t[3] = t[3].replace('customer review', '')
    t[3] = t[3].replace(',', '.')

    t[3] = float(t[3])

# removing "out of 5 stars" from col: 2
for t in training_features:
    t[2] = t[2].replace('out of 5 stars', '')
    t[2] = float(t[2])

# removing "out of 5 stars" from col: 2
for t in test_features:
    t[2] = t[2].replace('out of 5 stars', '')
    t[2] = float(t[2])


# Extracting years value from col: 1
year_training = list()
for t in training_features:
    year_training.append(t[1][-4:])

# Extracting years value from col: 1
year_test = list()
for t in test_features:
    year_test.append(t[1][-4:])

# Cutting col:1 for getting value "hardcover" or "papercover"
for t in training_features:
    t[1] = t[1][0:9]

# Cutting col:1 for getting value "hardcover" or "papercover"
for t in test_features:
    t[1] = t[1][0:9]

# Adding year column

year_training_np = np.array(year_training)
training_features = np.column_stack((training_features, year_training_np))

year_test_np = np.array(year_test)
test_features = np.column_stack((test_features, year_test_np))

print(training_features[0])
print(test_features[0])

authors = list()
edition = list()
gen = list()
years = list()


for t in training_features:
    print(t[4])


for row in training_features:

    authors.append(row[0])
    edition.append(row[1])
    gen.append(row[4])
    years.append(row[5])


for row in test_features:

    authors.append(row[0])
    edition.append(row[1])
    gen.append(row[4])
    years.append(row[5])


# preparo label encoding
authors_label = preprocessing.LabelEncoder()
# preparo il label encoding
x = authors_label.fit_transform(authors)

# preparo label encoding
edition_label = preprocessing.LabelEncoder()
# preparo il label encoding
edition_label.fit(edition)

# preparo label encoding
gen_label = preprocessing.LabelEncoder()
# preparo il label encoding
gen_label.fit(gen)

# preparo label encoding
year_label = preprocessing.LabelEncoder()
# preparo il label encoding
year_label.fit(years)


training_features[:,0] = authors_label.transform(training_features[:,0])
training_features[:,1] = edition_label.transform(training_features[:,1])
training_features[:,4] = gen_label.transform(training_features[:,4])
training_features[:,5] = year_label.transform(training_features[:,5])


# PREPROCESSING




scaled_training_features = preprocessing.scale(training_features, with_mean=True, with_std=True)

for t in scaled_training_features.transpose():
    print(t[0])
    mean_0 = np.mean(t[0])
    mean_1 = np.mean(t[1])
    mean_2 = np.mean(t[2])
    mean_3 = np.mean(t[3])
    mean_4 = np.mean(t[4])

    stddev_0 = np.std(t[0])
    stddev_1 = np.std(t[1])
    stddev_2 = np.std(t[2])
    stddev_3 = np.std(t[3])
    stddev_4 = np.std(t[4])


print('media',mean_0)
print('std',stddev_0)

print('media',mean_1)
print('std',stddev_1)

print('media',mean_2)
print('std',stddev_2)

print('media',mean_3)
print('std',stddev_3)

print('media',mean_4)
print('std',stddev_4)

input()

model = Sequential()

print('run ')
model.add(Dense(20, activation='relu',input_dim=6))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation="linear"))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

model.fit(batch_size=32, x=scaled_training_features, y=training_target, epochs=100, validation_split=0.2)

predictions = model.predict(training_features)

error = np.sqrt(mean_squared_log_error(training_target, predictions))

print('RMSLE ',error)