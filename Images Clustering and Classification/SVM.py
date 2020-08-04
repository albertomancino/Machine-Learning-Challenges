from sklearn import svm
from Input import Dataset, from_multiples_columns_to_one
import numpy as np

# https://scikit-learn.org/stable/modules/svm.html

dataset_path = 'Datasets/candy_ZSCORED.csv'
#dataset_path = 'Datasets/candy.csv'

data = Dataset(dataset_path)

feature_indeces = [0,1,2,3,4,5,6,7,8,9]
#feature_indeces = [7, 8]

target_indeces = [10,11,12]

features = data.giveme_cols(feature_indeces)
features = np.array(features)

target = data.giveme_cols(target_indeces)
target = np.array(target)
target = from_multiples_columns_to_one(target)

model = svm.SVC(gamma='scale')
model.fit(features,target)

solution = model.predict(features)

print(target)
print(solution)