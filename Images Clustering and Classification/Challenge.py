from Input import Dataset, from_multiples_columns_to_one
import Input
import Preprocessing
import numpy as np
from DimensionalityReduction import PCA
from Clustering import KMeans, PAM, HierarchicalClustering, GaussianMixture, DBSCAN
import PerformanceMeasurement as PM
from sklearn import mixture
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import k_means_
import hdbscan
from sklearn import svm


dataset_path = 'Datasets/mnist 2.csv'

#data = Dataset(dataset_path)

reduced_rows_data = Dataset(dataset_path)

reduced_rows_data.dataset = reduced_rows_data.dataset.iloc[0:2000,:]

feature_indeces = list(range(1,785))

features = reduced_rows_data.giveme_cols(feature_indeces)

features = np.array(features)

# NORMALIZZAZIONE DELLA SCALA DI COLORI
features = np.divide(features, 255)

target_index = [0]
target = reduced_rows_data.giveme_cols(target_index)

new_dimension = 100
pca = PCA(features)

'''
for x in range(100,700,10):
    print('retained covariance with new dimension =',x,'is',pca.__variance_retained__(x))

for x in range(140,150):
    print('retained covariance with new dimension =',x,'is',pca.__variance_retained__(x))
'''

clusters = list()

reducted_features = pca.__reduction__(pca.u, 145)

features = reducted_features

# selected dimension = 145 0.9497592469961726

cluster_no = 10
iterations = 5

'''
model = KMeans(features, cluster_no)
print('KMEANS')
solution = model.__fit__(iterations)

k_means_cluster = solution[1]
clusters.append(k_means_cluster)

k_means_clusters = from_multiples_columns_to_one(k_means_cluster)
'''




'''
model = PAM(features, cluster_no)
print('PAM')
solution = model.__fit__(1)


pam_cluster = solution[1]
clusters.append(pam_cluster)

pam_clusters = from_multiples_columns_to_one(pam_cluster)
'''

'''

model = HierarchicalClustering(features, cluster_no)
solution = model.__fit__()

hier_cluster = solution[1]
clusters.append(hier_cluster)
print(hier_cluster)
hier_clusters = from_multiples_columns_to_one(hier_cluster)
'''



#model = GaussianMixture(features, cluster_no)
#solution = model.__fit__(2)

#GMM_cluster = solution[1]
#clusters.append(GMM_cluster)
#GMM_clusters = from_multiples_columns_to_one(GMM_cluster)

#print(GMM_clusters)

seed = np.random.seed(1)

model = mixture.GaussianMixture(n_components=10)

model.fit(features)

prediction = model.predict(features)
prediction = np.array(prediction)

prediction_new = np.zeros((len(prediction),cluster_no))


for row in range(len(prediction_new)):

    pred = prediction[row]
    prediction_new[row][pred] = 1

clusters.append(prediction_new)


'''
print('DBSCAN')
model = DBSCAN(features, 10,10)
solution = model.__fit__()

DBSCAN_cluster = solution[1]
DBSCAN_clusters = from_multiples_columns_to_one(DBSCAN_cluster)

clusters.append(DBSCAN_cluster)
'''



affinity = 'euclidean'

epsilon = 20
min_samples = 10

min_cluster_size = 5

model = hdbscan.HDBSCAN(min_cluster_size)

prediction = model.fit_predict(features)
prediction = np.array(prediction)

print(prediction)


'''
# ELBOW

elbow_model = k_means_.KMeans(cluster_no)
visualizer = KElbowVisualizer(elbow_model, k=(5,12))
visualizer.fit(features)
visualizer.show()
'''
for c in clusters:

    s = PM.silhouette(features, c)
    s = np.array(s)
    print('Sil = ', s.mean())



'''
new_datasets = Preprocessing.split_sets(reduced_rows_data.dataset, 0.8)

model = svm.SVC(gamma='scale', C=5)

X_training = Input.giveme_cols(new_datasets[1], feature_indeces)
Y_training = Input.giveme_cols(new_datasets[1], target_index)

X_training = np.array(X_training)
Y_training = np.array(Y_training)

model.fit(X_training, Y_training)

X_validation = Input.giveme_cols(new_datasets[0], feature_indeces)
Y_validation = Input.giveme_cols(new_datasets[0], target_index)

prediction = model.predict(X_validation)


X_validation = np.array(X_validation)
Y_validation = np.array(Y_validation)

TP = 0
TN = 0

for y in range(len(Y_validation)):

    actual = Y_validation[y]
    pred = prediction[y]

    if actual == 5 and pred == 5:
        TP += 1

    if actual != 5 and pred != 5:
        TN += 1

print('Accuracy = ', (TP+TN)/len(Y_validation))

'''