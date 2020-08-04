import pandas as pd
from random import randint
from random import shuffle
import numpy as np
from numpy import linalg
from Plotting import __plot_clusters__
from math import *
from scipy.stats import multivariate_normal

class Gaussian:

    def __init__(self, mean, covariance, pi = None):

        self.mean = mean
        self.covariance = covariance
        self.pi = pi
        self.r = None

    def probability(self, x):

        x = np.matrix(x)
        n = self.covariance.shape[0]
        inverse_cov = linalg.pinv(self.covariance)
        exponent = - 0.5 * np.dot((np.dot((x - self.mean),inverse_cov)),(x - self.mean).transpose())
        exponent = exponent.item((0,0))
        #denominator = sqrt(linalg.det(self.covariance) * ((2*pi)**n) + 1e-8)
        #prob = (e**exponent) / denominator
        gau = multivariate_normal(mean=self.mean, cov=self.covariance)
        #print(prob, gau.pdf(x))
        return gau

class KMeans:

    def __init__(self, features, cluster_no):

        self.cluster_no = cluster_no
        self.features = np.array(features)
        self.rows = int(len(features))
        self.centroids = list()
        self.cluster_assignment = np.zeros((self.rows, self.cluster_no))

    def __set_random_centroids__(self):

        centroids = list()
        random_rows = list()
        for _ in range(self.cluster_no):
            random_row = randint(0, self.rows - 1)
            while random_row in random_rows:
                random_row = randint(0, self.rows - 1)
            random_rows.append(random_row)

        for row in random_rows:

            centroids.append(self.features[row])

        self.centroids = centroids

    def __reset_assignement__(self):

        self.cluster_assignment = np.zeros((self.rows, self.cluster_no))

    def __points_to_cluster_assignement__(self):

        self.__reset_assignement__()

        for row in range(len(self.features)):
            distances = list()
            feature = self.features[row]
            for centroid in self.centroids:
                distances.append(__two_points_distance__(feature, centroid))
            min_distance = min(distances)

            min_distance_centroid_index = distances.index(min_distance)
            self.cluster_assignment[row][min_distance_centroid_index] = 1


    def __clusters__(self):

        clusters = list()

        for _ in self.centroids:
            clusters.append(list())

        for row, feature in enumerate(self.features):
            for cluster in range(len(self.cluster_assignment[row])):
                if self.cluster_assignment[row][cluster] == 1:
                    clusters[cluster].append(feature)
                    break

        return clusters


    def __move_centroids__(self):

        clusters = self.__clusters__()
        new_centroids = list()

        for cluster in clusters:
            sum = 0
            for feature in cluster:
                sum += feature
            if (len(cluster) != 0):
                sum /= len(cluster)
            else:
                print('EMPTY CLUSTER!')
            new_centroids.append(sum)

        return new_centroids

    def __error__(self):
        clusters = self.__clusters__()
        error = 0
        # per ogni cluster
        for index in range(len(clusters)):
            # per ogni feature del cluster
            for feature in clusters[index]:
                error += __two_points_distance__(feature, self.centroids[index])
        return error

    def __squared_within_clusters__(self):
        clusters = self.__clusters__()
        error = 0
        # per ogni cluster
        for index in range(len(clusters)):
            # per ogni feature del cluster
            for feature in clusters[index]:
                error += __two_points_squared_distance__(feature, self.centroids[index])
        return error

    def __fit__(self, iterations):

        errors = list()
        solutions = list()

        for iteration in range(iterations):

            self.__set_random_centroids__()
            self.__points_to_cluster_assignement__()

            for _ in range(5):

                self.centroids = self.__move_centroids__()
                self.__points_to_cluster_assignement__()

            solutions.append(self.centroids)
            errors.append(self.__error__())

        min_error = min(errors)
        min_index = errors.index(min_error)
        solution = solutions[min_index]

        self.centroids = solution
        self.__points_to_cluster_assignement__()

        return [self.features, self.cluster_assignment, self.centroids]

class PAM(KMeans):

    def __init__(self, features, cluster_no):
        super().__init__(features, cluster_no)

    def __fit__(self, iterations):

        errors = list()
        solutions = list()

        for iteration in range(iterations):

            self.__set_random_centroids__()
            self.__points_to_cluster_assignement__()
            cost = self.__error__()
            new_cost = 0

            while new_cost < cost:
                self.__move_centroids__()
                cost = self.__error__()
                self.__move_centroids__()
                new_cost = self.__error__()

            solutions.append(self.centroids)
            errors.append(self.__error__())

        min_error = min(errors)
        min_index = errors.index(min_error)
        solution = solutions[min_index]

        self.centroids = solution
        self.__points_to_cluster_assignement__()

        return [self.features, self.cluster_assignment, self.centroids]


    def __move_centroids__(self):

        centroid_indeces = list()
        for c in self.centroids:
            for index in range(self.rows):
                if np.array_equal(c, self.features[index]):
                    centroid_indeces.append(index)


        for cluster, centroid in enumerate(self.centroids):

            #features = self.__retrieve_features_indeces_in_cluster__(cluster)
            actual_centroid  = centroid
            actual_cost = self.__error__()

            for feature_index in range(self.rows):
                if feature_index not in centroid_indeces:
                    #SWAP
                    self.centroids[cluster] = self.features[feature_index]
                    #ricalcolo dei cluster
                    self.__points_to_cluster_assignement__()
                    cost = self.__error__()

                    if cost > actual_cost:
                        self.centroids[cluster] = actual_centroid

                    else:
                        actual_centroid = self.features[feature_index]
                        actual_cost = cost

    def __move_centroids1__(self):

        for cluster, centroid in enumerate(self.centroids):

            features = self.__retrieve_features_indeces_in_cluster__(cluster)
            actual_centroid  = centroid
            actual_cost = self.__error__()

            for feature_index in features:
                #SWAP
                self.centroids[cluster] = self.features[feature_index]
                #ricalcolo dei cluster
                self.__points_to_cluster_assignement__()
                cost = self.__error__()

                if cost > actual_cost:
                    self.centroids[cluster] = actual_centroid

                else:
                    actual_centroid = self.features[feature_index]
                    actual_cost = cost


    def __retrieve_features_indeces_in_cluster__(self, cluster):

        features_in_cluster = list()
        for index, assignement in enumerate(self.cluster_assignment):
            if assignement[cluster] == 1:
                features_in_cluster.append(index)
        return features_in_cluster

class GaussianMixture:

    def __init__(self, features, cluster_no):

        self.cluster_no = cluster_no
        self.features = np.array(features)
        self.rows = int(len(features))
        self.cluster_assignment = np.zeros((self.rows, self.cluster_no))
        self.gaussians = list()
        #self.__set_uniform_clusters__()
        self.__set_uniform_random_clusters__()
        self.__set_gaussian_parameters_on_cluster__()

    def __set_uniform_clusters__(self):

        element_in_cluster = round((self.rows / self.cluster_no))
        flag = 0
        cluster = 0
        for row in range(len(self.features)):
            if flag == element_in_cluster and cluster < self.cluster_no - 1:
                flag = 0
                cluster += 1
            self.cluster_assignment[row][cluster] = 1
            flag += 1

    def __set_uniform_random_clusters__(self):

        m = len(self.features)
        feature_indeces = [*range(0, m, 1)]
        # random shuffle of features
        shuffle(feature_indeces)

        element_in_cluster = round((self.rows / self.cluster_no))
        flag = 0
        cluster = 0
        for row in feature_indeces:
            if flag == element_in_cluster and cluster < self.cluster_no - 1:
                flag = 0
                cluster += 1
            self.cluster_assignment[row][cluster] = 1
            flag += 1



    def __retrieve_features_indeces_in_cluster__(self, cluster):

        features_in_cluster = list()
        for index, assignement in enumerate(self.cluster_assignment):
            if assignement[cluster] == 1:
                features_in_cluster.append(index)
        return features_in_cluster

    def __retrieve_features_values_in_cluster__(self, cluster):

        features_indeces = self.__retrieve_features_indeces_in_cluster__(cluster)
        features = self.features[np.array(features_indeces)]
        return features

    def __set_gaussian_parameters_on_cluster__(self):

        self.gaussians.clear()

        for cluster in range(self.cluster_no):

            features = self.__retrieve_features_values_in_cluster__(cluster)
            mean = self.__features_mean__(features)
            covariance = self.__features_covariance__(features, mean)
            # mixing coefficient inizializzata a 1 / numero di cluster dato che abbiamo diviso equamente il dataset
            self.gaussians.append(Gaussian(mean, covariance, 1/self.cluster_no))


    def __features_mean__(self, features):

        return features.mean(axis=0)

    def __features_covariance__(self, features, mean):

        variance = features - mean
        covariance = None
        for var in variance:
            var = np.matrix((var))
            cov = np.dot(var.transpose(), var)
            if covariance is not None:
                covariance += cov
            else:
                covariance = cov

        return (covariance / (len(features) - 1))


    def __expectation__(self):

        responsabilities = list()

        for feature in self.features:

            feature_responsabilities = list()
            responsability = None

            for gaussian in self.gaussians:

                responsability = gaussian.probability(feature) * gaussian.pi
                #responsability = gaussian.probability(feature) * gaussian.pi + 1e-8 # ho aggiunto una correzione
                feature_responsabilities.append(responsability)

            #print(feature_responsabilities)
            responsabilities_sum = np.sum(feature_responsabilities)
            responsabilities.append(np.divide(feature_responsabilities, responsabilities_sum))

        return responsabilities

    def __maximization__(self, responsabilites):

        responsabilites = np.matrix(responsabilites)

        for index, gaussian in enumerate(self.gaussians):

            cluster_respoabilities = responsabilites[:, index]
            gaussian.pi = cluster_respoabilities.mean(axis=0).item((0,0))
            gaussian.mean = self.__maximization_cluster_mean__(cluster_respoabilities)
            gaussian.covariance = self.__maximization_cluester_covariance__(cluster_respoabilities, gaussian.mean)


    def __maximization_cluster_mean__(self, responsabilities):

        accumulator = None
        for index in range(len(responsabilities)):

            contribution = self.features[index] * responsabilities[index].item((0,0))
            if index == 0:
                accumulator = contribution
            else:
                accumulator += contribution
        return accumulator / len(responsabilities)

    def __maximization_cluester_covariance__(self, responsabilities, mean):

        covariance = None

        for index in range(len(responsabilities)):

            variance = self.features[index] - mean
            variance = np.matrix(variance)
            var = np.dot(variance.transpose(), variance)
            var *= responsabilities[index].item((0,0))
            if index == 0:
                covariance = var
            else:
                covariance += var

        return covariance / responsabilities.sum()

    def __fit__(self, iterations):

        responsabilities = None

        for _ in range(iterations):
            responsabilities = self.__expectation__()
            self.__maximization__(responsabilities)

        responsabilities = self.__expectation__()
        self.__assign_feature_to_cluster__(responsabilities)

        return [self.features, self.cluster_assignment]

    def __clear_assignement__(self):

        self.cluster_assignment = np.zeros((self.rows, self.cluster_no))

    def __assign_feature_to_cluster__(self, responsabilities):

        self.__clear_assignement__()

        for index in range(len(responsabilities)):
            max = responsabilities[index].max()
            max_index = np.where(responsabilities[index] == max)[0][0]
            #print(responsabilities[index], max,max_index)
            self.cluster_assignment[index][max_index] = 1

class Cluster:

    def __init__(self, features, centroid = None):

        self.features = features
        self.centroid = centroid

class HierarchicalClustering:

    def __init__(self, features, cluster_no):

        self.cluster_no = cluster_no
        self.features = np.array(features)
        self.rows = int(len(features))
        self.centroids = list()
        self.cluster_assignment = np.zeros((self.rows, cluster_no))
        self.clusters = list()
        self.__set_initial_condition_clusters__()


    def __set_initial_condition_clusters__(self):

        for index in range(len(self.features)):

            #self.cluster_assignment[index][index] = 1
            self.clusters.append(Cluster([index], self.features[index]))

    def __fit__(self):

        print(len(self.clusters))

        while(len(self.clusters)>self.cluster_no):
            distance = self.__find_min_distance__()
            clusters = distance[1]
            self.__cluster_union__(clusters)

        self.__assign_features_to_cluster__()

        return [self.features, self.cluster_assignment, self.centroids]


    def __find_min_distance__(self):

        min_distance = None
        clusters = [None, None]

        for index1 in range(len(self.clusters)):
            for index2 in range(index1+1,len(self.clusters)):
                ward_distance = self.__compute_Ward_distances__(index1,index2)
                if min_distance is None:
                    min_distance = ward_distance
                    clusters[0] = index1
                    clusters[1] = index2
                else:
                    if ward_distance < min_distance:
                        clusters[0] = index1
                        clusters[1] = index2
                        min_distance = ward_distance

        return [min_distance, clusters]

    def __compute_Ward_distances__(self, cluster1, cluster2):

        features = list()
        features1 = self.clusters[cluster1].features
        features2 = self.clusters[cluster2].features
        features = features1 + features2
        new_centroid = self.__compute_centroid__(features)
        distance = self.__distance_from_centroid__(features, new_centroid)

        return distance

    def __compute_centroid__(self, indeces):

        features = list()
        for index in indeces:
            features.append(self.features[index])
        features = np.array(features)
        centroid = features.mean(axis=0)

        return centroid

    def __distance_from_centroid__(self, indeces, centroid):

        total_distance = 0
        for feature in indeces:
            total_distance += __two_points_distance__(centroid, self.features[feature])

        return total_distance

    def __cluster_union__(self, clusters):

        features = list()
        # ordino in maniera discendente per non avere conflitto tra gli indici all'interno della lista una volta
        # tolto il primo elemento
        clusters.sort(reverse=True)

        for cluster in clusters:
            features = features + self.clusters[cluster].features

        for cluster in clusters:
            self.clusters.remove(self.clusters[cluster])

        new_centroid = self.__compute_centroid__(features)
        self.clusters.append(Cluster(features, new_centroid))

    def __assign_features_to_cluster__(self):


        for index in range(len(self.clusters)):

            for feature in self.clusters[index].features:

                self.cluster_assignment[feature][index] = 1

            self.centroids.append(self.clusters[index].centroid)

class CorePoint:

    def __init__(self):

        self.neighborhood = list()
        self.core_points = list()

class DBSCAN:

    def __init__(self, features, epsilon, min_points):

        self.epsilon = epsilon
        self.min_points = min_points
        self.features = np.array(features)
        self.rows = len(self.features)
        self.clusters = list()
        self.cluster_assignment = None
        self.core_points = list()
        self.outliers = list()


    def __fit__(self):


        self.__find_core_points__()
        print('core points', self.core_points)
        self.__merge_clusters__()
        self.__assign_points_to_cluster__()

        return [self.features, self.cluster_assignment]


    def __find_core_points__(self):

        for f1 in range(self.rows):
            neighborhood = list()
            feature1 = self.features[f1]
            for f2 in range(self.rows):
                feature2 = self.features[f2]
                if f1 != f2 and __two_points_distance__(feature1, feature2) < self.epsilon:
                    neighborhood.append(f2)
            if len(neighborhood) >= self.min_points:
                self.core_points.append(f1)
                new_Cluster = CorePoint()
                new_Cluster.core_points.append(f1)
                self.clusters.append(new_Cluster)


    def __merge_clusters__(self):

        to_merge = list()

        # nella lista to_merge inserisco le coppie di cluster (formati da un singolo core point in questa fase)
        # che devono essere uniti tra loro perchè a distanza inferiore della minima imposta
        # le coppie vengono contate in un solo senso, da indice più basso ad indice più alto

        for c1 in range(len(self.core_points)):

            # indice delle feature core point
            cp1 = self.core_points[c1]
            # comincio il controllo dall'indide del cluster1 per evitare di contare lo stesso merging 2 volte

            for c2 in range(c1 + 1, len(self.core_points)):
                # indice delle feature core point
                cp2 = self.core_points[c2]
                # condizione di merging dei cluster
                if __two_points_distance__(self.features[cp1], self.features[cp2]) < self.epsilon:
                    to_merge.append((c1, c2))

        # in to_merge ci sono gli indici dei corepoints all'interno della lista self.core_points
        # che vanno uniti tra loro
        # questi indici coincidono con le posizioni dei relativi cluster nella lista self.cluster
        # nota bene che i cluster sono delle classi CorePoint


        merged = {}

        # ricavo le effettive coppie di cluster che devo andare ad unire.
        # ricavo cioè un cluster comune cui accorpare i cluster vicini per evitare accorpamenti a catena
        # Nota: le coppie vanno interpretate così: (a,b) indica che il cluster b deve essere accorpato con a

        for c1, c2 in to_merge:

            #voglio accorpare c2 in c1
            destination = c1
            # questo if controlla che il corepoint c1 non sia già stato accorpato ad un cluster in precedenza
            if c1 in merged:
                # se c1 è stato già accorpato il cluster in cui dovrò accorpare
                destination = merged[c1]
            #  questo if controlla che il corepoint c2 non sia già stato accorpato ad un cluster in precedenza
            if c2 in merged:
                # se c2 è stato già accorpato il cluster in cui dovrò lasciarlo nel cluster cui era
                # stato già accorpato ma dovrò spostare tutti i corepoints a lui collegati nel cluster cui appartiene
                # questo è il caso in cui un cp fa da ponte tra due insiemi di corepoints
                destination = merged[c2]

                for c in merged:
                    if merged[c] == c1:
                        merged[c] = destination
                merged[c1] = destination
            merged[c2] = destination
        # cluster da rimuovere una volta effettuato il merging
        to_remove = list()


        for to_merge in merged:

            merge_in = merged[to_merge]
            # array di core points della classe CorePoint in cui andrò ad accorpare i corepoints
            merge_in_core_points = self.clusters[merge_in].core_points
            # array di core points della classe CorePoint che andrò ad inserire in merge_in_core_points
            to_merge_core_point = self.core_points[to_merge]
            self.clusters[merge_in].core_points.append(to_merge_core_point)
            to_remove.append(to_merge)

        # per evitare il cambiamento degli indici
        to_remove.sort(reverse=True)

        # rimuovo i cluster che sono stati accorpati con altri
        for cluster in to_remove:
            del self.clusters[cluster]


    def __assign_points_to_cluster__(self):

        assigned = list()

        for row in range(len(self.features)):

            min_distance = None
            best_cluster = None

            for cluster in range(len(self.clusters)):

                actual_cluster = self.clusters[cluster]

                for cp in actual_cluster.core_points:
                    if row not in self.core_points:
                        distance = __two_points_distance__(self.features[row], self.features[cp])

                        if (min_distance is None) or (distance < min_distance):
                            min_distance = distance
                            best_cluster = cluster
            if best_cluster is not None:
                self.clusters[best_cluster].neighborhood.append(row)
                assigned.append(row)


        for row in range(len(self.features)):

            min_distance = None
            best_cluster = None

            if row not in assigned:

                for cluster in range(len(self.clusters)):

                    actual_cluster = self.clusters[cluster]

                    for point in actual_cluster.neighborhood:

                        if row not in self.core_points:
                            distance = __two_points_distance__(self.features[row], self.features[point])

                            if (min_distance is None) or (distance < min_distance):
                                min_distance = distance
                                best_cluster = cluster

                if best_cluster is not None:
                    self.clusters[best_cluster].neighborhood.append(row)
                    assigned.append(row)



        self.cluster_assignment = np.zeros((self.rows, len(self.clusters)))

        for c in range(len(self.clusters)):
            actual_cluster = self.clusters[c]

            for cp in actual_cluster.core_points:

                self.cluster_assignment[cp][c] = 2

            for point in actual_cluster.neighborhood:

                self.cluster_assignment[point][c] = 1





def __two_points_distance__(a, b):
    return np.sqrt(np.sum(np.square(a - b)))

def __two_points_squared_distance__(a, b):
    return np.sum(np.square(a - b))

def __elbow_method__(features, max_cluster_no):

    elbow = []
    cluster_no = []
    iterarations = 10

    for clusters in range(1, max_cluster_no):
        model = KMeans(features, clusters)
        model.__fit__(10)
        elbow.append(model.__squared_within_clusters__())
        cluster_no.append(clusters)
    return [elbow, cluster_no]


# DUBBI
# quando scorro su un numpy array di dimensione n,2 mi restituisce righe n, e non n,1. In questo modo non posso
# cacolare la traposta ma devo ricorrere a np.matrix