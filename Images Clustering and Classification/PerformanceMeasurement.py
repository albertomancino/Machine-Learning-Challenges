import numpy as np


def silhouette(features, cluster_assignement):

    silhouette = list()
    cluster_no = len(cluster_assignement[0])

    for index, selected_feature in enumerate(features):

        distances = list()

        for cluster in range(len(cluster_assignement[0])):

            distances.append(list())

        for feature in range(len(features)):

            distance = two_points_distance(selected_feature, features[feature])
            feature_cluster = np.where(cluster_assignement[feature] == 1)[0][0]

            distances[feature_cluster].append(distance)

        selected_feature_cluster = np.where(cluster_assignement[index] == 1)[0][0]

        coh = np.array(distances[selected_feature_cluster]).mean()

        sep = None

        for other_cluster in range(cluster_no):

            if other_cluster != selected_feature_cluster and len(distances[other_cluster]) > 0:

                distance_from_cluster = np.array(distances[other_cluster]).mean()

                if sep == None:

                    sep = distance_from_cluster

                elif other_cluster != selected_feature_cluster:

                    if distance_from_cluster < sep:

                        sep = distance_from_cluster

        if coh <= sep:

            silhouette.append(1 - coh/sep)

        else:
            silhouette.append(sep/coh - 1)

    return silhouette

def silhouette_mean(silhouette):

    silhouette = np.array(silhouette)
    return silhouette.mean()


def two_points_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))

def silhouette2(features, cluster_assignement, cluster_no):

    silhouette = list()

    for index, selected_feature in enumerate(features):

        distances = list()

        for cluster in range(cluster_no):

            distances.append(list())

        for feature in range(len(features)):

            distance = two_points_distance(selected_feature, features[feature])

            feature_cluester = cluster_assignement[feature]

            distances[feature_cluester].append(distance)

        selected_feature_cluster = cluster_assignement[index]

        coh = np.array(distances[selected_feature_cluster]).mean()

        sep = None

        for other_cluster in range(cluster_no):

            if other_cluster != selected_feature_cluster and len(distances[other_cluster]) > 0:

                distance_from_cluster = np.array(distances[other_cluster]).mean()

                if sep == None:

                    sep = distance_from_cluster

                elif other_cluster != selected_feature_cluster:

                    if distance_from_cluster < sep:

                        sep = distance_from_cluster

        if coh <= sep:

            silhouette.append(1 - coh/sep)

        else:
            silhouette.append(sep/coh - 1)

    return silhouette
