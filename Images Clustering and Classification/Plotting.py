import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def __two_dimension_features__(dataframe, x_index, y_index):

    dataframe.plot(kind = 'scatter', x = x_index, y = y_index, color = 'red')

    plt.show()

def __multiple_dataset__(clusters: list, x_indeces: list, y_indeces: list, colors: list, centroids: list):

    axis = plt.gca()

    for index in range(len(clusters)):

        clusters[index].plot(kind='scatter', x=x_indeces[index], y=y_indeces[index], color=colors[index], ax = axis)

        centroids[index].plot(kind='scatter', marker='x', x=0, y=1, color=colors[index], ax = axis)

    plt.show()

def __plot_clusters__(features: np.ndarray, centroids: list, assignements: np.ndarray, colors: list):

    #features = np.transpose(features)
    #plt.scatter(features[0], features[1], color='blue')
    for row in range(len(features)):
        for cluster in range(len(assignements[row])):
            if assignements[row][cluster] == 1:
                plt.scatter(features[row][0], features[row][1], color=colors[cluster])
                break


    for index, centroid in enumerate(centroids):
        plt.scatter(centroid[0], centroid[1], color=colors[index], marker='x', s = 100)

    plt.show()

def __plot_gaussian_clusters__(features: np.ndarray, assignements: np.ndarray, colors: list):

    for row in range(len(features)):
        for cluster in range(len(assignements[row])):
            if assignements[row][cluster] == 1:
                plt.scatter(features[row][0], features[row][1], color=colors[cluster])
                break
    plt.show()

def __plot_DBSCAN_clusters__(features: np.ndarray, assignements: np.ndarray, colors: list):

    for row in range(len(features)):
        assigned = False

        for cluster in range(len(assignements[row])):
            if assignements[row][cluster] == 1:
                plt.scatter(features[row][0], features[row][1], color=colors[cluster])
                assigned = True
                break
            elif assignements[row][cluster] == 2:
                plt.scatter(features[row][0], features[row][1], color=colors[cluster], marker='x')
                assigned = True
                break
        # se non è classificato in nessun cluster
        if assigned is False:
            plt.scatter(features[row][0], features[row][1], color='grey', marker='o')

    plt.show()


def __plot_elbow__(elbow):

    plt.plot(elbow[1], elbow[0], c='blue')
    plt.show()

def __plot_principal_components__(features: np.ndarray, s: np.matrix = None):

    plt.scatter(features[:,:1], features[:,1:])

    if s is not None:
        s = s.transpose()
        s_list = s.tolist()
        x = np.linspace(-2,2,100)
        m1 = s_list[0][1] / s_list[0][0]
        y1 = m1*x

        m2 = s_list[1][1] / s_list[1][0]
        y2 = m2*x
        plt.plot(x, y1, c='blue')
        plt.plot(x, y2, c='red')

    plt.show()


