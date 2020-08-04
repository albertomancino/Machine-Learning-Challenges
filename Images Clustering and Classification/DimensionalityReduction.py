import numpy as np
import Preprocessing as pre

class PCA:

    def __init__(self, features :np.array):

        self.features = features.transpose()
        print('zero to mean')
        self.__to_zero_mean__()
        print('covariance')
        #covariance = self.__covariance__()
        self.actual_covariance = np.cov(self.features)
        print('u,d,v')
        self.u, self.d, self.v = np.linalg.svd(self.actual_covariance)

    def __do__(self, new_dimension):

        reducted_features = self.__reduction__(self.u, new_dimension)
        return reducted_features


    def __to_zero_mean__(self):
        mean = self.features.mean(axis=0)
        self.features = self.features - mean
        '''
        std = np.power(self.features, 2)
        std = std.mean(axis=0)
        std = np.sqrt(std)
        self.features = np.divide(self.features, std)
        '''

    def __covariance__(self, features = None):

        covariance = None

        if features is None:
            features = self.features

        for feature in features:
            feature = np.matrix(feature)
            cov = np.dot(feature.transpose(), feature)
            if covariance is None:
                covariance = cov
            else:
                covariance += cov
        return np.divide(covariance, len(features))

    def __reduction__(self, u: np.matrix, new_dimension):

        u = np.array(u)
        selection = u[:,:new_dimension]
        new_features = np.dot(self.features.transpose(), selection)
        return new_features

    def __expansion__(self, features: np.array,  u: np.matrix, new_dimension: int):

        u = np.array(u)
        selection = u[:,:new_dimension]
        return np.dot(features, selection.transpose())

    def __variance_retained__(self, new_dimension):

        return self.d[:new_dimension].sum() / self.d.sum()

