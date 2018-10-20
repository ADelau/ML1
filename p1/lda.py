"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin



class LinearDiscriminantAnalysis(BaseEstimator, ClassifierMixin):


    def fit(self, X, y):
        """Fit a linear discriminant analysis model using the training set
        (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        # ====================
        # TODO your code here.
        # ====================

        classes = list(set(y))
        nb_classes = len(classes)
        samples = [[]]*nb_classes
        print(samples)

        #samples[i][j][k] correspond to the kth element of the jth sample corresponding to the class classes[i].
        for i in range(len(y)):
            index = classes.index(y[i])
            print(index)
            (samples[index]).append(X[i])
            print(samples)

        samples = np.asarray(samples, dtype = np.float)
        print(samples)


        means = [None]*nb_classes
        covariances = [None]*nb_classes
        for i in range(len(classes)):
            means[i] = np.mean(samples[i], axis = 0)
            covariances[i] = np.cov(samples[i], rowvar = False)

        #using homoscedasticity property
        covariances = np.asarray(covariances, dtype = np.float)
        covariance = np.mean(covariances, axis = 0)

        priorProba = []

        totalSamples = len(y)
        for i in range(len(classes)):
            priorProba.append(len(samples[i])/totalSamples) #len(samples[i]) = number of samples in class i

        print(means)
        print(covariance)
        print(priorProba)

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        # ====================
        # TODO your code here.
        # ====================

        pass

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        # ====================
        # TODO your code here.
        # ====================

        pass

if __name__ == "__main__":
    from data import make_dataset2
    from plot import plot_boundary

    N_POINTS = 10
    seed = 11

    X, y = make_dataset2(N_POINTS, seed)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

