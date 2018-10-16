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

        for i in range(len(y)):
            index = classes.index(y[i])
            samples[index].append(X[i])


        means = [None]*nb_classes
        covariances = [None]*nb_classes
        for i in range(len(classes)):
            means[i] = np.mean(np.array(samples[i]), axis = 0)
            covariances[i] = np.cov(np.array(samples[i]))

        print(means)
        print(covariances)

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

    N_POINTS = 1500

    X, y = make_dataset2(N_POINTS)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

