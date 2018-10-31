"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
from sklearn.model_selection import train_test_split
from plot import plot_boundary
from sklearn.metrics import accuracy_score

from sklearn.base import BaseEstimator, ClassifierMixin

N_POINTS = 1500
SEED = 11

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

        self.classes = sorted(list(set(y)))
        nb_classes = len(self.classes)

        samples = []
        for i in range(nb_classes):
            samples.append([])

        #samples[i][j][k] correspond to the kth element of the jth sample corresponding to the class classes[i].
        for i in range(len(y)):
            index = self.classes.index(y[i])
            samples[index].append(X[i])

        self.means = [None]*nb_classes
        covariances = [None]*nb_classes
        for i in range(len(self.classes)):
            samples[i] = np.asarray(samples[i], dtype = np.float)
            self.means[i] = np.mean(samples[i], axis = 0)
            covariances[i] = np.cov(samples[i], rowvar = False)

        #using homoscedasticity property
        covariances = np.asarray(covariances, dtype = np.float)
        self.covariance = np.mean(covariances, axis = 0)

        self.priorProba = []

        totalSamples = len(y)
        for i in range(len(self.classes)):
            self.priorProba.append(len(samples[i])/totalSamples) #len(samples[i]) = number of samples in class i

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

        proba = self.predict_proba(X)
        proba = np.asarray(proba, dtype = np.float)
        predicted = [self.classes[np.argmax(sampleProba)] for sampleProba in proba]
        return predicted

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

        p = []

        for sample in X:
            densities = []
            for i in range(len(self.classes)):
                densities.append(self.probX(sample, i))

            denum = sum(list(map(lambda x, y: x * y, densities, self.priorProba)))

            probabilities = []
            for i in range(len(self.classes)): 
                probabilities.append(densities[i] * self.priorProba[i] / denum)

            p.append(probabilities)

        return np.array(p)

    def probX(self, X, classIndex):
        #compute fkx
        p = len(X)
        diffMeanX = list(map(lambda a, b: a - b, X , self.means[classIndex]))
        expArg = -0.5 * np.dot(np.dot(diffMeanX, np.linalg.inv(self.covariance)), diffMeanX)
        coefficient = 1/((2*math.pi)**(p/2) * math.sqrt(np.linalg.det(self.covariance)))
        return coefficient * math.exp(expArg)


    def __init__(self):
        self.classes = None
        self.means = None
        self.covariance = None
        self.priorProba = None

def plot_decision_boundary():
    files =  ["lda_dataset1", "lda_dataset2"]
    X = []
    y = []

    tmpX, tmpY = make_dataset1(N_POINTS, SEED)
    X.append(tmpX)
    y.append(tmpY)

    tmpX, tmpY = make_dataset2(N_POINTS, SEED)
    X.append(tmpX)
    y.append(tmpY)

    for i in range(len(files)):
        trainSetX, testSetX, trainSetY, testSetY = train_test_split(X[i], y[i], test_size = 0.2, random_state = SEED)
        estimator = LinearDiscriminantAnalysis()
        estimator.fit(trainSetX, trainSetY)
        
        plot_boundary(files[i], estimator, testSetX, testSetY)

def compute_statistics():
    SEEDS = [11, 13, 17, 23, 27]

    accuracies1 = []
    accuracies2 = []

    for i in range(len(SEEDS)):
        X1, y1 = make_dataset1(N_POINTS, SEEDS[i])
        X2, y2 = make_dataset2(N_POINTS, SEEDS[i])
        trainSetX1, testSetX1, trainSetY1, testSetY1 = train_test_split(X1, y1, test_size = 0.2, random_state = SEEDS[i])
        trainSetX2, testSetX2, trainSetY2, testSetY2 = train_test_split(X2, y2, test_size = 0.2, random_state = SEEDS[i])

        lda1 = LinearDiscriminantAnalysis()
        lda2 = LinearDiscriminantAnalysis()
        lda1.fit(trainSetX1, trainSetY1)
        lda2.fit(trainSetX2, trainSetY2)

        accuracies1.append(accuracy_score(testSetY1, lda1.predict(testSetX1)))
        accuracies2.append(accuracy_score(testSetY2, lda2.predict(testSetX2)))

    accuracies1 = np.array(accuracies1)
    accuracies2 = np.array(accuracies2)

    mean1 = np.mean(accuracies1)
    mean2 = np.mean(accuracies2)
    std1 = np.std(accuracies1)
    std2 = np.std(accuracies2)

    return mean1, std1, mean2, std2

if __name__ == "__main__":
    from data import make_dataset1
    from data import make_dataset2
    from plot import plot_boundary


    X, y = make_dataset1(N_POINTS, SEED)
    trainSetX, testSetX, trainSetY, testSetY = train_test_split(X, y, test_size = 0.2, random_state = SEED)

    lda = LinearDiscriminantAnalysis()
    lda.fit(trainSetX, trainSetY)

    print("classe = {}".format(lda.classes))
    print("means = {}".format(lda.means))
    print("covariance = {}".format(lda.covariance))
    print("priorProba = {}".format(lda.priorProba))

    proba = lda.predict_proba(testSetX)
    print("proba = {}".format(proba))
    predicted = lda.predict(testSetX)
    print("predicted = {}".format(predicted))

    plot_decision_boundary()

    mean1, std1, mean2, std2 = compute_statistics()
    print("mean1 = {}\n".format(mean1))
    print("mean2 = {}\n".format(mean2))
    print("std1 = {}\n".format(std1))
    print("std2 = {}\n".format(std2))

