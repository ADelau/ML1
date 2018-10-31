"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from data import make_dataset1, make_dataset2
from plot import plot_boundary
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from statistics import mean
from sklearn.model_selection import train_test_split

def get_classifier(datasetX, datasetY, nNeighbors) :
    """
    Given a learning sanple and a number of neigbours, return a 
    KNeighborsClassifier object fitting this dataset with the given number
    of neighbours.

    Arguments:
    ----------
    - `datasetX`: the X values of the LS.
    - `datasetY`: the Y values of the LS.
    - `nNeighbors`: the number of neighbours for the KNeighborsClassifier.

    Return:
    -------
    - A KNeighborsClassifier object fitting the LS.
    """

    # Create classifier with the given number of neighbours
    neigh = KNeighborsClassifier(nNeighbors)

    # Fit the model over the learning sample
    neigh.fit(datasetX, datasetY)

    return neigh

def draw_boundary(classifier, datasetX, datasetY, nNeighbors):
    """
    Given a dataset, and a KNeighborsClassifier already trained with 
    'n_neighbors' neighbours, plot the boundary for the dataset.

    Arguments:
    ----------
    - `classifier`: a KNeighborsClassifier already trained.
    - `datasetX`: the X values of the dataset.
    - `datasetY`: the Y values of the dataset.
    - `nNeighbors`: the number of neighbours for the KNeighborsClassifier.
    """

    #Plot the boundaries for the test samples
    plot_boundary("n_neighbours" + str(nNeighbors), 
                    classifier, datasetX, datasetY,
                    title="Number of Neighbours = %s" % str(nNeighbors))

def cross_validation(datasetX, datasetY, nNeighbors, KFlod=10):
    """
    Given a dataset, and a KNeighborsClassifier already trained with 
    'nNeighbors' neighbours, plot the boundary for the dataset.

    Arguments:
    ----------
    - `datasetX`: the X values of the dataset.
    - `datasetY`: the Y values of the dataset.
    - `nNeighbors`: the number of neighbours for the KNeighborsClassifier.
    - `KFlod`: the number of fold in the cross validation.

    Return:
    -------
    - The score of the cross validation.
    """
    # Create classifier with the given number of neighbours
    neigh = KNeighborsClassifier(nNeighbors)

    # Compute the cross-validation score of our data
    crossScore = cross_val_score(neigh, datasetX, datasetY,
                                 cv=KFlod, n_jobs=-1)

    return mean(crossScore)

if __name__ == "__main__":
    SEED = 687
    N_POINTS = 1500

    datasetX, datasetY = make_dataset2(N_POINTS, SEED)
    N_NEIGHBOURS = [1, 5, 25, 125, 625, 1200]

    # Split the dataset in Traning and Learning set
    xTrain, xTest, yTrain, yTest = \
        train_test_split(datasetX, datasetY, test_size = 0.2,random_state = SEED)

    # Question 1
    for number in N_NEIGHBOURS:
        classifier = get_classifier(xTrain, yTrain, number)
        draw_boundary(classifier, xTest, yTest, number)

    # Question 2
    scores = [[], []]
    STEP = 100

    # Loop over the number of neighbours used to the classifier
    for number in range(N_NEIGHBOURS[0], N_NEIGHBOURS[-1], STEP) :

        # Get the cross validation score
        crossScore = cross_validation(datasetX, datasetY, number)
        
        # Add this score to the array
        scores[0].append(number)
        scores[1].append(crossScore)

    print("Best number of neighbours = ", scores[1].index(max(scores[1])) * STEP)
    print("Score of the best number of neighbours = ", max(scores[1]))
    
    plt.figure()
    plt.plot(scores[0], scores[1])
    plt.savefig("scores.pdf")
    plt.close()
