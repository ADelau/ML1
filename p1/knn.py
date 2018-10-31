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

saveDirectory = "D:/OneDrive/Cours/master1/machine/projet1/p1/"

def get_classifier(dataset_x, dataset_y, n_neighbors) :
    """
    Given a learning sanple and a number of neigbours, return a 
    KNeighborsClassifier object fitting this dataset with the given number
    of neighbours.

    Arguments:
    ----------
    - `dataset_x`: the X values of the LS.
    - `dataset_y`: the Y values of the LS.
    - `n_neighbors`: the number of neighbours for the KNeighborsClassifier.

    Return:
    -------
    - A KNeighborsClassifier object fitting the LS.
    """

    # Create classifier with the given number of neighbours
    neigh = KNeighborsClassifier(n_neighbors)

    # Fit the model over the learning sample
    neigh.fit(dataset_x, dataset_y)

    return neigh

def draw_boundary(classifier, dataset_x, dataset_y, n_neighbors):
    """
    Given a dataset, and a KNeighborsClassifier already trained with 
    'n_neighbors' neighbours, plot the boundary for the dataset.

    Arguments:
    ----------
    - `classifier`: a KNeighborsClassifier already trained.
    - `dataset_x`: the X values of the dataset.
    - `dataset_y`: the Y values of the dataset.
    - `n_neighbors`: the number of neighbours for the KNeighborsClassifier.
    """

    #Plot the boundaries for the test samples
    plot_boundary(saveDirectory + "n_neighbours" + str(n_neighbors), 
                    classifier, dataset_x, dataset_y,
                    title="Number of Neighbours = %s" % str(n_neighbors))

def cross_validation(dataset_x, dataset_y, n_neigbours, KFlod=10):
    """
    Given a dataset, and a KNeighborsClassifier already trained with 
    'n_neighbors' neighbours, plot the boundary for the dataset.

    Arguments:
    ----------
    - `dataset_x`: the X values of the dataset.
    - `dataset_y`: the Y values of the dataset.
    - `n_neighbors`: the number of neighbours for the KNeighborsClassifier.
    - `KFlod`: the number of fold in the cross validation.

    Return:
    -------
    - The score of the cross validation.
    """
    # Create classifier with the given number of neighbours
    neigh = KNeighborsClassifier(number)

    # Compute the cross-validation score of our data
    crossScore = cross_val_score(neigh, dataset_x, dataset_y,
                                 cv=KFlod, n_jobs=-1)

    return mean(crossScore)

if __name__ == "__main__":
    seed = 687

    dataset_x, dataset_y = make_dataset2(1500, seed)
    n_neighbors = [1, 5, 25, 125, 625, 1200]

    LS_size = 1200

    X_train, X_test, y_train, y_test = \
        train_test_split(dataset_x, dataset_y, test_size = 0.2,random_state = seed)
    # Question 1
    for number in n_neighbors:
        classifier = get_classifier(dataset_x[0:LS_size], dataset_y[0:LS_size], number)
        draw_boundary(classifier, dataset_x[LS_size + 1:], dataset_y[LS_size + 1:], number)

    # Question 2
    scores = [[], []]
    step = 100

    # Loop over the number of neighbours used to the classifier
    for number in range(1,1200,step) :

        # Get the cross validation score
        crossScore = cross_validation(dataset_x, dataset_y, number)
        
        # Add this score to the array
        scores[0].append(number)
        scores[1].append(crossScore)

    print("Best number of neighbours = ", scores[1].index(max(scores[1])) * step)
    print("Score of the best number of neighbours = ", max(scores[1]))
    
    plt.figure()
    plt.plot(scores[0], scores[1])
    plt.savefig(saveDirectory + "scores.pdf")
    plt.show()
    plt.close()
