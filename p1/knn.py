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

testStr = "D:/OneDrive/Cours/master1/machine/projet1/p1/"

# (Question 2)
# Put your funtions here
def boundary(dataset_x, dataset_y, LS_size, n_neighbors) :
    for number in n_neighbors :
        neigh = KNeighborsClassifier(number)
        neigh.fit(dataset_x[0:LS_size], dataset_y[0:LS_size])
        plot_boundary(testStr + "n_neighbours" + str(number), neigh, dataset_x[LS_size + 1:], dataset_y[LS_size + 1:], title="Number of Neighbours = %s" % str(number))

def validation(dataset_x, dataset_y, n_neighbors, K=10):
        scores = [[], []]

        for number in n_neighbors :
            neigh = KNeighborsClassifier(number)
            fragment_size = len(dataset_y)//K
            score = 0
            for x in range(0, K-1):
                if x == 0 :
                    LS_X = dataset_x[fragment_size:]
                    LS_Y = dataset_y[fragment_size:]
                    TS_X = dataset_x[0:fragment_size]
                    TS_Y = dataset_y[0:fragment_size]
                    
                elif x == K-1:
                    LS_X = dataset_x[0:x * fragment_size]
                    LS_Y = dataset_y[0:x * fragment_size]
                    TS_X = dataset_x[x * fragment_size:]
                    TS_Y = dataset_y[x * fragment_size:]

                else:
                    LS_X = np.concatenate((dataset_x[0:x * fragment_size], (dataset_x[(x+1) * fragment_size:])))
                    LS_Y = np.concatenate((dataset_y[0:x * fragment_size], (dataset_y[(x+1) * fragment_size:])))
                    TS_X = dataset_x[x * fragment_size:(x+1) * fragment_size]
                    TS_Y = dataset_y[x * fragment_size:(x+1) * fragment_size:]

                neigh.fit(LS_X, LS_Y)
                score += neigh.score(TS_X, TS_Y)

            score /= K
            #print(str(number) + " scors : " + str(score))
            scores[0] += [number]
            scores[1] += [score]

        plt.figure()
        plt.plot(scores[0], scores[1])
        plt.savefig(testStr + "scores.pdf")
        plt.close()

if __name__ == "__main__":
    dataset_x, dataset_y = make_dataset2(1500, 687)
    n_neighbors = [1, 5, 25, 125, 625, 1200]
    boundary(dataset_x, dataset_y, 1200, n_neighbors)
    validation(dataset_x, dataset_y, n_neighbors)
