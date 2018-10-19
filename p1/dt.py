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
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from plot import plot_boundary
import graphviz



# (Question 1)

# Put your funtions here
# ...

N_POINTS = 1500
depth = [1, 2, 4, 8, None]
seed = 11

def create_trees(X, y):
    
    decTrees = []

    for i in range(len(depth)):
        decTree = DecisionTreeClassifier(max_depth = depth[i])
        decTree.fit(X, y)
        decTrees.append(decTree)

    return decTrees

def make_plot():
    
    files = ["decTree_1", "decTree_2", "decTree_4", "dec_Tree_8", "decTree_none"]
    filesTree = ["tree_" + x for x in files]

    X, y = make_dataset2(N_POINTS, seed)

    decTrees = create_trees(X, y)

    for i in range(len(depth)):
        plot_boundary(files[i], decTrees[i], X, y) #Il faut mettre les test et pas les train?
        
        graph = graphviz.Source(export_graphviz(decTrees[i], out_file = None))
        graph.render(filesTree[i], view = False)

def compute_statistics():
    NB_TEST = 5

    accuracies = [[0] * len(depth)] * NB_TEST

    for i in range(NB_TEST):
        X, y = make_dataset2(N_POINTS, seed)
        trainSetX, testSetX, trainSetY, testSetY = train_test_split(X, y, test_size = 0.2)

        decTrees = create_trees(trainSetX, trainSetY)
        accuracies[i] = [accuracy_score(testSetY, decTree.predict(testSetX)) for decTree in decTrees]

    accuracies = np.array(accuracies)
    mean = np.mean(accuracies, axis = 1)
    std = np.std(accuracies, axis = 1)

    return mean, std

if __name__ == "__main__":

    #make_plot()
    mean, std = compute_statistics()
    print("mean = {}".format(mean))
    print("std = {}".format(std))
