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

def create_trees(X, y):

    depth = [1, 2, 4, 8, None]

    decTrees = []

    for i in range(len(depth)):
        decTree = DecisionTreeClassifier(max_depth = depth[i])
        decTree.fit(X, y)
        decTrees.append(decTree)

    return decTrees

def makePlot():
    
    
    files = ["decTree_1", "decTree_2", "decTree_4", "dec_Tree_8", "decTree_none"]
    filesTree = ["tree_" + x for x in files]

    X, y = make_dataset1(N_POINTS)

    decTrees = create_trees(X, y)

    for i in range(len(depth)):
        plot_boundary(files[i], decTrees[i], X, y) #Il faut mettre les test et pas les train?
        graph = graphviz.Source(export_graphviz(decTrees[i], out_file = None))
        graph.render(filesTree[i], view = False)

def compute_accuracies():
    NB_TEST = 5

    accuracies = [0] * 5

    for i in range(NB_TEST):
        X, y = make_dataset1(N_POINTS)
        trainSetX, testSetX, trainSetY, testSetY = train_test_split(X, y, test_size = 0.2)

        decTrees = create_trees(trainSetX, trainSetY)
        current_accuracy = [accuracy_score(testSetY, decTree.predict(testSetX)) for decTree in decTrees]
        accuracies = list(map(lambda x,y: x + y, accuracies, current_accuracy))

    accuracies = [x/5 for x in accuracies]

    return accuracies

if __name__ == "__main__":

    accuracies = compute_accuracies()
    print(accuracies)
   
    #graph = graphviz.Source(export_graphviz(decTrees1[0], out_file = None))
    #graph.render("test", view = True)


