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

import graphviz



# (Question 1)

# Put your funtions here
# ...
def create_trees(X, y):
	
    decTree_1 = DecisionTreeClassifier(max_depth = 1)
    decTree_2 = DecisionTreeClassifier(max_depth = 2)
    decTree_4 = DecisionTreeClassifier(max_depth = 4)
    decTree_8 = DecisionTreeClassifier(max_depth = 8)
    decTree_none = DecisionTreeClassifier(max_depth = None)

    decTrees = [decTree_1, decTree_2, decTree_4, decTree_8, decTree_none]

    for decTree in decTrees:
    	decTree.fit(X, y)

    return decTrees


if __name__ == "__main__":
    N_POINTS = 1500

    dataset1_X, dataset1_y = make_dataset1(N_POINTS)
    dataset2_X, dataset2_y = make_dataset2(N_POINTS)

    decTrees1 = create_trees(dataset1_X, dataset1_y)
    decTrees2 = create_trees(dataset2_X, dataset2_y)

    graph = graphviz.Source(export_graphviz(decTrees1[0], out_file = None))
    graph.render("test", view = True)
