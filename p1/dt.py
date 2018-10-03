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


# (Question 1)

# Put your funtions here
# ...
def create_trees():
	dataset1_X, dataset1_y = make_dataset1
	dataset2_X, dataset2_y = make_dataset2

	decTree_1 = DecisionTreeClassifier(max_depth = 1)
	decTree_2 = DecisionTreeClassifier(max_depth = 2)
	decTree_4 = DecisionTreeClassifier(max_depth = 4)
	decTree_8 = DecisionTreeClassifier(max_depth = 8)
	decTree_none = DecisionTreeClassifier(max_depth = none)

	decTrees[] = {decTree_1, decTree_2, decTree_4, decTree_8, decTree_none}

	for(decTree in decTrees):
		decTree.fit(X, y)

	return decTrees


if __name__ == "__main__":
    pass # Make your experiments here
