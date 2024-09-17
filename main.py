import numpy as np
import pandas as pd

from DecisionTreeClassifier import DecisionTreeClassifier

# Load data
credit = np.loadtxt('credit.txt', delimiter=',', skiprows=1)
print(credit.shape)

pima = np.loadtxt('pima.txt', delimiter=',', skiprows=0)
print(pima.shape)

## Decision Tree

# Init decision tree
dt = DecisionTreeClassifier.DecisionTree()
dt.tree_grow(credit[:, :-1], credit[:, -1], nmin=2, minleaf=2, nfeat=2)
dt.print_tree()

# Test decision tree
print(dt.tree_predict(credit[0, :-1]))

## Random Forest

# Init random forest
rf = DecisionTreeClassifier.RandomForest()

# Test random forest
