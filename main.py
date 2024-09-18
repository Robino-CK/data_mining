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
dt_credit = DecisionTreeClassifier.DecisionTree()
dt_credit.tree_grow(credit[:, :-1], credit[:, -1], nmin=1, minleaf=1, nfeat=None)

print("Credit Tree")
dt_credit.print_tree()

dt_pima = DecisionTreeClassifier.DecisionTree()
dt_pima.tree_grow(pima[:, :-1], pima[:, -1], nmin=1, minleaf=1, nfeat=None)

print("Pima Tree")
dt_pima.print_tree()


# # Test decision tree
print(dt_credit.tree_predict(credit[0:10, :-1]))
print(dt_pima.tree_predict(pima[0:10, :-1]))

## Random Forest

# Init random forest
rf = DecisionTreeClassifier.RandomForest()
rf.tree_grow_b(credit[:, :-1], credit[:, -1], nmin=1, minleaf=1, nfeat=2, m=3)

rf_pima = DecisionTreeClassifier.RandomForest()
rf_pima.tree_grow_b(pima[:, :-1], pima[:, -1], nmin=1, minleaf=1, nfeat=4, m=3)

# Test random forest
print(rf.tree_pred_b(credit[0:10, :-1]))
print(rf_pima.tree_pred_b(pima[0:10, :-1]))