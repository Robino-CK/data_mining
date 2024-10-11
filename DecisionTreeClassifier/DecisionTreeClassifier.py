'''
    Contributors:    
        Patrick Junghenn (1140761) 
        Ellora Keemink (6529771)
        Robin Kollmann (0994359) 
'''

import numpy as np
import warnings
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import pydot


'''' 
    Node class returns a node object, to create decision tree:
    
    the node class stores the nodes children as attribute, so we can 
    create a tree structure with iterating over the root.
    
    Takes in 3 parameters:
    feature: int, the feature number of the feature to split on
    value: int, leaf node value of majority class
    split_value: int, the split value (in our case Gini Index)
    count_left: int, the number of samples in the left node
    count_right: int, the number of samples in the right node
    
    
'''
class Node():
    
    def __init__(self, feature:int=None, value:int=None, split_value:int=None, count_left:int=None, count_right:int=None):
        self.feature = feature
        self.value = value
        self.split_value = split_value
        self.left = None
        self.right = None
        self.count_left = count_left
        self.count_right = count_right
    
    def is_leaf(self):
        return self.left is None and self.right is None
    
    def __str__(self):
        return f'Node: {self.feature} {self.split_value}'



"""
    The DecisionTree class implementing of a decision tree classifier and uses the Node class to create the tree.
    
    The class has 5 (public) methods/functions:
    
    tree_grow(
        X:np.array, - input data to find the best split
        y:np.array, - target data  
        nmin:int, - minimum number of samples to split
        minleaf:int, - minimum number of samples in leaf
        nfeat:int - number of features to consider
        )
    is the method to grow the tree and doesn't return anything.
    
    depth() returns the depth of the tree.
    
    number_nodes() returns the number of nodes in the tree.
    
    tree_predict(X:np.array) takes input data and returns the predicted values as np.array.
    
    print_tree() prints the tree structure. (doesnt take any arguments or return anything) 
    
    
"""

class DecisionTree():

    def __init__(self):
        self.root = None
    
    def tree_grow(self, X:np.array, y:np.array, nmin:int, minleaf:int, nfeat:int) -> None:
        '''Train Decision Tree'''
        self.root = self._tree_grow(X, y, nmin, minleaf, nfeat)

    def _tree_grow(self, X, y, nmin, minleaf, nfeat) -> Node:
        count_left = len(y[y == 0])
        count_right = len(y[y == 1])
        value = 0 if count_left > count_right else 1
        # Early Stopping criteria
        if len(y) <= nmin: #y[np.argmax(np.bincount(y.astype(int)))] - this took way too long and was not really understandable
            return Node(value = value, count_left = count_left, count_right = count_right)
        if len(np.unique(y)) == 1 or len(y) == 0: 
            return Node(value = value, count_left = count_left, count_right = count_right)
        
        # Calculate Gini Index
        def gini_index(y):
            p = np.mean(y)
            return p * (1 - p)

        
        # Select Features 
        if nfeat is not None:
            n_feat = nfeat
            feat = np.random.choice(X.shape[1], n_feat, replace=False)
            X_subset = X[:, feat]
            n_feat = X_subset.shape[1]
        else:
            n_feat = X.shape[1]
            X_subset = X
        
        # Set initial Gini Index as Gini Index of all data in node
        gini = None
        best_gini = gini_index(y)
        best_feature = None
        best_split_value = None

        # Find best split feature and split value
        for i in range(n_feat):
            for split_value in np.unique(X_subset[:, i]):
                left = y[X_subset[:, i] < split_value]
                right = y[X_subset[:, i] >= split_value]
                gini = (len(left) / len(y)) * gini_index(left) + (len(right) / len(y)) * gini_index(right)
                if gini < best_gini and len(left) >= minleaf and len(right) >= minleaf:
                    best_gini = gini
                    best_feature = i
                    best_split_value = split_value


        # Split data
        x_left = X_subset[:, best_feature] < best_split_value
        x_right = X_subset[:, best_feature] >= best_split_value

        X_left = X[x_left]
        X_right = X[x_right]
        y_left = y[x_left]
        y_right = y[x_right]

        # Create Node
        if best_feature is None:
            return Node(value =value, count_left = count_left, count_right = count_right)
        elif nfeat is not None:
            node = Node(feature = feat[int(best_feature)], split_value = best_split_value, count_left = count_left, count_right = count_right)
        else:
            node = Node(feature = int(best_feature), split_value = best_split_value, count_left = count_left, count_right = count_right)

        # Recursively grow tree
        node.left = self._tree_grow(X_left, y_left, nmin, minleaf, nfeat)
        node.right = self._tree_grow(X_right, y_right, nmin, minleaf, nfeat)

        return node

    def tree_predict(self, X:np.array) -> np.array:
        '''Return decision tree predictions'''

        pred = []
        for row in X:
            pred.append(self._tree_predict(self.root, row))

        return np.array(pred).astype(int)
    
    def _tree_predict(self, node:Node, X:np.array):
        if node.is_leaf():
            return node.value
        if X[node.feature] < node.split_value:
            return self._tree_predict(node.left, X)
        else:
            return self._tree_predict(node.right, X)
    
    def plot_tree(self, max_depth:int=None, feature_names:dict=None):
        '''Create Tree Plot'''
        G = nx.DiGraph()
        nodelabels = {}
        nx.draw(G,  with_labels = True)
        self._plot_tree(self.root, G, nodelabels, max_depth,0, feature_names)
        pos = graphviz_layout(G, prog="dot")
        #nx.draw_networkx_edge_labels(G, pos,labels=nodelabels,edge_labels=edgelabels, with_labels=True)
        nx.draw(G,pos, labels=nodelabels,with_labels=True, node_shape="s", node_size=10000, font_size=15)
        
    
    def _plot_tree(self, node:Node, G:nx.DiGraph,nodelabels:dict, max_depth:int=None, current_depth:int=0, feature_names:None=dict): 
        if node is None:
            return
        
        if node.is_leaf():
            G.add_node(hash(node))
            nodelabels[hash(node)] =  "Prediction:\n" + str(node.value) + "\n"
            nodelabels[hash(node)] +=  "Left: " + str(node.count_left) + "\nRight: "+  str(node.count_right)

            
        else:
            if feature_names:
                nodelabels[hash(node)] =  "Feature:\n" + feature_names[node.feature] + "\n"
                
            else: 
                nodelabels[hash(node)] =  "Feature:\n" + str(node.feature) + "\n"

            
            
            nodelabels[hash(node)] +=  "Left: " + str(node.count_left) + "\nRight: "+  str(node.count_right)
            
            G.add_node(hash(node))
            
            if max_depth is not None:
                if max_depth == current_depth:
                    return
            G.add_edge(hash(node), hash(node.left))
            G.add_edge(hash(node), hash(node.right))
            
            self._plot_tree(node.left, G, nodelabels, max_depth, current_depth + 1, feature_names)
            self._plot_tree(node.right, G, nodelabels, max_depth, current_depth + 1, feature_names)
    
    
    def number_nodes(self):
        '''Return total number of nodes in tree'''
        return self._number_nodes(self.root)
    
    def _number_nodes(self, node:Node):
        if node is None:
            return 0
        return 1 + self._number_nodes(node.left) + self._number_nodes(node.right)
    
    def depth(self):
        '''Return Max Depth of Tree'''
        return self._depth(self.root)
    
    def _depth(self, node:Node):
        return max(self._depth(node.left), self._depth(node.right)) + 1 if node is not None else 0

    def print_tree(self):
        '''Print Tree Structure'''
        self._print_tree(self.root, 0)

    def _print_tree(self, node:Node, depth:int):
        if node is None:
            return
        if node.is_leaf():
            print(' ' * depth, 'Leaf: ', node.value)
        else:
            print(' ' * depth, 'Node: ', node.feature, node.split_value)
            self._print_tree(node.left, depth + 1)
            self._print_tree(node.right, depth + 1)


"""
    The RandomForest class implements of a Random Forest Tree and uses the Decision Tree for implementation.
    The class has 2 (public) methods/functions:
    
    tree_grow_b(
        X:np.array, - input data to find the best split
        y:np.array, - target data  
        nmin:int, - minimum number of samples to split
        minleaf:int, - minimum number of samples in leaf
        nfeat:int - number of features to consider,
        m:int - number of trees to grow
        )
    is the method to grow the tree and doesn't return anything.
    
    
    tree_pred_b(
        X:np.array, - input data to predict on
    )  returns the predicted values as np.array.
    
    
    
"""

class RandomForest():

    def __init__(self):
        self.trees = []

    def tree_grow_b(self, X:np.array, y:np.array, nmin:int, minleaf:int, nfeat:int, m:int):
        '''Train Random Forest'''

        for i in range(m):
            replacement = np.random.choice(X.shape[0], X.shape[0], replace=True)
            x_sample = X[replacement]
            y_sample = y[replacement]
            dt = DecisionTree()
            dt.tree_grow(x_sample, y_sample, nmin, minleaf, nfeat)
            self.trees.append(dt)
     




    def tree_pred_b(self, X:np.array) -> np.array:
        '''Return decision tree predictions'''
        pred = []
        for row in X:

            pred_i = []
            for tree in self.trees:
                row = row.reshape(1, -1)
                pred_i.append(tree.tree_predict(row))

            pred_i = np.array(pred_i).flatten().astype(int)

            pred.append(np.argmax(np.bincount(pred_i)))

        return np.array(pred)
