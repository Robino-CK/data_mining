import numpy as np
import warnings
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import pydot
# Need to fix warnings
warnings.filterwarnings('ignore')


'''' 
    Node class returns a node object, to create decision tree:
    
    the node class stores the nodes children as attribute, so we can 
    create a tree structure with iterating over the root.
    
    Takes in 3 parameters:
    feature: int, the feature number of the feature to split on
    value: int, leaf node value of majority class?? # TODO: check this, I think it is always 1, if 1 exists in the leaf, otherwise 0. (bc argmax)
    split_value: int, the split value (in our case Gini Index)
    
'''
class Node():
    
    def __init__(self, feature:int=None, value:int=None, split_value:int=None):
        self.feature = feature
        self.value = value
        self.split_value = split_value
        self.left = None
        self.right = None
    
    def is_leaf(self):
        return self.left is None and self.right is None
    
    def __str__(self):
        return f'Node: {self.feature} {self.split_value}'

"""
    The DecisionTree class implementing of a decision tree classifier and uses the Node class to create the tree.
    
    The class has 3 (public) methods/functions:
    
    tree_grow(
        X:np.array, - input data to find the best split
        y:np.array, - target data  
        nmin:int, - minimum number of samples to split
        minleaf:int, - minimum number of samples in leaf
        nfeat:int - number of features to consider
        )
    is the method to grow the tree and doesn't return anything.
    
    
    tree_predict(X:np.array) takes input data and returns the predicted values as np.array.
    
    print_tree() prints the tree structure. (doesnt take any arguments or return anything) 
    
    
"""
class DecisionTree():

    def __init__(self):
        self.root = None
    
    """ """
    def tree_grow(self, X:np.array, y:np.array, nmin:int, minleaf:int, nfeat:int) -> None:
        self.root = self._tree_grow(X, y, nmin, minleaf, nfeat)

    def _tree_grow(self, X, y, nmin, minleaf, nfeat) -> Node:
        
        # Early Stopping criteria
        if len(y) <= nmin:
            return Node(value = y[np.argmax(y)])
        if len(np.unique(y)) == 1 or len(y) == 0:
            return Node(value = y[0])
        
        # Find best split
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
        
        gini = None
        best_gini = gini_index(y)
        best_feature = None
        best_split_value = None

        for i in range(n_feat):
            for split_value in np.unique(X_subset[:, i]):
                left = y[X_subset[:, i] < split_value]
                right = y[X_subset[:, i] >= split_value]
                gini = (len(left) / len(y)) * gini_index(left) + (len(right) / len(y)) * gini_index(right)
                if gini < best_gini and len(left) >= minleaf and len(right) >= minleaf:
                    best_gini = gini
                    best_feature = i
                    best_split_value = split_value

        # Create Node
        if best_feature is None:
            return Node(value = y[np.argmax(y)])
        elif nfeat is not None:
            node = Node(feature = feat[int(best_feature)], split_value = best_split_value)
        else:
            node = Node(feature = int(best_feature), split_value = best_split_value)

        # Split data
        x_left = X_subset[:, best_feature] < best_split_value
        x_right = X_subset[:, best_feature] >= best_split_value

        X_left = X[x_left]
        X_right = X[x_right]
        y_left = y[x_left]
        y_right = y[x_right]

        # Recursively grow tree
        node.left = self._tree_grow(X_left, y_left, nmin, minleaf, nfeat)
        node.right = self._tree_grow(X_right, y_right, nmin, minleaf, nfeat)

        return node

    def tree_predict(self, X:np.array) -> np.array:
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
    
    def plot_tree(self):
        
        G = nx.DiGraph()
        nodelabels = {}
        nx.draw(G,  with_labels = True)
        self._plot_tree(self.root, G, nodelabels)
        pos = graphviz_layout(G, prog="dot")
        #nx.draw_networkx_edge_labels(G, pos,labels=nodelabels,edge_labels=edgelabels, with_labels=True)
        nx.draw(G,pos, labels=nodelabels,with_labels=True)
        
    
    def _plot_tree(self, node:Node, G:nx.DiGraph,nodelabels:dict):
        if node is None:
            return
        
        #nodelabels[hash(node)] =  " Gini index: " + str(node.split_value)
        
        if node.is_leaf():
            G.add_node(hash(node))
            nodelabels[hash(node)] =  "prediction " + str(node.value)
       
        else:
            nodelabels[hash(node)] =  "Splitting feature " + str(node.feature)
       
            G.add_node(hash(node))

            G.add_edge(hash(node), hash(node.left))
            G.add_edge(hash(node), hash(node.right))
            self._plot_tree(node.left, G, nodelabels)
            self._plot_tree(node.right, G, nodelabels)
    

    def print_tree(self):
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
        X:np.array, - input data 
        tree_list:list[DecisionTree], - TODO: list of trees to predict 
        prop:bool=False - TODO: what is it doing?
    )  returns the predicted values as np.array.
    
    
    
"""

class RandomForest():

    def __init__(self):
        self.trees = []

    def tree_grow_b(self, X:np.array, y:np.array, nmin:int, minleaf:int, nfeat:int, m:int):
        for i in range(m):
            dt = DecisionTree()
            dt.tree_grow(X, y, nmin, minleaf, nfeat)
            self.trees.append(dt)



    def tree_pred_b(self, X:np.array, prop:bool=False) -> np.array:
        pred = []
        for row in X:

            pred_i = []
            for tree in self.trees:
                row = row.reshape(1, -1)
                pred_i.append(tree.tree_predict(row))

            pred_i = np.array(pred_i).flatten().astype(int)

            if prop:
                pred.append(np.bincount(pred_i) / len(pred_i))
            else:
                pred.append(np.argmax(np.bincount(pred_i)))

        return np.array(pred)
