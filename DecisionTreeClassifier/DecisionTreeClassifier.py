import numpy as np
import warnings

# Need to fix warnings
warnings.filterwarnings('ignore')

class Node():

    def __init__(self, feature=None, value=None, split_value=None):
        self.feature = feature
        self.value = value
        self.split_value = split_value
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None and self.right is None
    
    def __str__(self):
        return f'Node: {self.feature} {self.split_value}'


class DecisionTree():

    def __init__(self):
        self.root = None
    
    def tree_grow(self, X, y, nmin, minleaf, nfeat):
        self.root = self._tree_grow(X, y, nmin, minleaf, nfeat)

    def _tree_grow(self, X, y, nmin, minleaf, nfeat):
        
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

    def tree_predict(self, X):
        pred = []
        for row in X:
            pred.append(self._tree_predict(self.root, row))

        return np.array(pred).astype(int)
    
    def _tree_predict(self, node, X):
        if node.is_leaf():
            return node.value
        if X[node.feature] < node.split_value:
            return self._tree_predict(node.left, X)
        else:
            return self._tree_predict(node.right, X)

    def print_tree(self):
        self._print_tree(self.root, 0)

    def _print_tree(self, node, depth):
        if node is None:
            return
        if node.is_leaf():
            print(' ' * depth, 'Leaf:', node.value)
        else:
            print(' ' * depth, 'Node:', node.feature, node.split_value)
            self._print_tree(node.left, depth + 1)
            self._print_tree(node.right, depth + 1)


class RandomForest():

    def __init__(self):
        self.trees = []

    def tree_grow_b(self, X, y, nmin, minleaf, nfeat, m):
        for i in range(m):
            dt = DecisionTree()
            dt.tree_grow(X, y, nmin, minleaf, nfeat)
            self.trees.append(dt)

    def tree_pred_b(self, X, prop=False):
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
