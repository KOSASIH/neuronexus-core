import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = {}

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _build_tree(self, X, y):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (self.max_depth is not None and self.max_depth == 1) or n_labels == 1 or n_features == 0:
            leaf_value = np.argmax(np.bincount(y))
            return leaf_value

        feat_idxs = np.random.choice(n_features, n_features, replace=False)
        best_feat = None
        best_thr = None
        best_gain = -1
        for idx in feat_idxs:
            X_column = X[:, idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = idx
                    best_thr = threshold

        if best_feat is None:
            leaf_value = np.argmax(np.bincount(y))
            return leaf_value

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thr)
        left = self._build_tree(X[left_idxs, :], y[left_idxs])
        right = self._build_tree(X[right_idxs, :], y[right_idxs])
        return {"feature": best_feat, "threshold": best_thr, "left": left, "right": right}

    def _predict(self, inputs):
        node = self.tree
        while isinstance(node, dict):
            feature = node["feature"]
            threshold = node["threshold"]
            if inputs[feature] <= threshold:
                node = node["left"]
            else:
                node = node["right"]

        return node

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        e1 = self._entropy(y[left_idxs])
        e2 = self._entropy(y[right_idxs])

        child_entropy = (len(left_idxs) / n) * e1 + (len(right_idxs) / n) * e2

        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

def train(model, X, y):
    model.fit(X, y)

def test(model, X, y):
    y_pred = model.predict(X)
    accuracy = np.sum(y_pred == y) / len(y)
    return accuracy

def main():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np .array([0, 0, 1])
    model = DecisionTree(max_depth=2)
    train(model, X, y)
    accuracy = test(model, X, y)
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
