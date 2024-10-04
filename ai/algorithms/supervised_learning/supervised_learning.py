import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class SupervisedLearning:
    def __init__(self, model):
        self.model = model

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.w) + self.b
            y_predicted = self._sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.w) + self.b
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

class DecisionTreeClassifier:
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

    def _ split(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X):
        predictions = [tree.predict(X) for tree in self.trees]
        predictions = np.array(predictions).T
        predictions = [np.bincount(prediction).argmax() for prediction in predictions]
        return np.array(predictions)

class SupportVectorMachine:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

def main():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 0, 1])
    model = SupervisedLearning(LogisticRegression())
    model.train(X, y)

if __name__ == "__main__":
    main()
