import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2

class FeatureExtraction:
    def __init__(self):
        pass

    def pca(self, X, n_components=2):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        return X_pca

    def select_k_best(self, X, y, k=10):
        selector = SelectKBest(chi2, k=k)
        X_selected = selector.fit_transform(X, y)
        return X_selected

    def recursive_feature_elimination(self, X, y, n_features=10):
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
        estimator = LogisticRegression()
        selector = RFE(estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        return X_selected

def main():
    feature_extraction = FeatureExtraction()
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 0, 1])
    X_pca = feature_extraction.pca(X)
    X_selected = feature_extraction.select_k_best(X, y)
    X_selected = feature_extraction.recursive_feature_elimination(X, y)

if __name__ == "__main__":
    main()
