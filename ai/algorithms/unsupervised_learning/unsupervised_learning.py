import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

class UnsupervisedLearning:
    def __init__(self, model):
        self.model = model

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

class KMeansClustering:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

class PrincipalComponentAnalysis:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.model = PCA(n_components=n_components)

    def fit(self, X):
        self.model.fit(X)

    def transform(self, X):
        return self.model.transform(X)

class GaussianMixtureModel:
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.model = GaussianMixture(n_components=n_components)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

def main():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    model = UnsupervisedLearning(KMeansClustering())
    model.fit(X)
    predictions = model.predict(X)
    print(predictions)

if __name__ == "__main__":
    main()
