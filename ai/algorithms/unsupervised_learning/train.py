import numpy as np
from unsupervised_learning import Unsup ervisedLearning, KMeansClustering, PrincipalComponentAnalysis, GaussianMixtureModel

def main():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    model = UnsupervisedLearning(KMeansClustering())
    model.fit(X)
    predictions = model.predict(X)
    print(predictions)

if __name__ == "__main__":
    main()
