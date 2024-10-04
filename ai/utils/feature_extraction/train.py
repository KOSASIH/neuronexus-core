import numpy as np
from feature_extraction import FeatureExtraction

def main():
    feature_extraction = FeatureExtraction()
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 0, 1])
    X_pca = feature_extraction.pca(X)
    X_selected =feature_extraction.select_k_best(X, y)
    X_selected = feature_extraction.recursive_feature_elimination(X, y)

if __name__ == "__main__":
    main()
