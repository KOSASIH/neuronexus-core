import numpy as np
from decision_tree import DecisionTree, train, test

def main():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 0, 1])
    model = DecisionTree(max_depth=2)
    train(model, X, y)
    accuracy = test(model, X, y)
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
