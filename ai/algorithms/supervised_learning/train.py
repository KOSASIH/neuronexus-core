import numpy as np
from supervised_learning import SupervisedLearning, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, SupportVectorMachine

def main():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 0, 1])
    model = SupervisedLearning(LogisticRegression())
    model.train(X, y)

if __name__ == "__main__":
    main()
