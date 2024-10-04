import numpy as np
from data_preprocessing import DataPreprocessing

def main():
    data_preprocessing = DataPreprocessing()
    data = data_preprocessing.load_data('data.csv')
    X_train, X_test, y_train, y_test = data_preprocessing.split_data(data)
    X_train_scaled, X_test_scaled = data_preprocessing.scale_data(X_train, X_test)
    data = data_preprocessing.handle_missing_values(data)

if __name__ == "__main__":
    main()
