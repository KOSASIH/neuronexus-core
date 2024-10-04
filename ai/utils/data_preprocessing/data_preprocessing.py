import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreprocessing:
    def __init__(self):
        pass

    def load_data(self, file_path):
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")

    def split_data(self, data, test_size=0.2, random_state=42):
        X = data.drop('target', axis=1)
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test, scaling_method='standard'):
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'min_max':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaling method. Please choose 'standard' or 'min_max'.")

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def handle_missing_values(self, data, strategy='mean'):
        if strategy == 'mean':
            data.fillna(data.mean(), inplace=True)
        elif strategy == 'median':
            data.fillna(data.median(), inplace=True)
        elif strategy == 'mode':
            data.fillna(data.mode().iloc[0], inplace=True)
        else:
            raise ValueError("Invalid strategy for handling missing values. Please choose 'mean', 'median', or 'mode'.")

        return data

def main():
    data_preprocessing = DataPreprocessing()
    data = data_preprocessing.load_data('data.csv')
    X_train, X_test, y_train, y_test = data_preprocessing.split_data(data)
    X_train_scaled, X_test_scaled = data_preprocessing.scale_data(X_train, X_test)
    data = data_preprocessing.handle_missing_values(data)

if __name__ == "__main__":
    main()
