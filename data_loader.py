import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

class DataLoaderModule:
    def __init__(self, filepath, batch_size=64):
        self.filepath = filepath
        self.batch_size = batch_size
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data()

    def load_data(self):
        data = pd.read_csv(self.filepath)
        label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
        data['label'] = data['label'].map(label_mapping)

        X = data.drop('label', axis=1)
        y = data['label']
        X = np.array(X)
        y = np.array(y)

        # Normalize the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

        return X_train, X_test, y_train, y_test

    def get_loaders(self):
        train_dataset = TensorDataset(self.X_train, self.y_train)
        test_dataset = TensorDataset(self.X_test, self.y_test)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader
