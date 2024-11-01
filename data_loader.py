# data_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

class NDataset:
    def __init__(self, data):
        self.X = data.drop('label', axis=1)
        self.y = data['label']
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None

    def get_dataset(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=123)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train.values)
        y_test = torch.LongTensor(y_test.values)

        self.train_dataset = TensorDataset(X_train, y_train)
        self.test_dataset = TensorDataset(X_test, y_test)
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=64, shuffle=False)

def load_data(file_path):
    data = pd.read_csv(file_path)
    label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
    data['label'] = data['label'].map(label_mapping)
    return NDataset(data)