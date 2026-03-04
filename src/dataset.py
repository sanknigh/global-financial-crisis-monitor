import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class CrisisDataset(Dataset):
    def __init__(self, csv_path, window_size=12):
        self.window_size = window_size

        # Load CSV
        data = pd.read_csv(csv_path)

        # 🔥 Remove date column if exists
        if "Unnamed: 0" in data.columns:
            data = data.drop(columns=["Unnamed: 0"])

        data = data.dropna()

        # Separate label
        labels = data["Crisis"].values

        # Drop Crisis column
        features = data.drop(columns=["Crisis"])

        # 🔥 Keep only numeric columns
        features = features.select_dtypes(include=[np.number])

        features = features.values

        # 🔥 Scale features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        X = []
        y = []

        for i in range(len(features) - window_size):
            X.append(features[i:i + window_size])
            y.append(labels[i + window_size])

        X = np.array(X)
        y = np.array(y)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    