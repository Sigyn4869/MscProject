import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

# load X array
X_loaded = np.load('X_array.npy')

# load Y data
Y_loaded = pd.read_pickle('Y_series.pkl')

X_original = X_loaded
Y_original = Y_loaded
seq_length = 20


class TemporalSpatialDataset:
    def __init__(self, X_spatial, Y_temporal, seq_length):
        self.X_spatial = X_spatial
        self.Y_temporal = Y_temporal
        self.seq_length = seq_length

    def __len__(self):
        return len(self.Y_temporal) - self.seq_length

    def __getitem__(self, idx):
        Y_seq = self.Y_temporal[idx: idx + self.seq_length]
        Y_target = self.Y_temporal[idx + self.seq_length]

        Y_seq = torch.tensor(np.array(Y_seq), dtype=torch.float32).unsqueeze(-1)
        Y_target = torch.tensor(np.array(Y_target), dtype=torch.float32).unsqueeze(-1)

        X_spatial = torch.tensor(np.array(self.X_spatial), dtype=torch.float32).permute(2, 0, 1)
        X_spatial = X_spatial.unsqueeze(1).repeat(1, self.seq_length, 1, 1)
        X_spatial = X_spatial.permute(0, 2, 3, 1)
        X_spatial = X_spatial.unsqueeze(0)
        X_spatial = X_spatial.permute(0, 1, 4, 2, 3)

        return X_spatial.squeeze(0), Y_seq, Y_target


dataset = TemporalSpatialDataset(X_original, Y_original, seq_length)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


class CNN3DModel(nn.Module):
    def __init__(self, input_channels, seq_length):
        super(CNN3DModel, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=input_channels, out_channels=32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.fc1 = nn.Linear(128 * (seq_length // 2 // 2 // 2) * (400 // 2 // 2 // 2) * (400 // 2 // 2 // 2), 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNN3DModel(input_channels=5, seq_length=seq_length)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (X_spatial, Y_seq, Y_target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(X_spatial)
        loss = criterion(output, Y_target)
        loss.backward()
        optimizer.step()
        print(loss.item())
