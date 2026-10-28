"""A small convolutional neural network (CNN) for classifying MNIST digits."""

import torch
from torch import nn


class DigitCNN(nn.Module):
    """Two convolutional blocks followed by two fully connected layers.

    Input:  (batch, 1, 28, 28) normalized grayscale images
    Output: (batch, 10) raw scores (logits), one per digit 0-9
    """

    def __init__(self):
        super().__init__()
        # Block 1: (1, 28, 28) -> conv -> (32, 28, 28) -> pool -> (32, 14, 14)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # Block 2: (32, 14, 14) -> conv -> (64, 14, 14) -> pool -> (64, 7, 7)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

        # Classifier: 64 * 7 * 7 = 3136 features -> 128 -> 10 digit scores
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)
