import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Santorni(nn.Module):
    def __init__(self, device):
        super(Santorni, self).__init__()
        self.device = device
        self.size = (5,5)
        self.fc1 = nn.Linear(in_features=self.size, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=64)
        self.value_head = nn.Linear(in_features=64, out_features=1)
        self.to(device)

    def forward(self, x):
        """
        Feed forward into the Neural Network
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value_logit = self.value_head(x)

        return torch.tanh(value_logit)

    def predict(self, board):
        """
        Predict the value of a state
        """
        board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        board = board.view(1, self.size)
        self.eval()
        with torch.no_grad():
            v = self.forward(board)

        return v.data.cpu().numpy()[0]