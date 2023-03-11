import torch
import torch.nn as nn
import torch.nn.functional as F

# class QNetwork(nn.Module):
#     """Actor (Policy) Model."""

#     def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_units (int): Number of nodes in first hidden layer
#             fc2_units (int): Number of nodes in second hidden layer
#         """
#         super(QNetwork, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, action_size)

#     def forward(self, state):
#         """Build a network that maps state -> action values."""
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)
    
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.bn1 = nn.LayerNorm(state_size)
        self.lstm1 = nn.LSTM(state_size, 64)
        
        self.bn2 = nn.LayerNorm(64)
        self.fc1 = nn.Linear(64, 128)
        
        self.drop1 = nn.Dropout(0.05)
        
        self.bn3 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)
        
        self.drop2 = nn.Dropout(0.1)
        
        self.bn4 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 32)
        
        self.bn5 = nn.LayerNorm(32)
        self.lstm2 = nn.LSTM(32, action_size)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        
        x = self.bn1(x)
        x, y = self.lstm1(x)
        
        x = self.bn2(x)
        x = F.relu(self.fc1(x))
        
        x = self.drop1(x)
        
        x = self.bn3(x)
        x = F.relu(self.fc2(x))
        
        x = self.drop2(x)
        
        x = self.bn4(x)
        x = F.relu(self.fc3(x))
        
        x = self.bn5(x)
        x, y = self.lstm2(x)
        
        actions = x
        return actions
