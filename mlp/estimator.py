import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class bwe_agent(nn.Module):
    def __init__(self, max_action, state_dim, action_dim, device):
        super(bwe_agent, self).__init__()
        self.max_action = max_action
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.float32).to(self.device))

        self.fc1 = nn.Linear(self.state_dim, 256).to(self.device)  
        self.ac1 = nn.LeakyReLU().to(self.device)
        self.fc2 = nn.Linear(256, 256).to(self.device)  
        self.ac2 = nn.LeakyReLU().to(self.device)        
        self.fc3 = nn.Linear(256, self.action_dim).to(self.device) 
        self.ac3 = nn.LeakyReLU().to(self.device)  
        self.tanh = nn.Tanh().to(self.device)

    def forward(self, state):
        x = self.ac1(self.fc1(state))  
        x = self.ac2(self.fc2(x))      
        x = self.ac3(self.fc3(x))     
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        std = std.expand(x.shape[0], 1)        
        return self.tanh(x) * self.max_action * 1e6, std
