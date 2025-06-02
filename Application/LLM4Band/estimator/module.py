import torch
import torch.nn as nn

class StateEncoder(nn.Module):
    def __init__(self, observation_shape, hidden_dim, device):
        super(StateEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.fc_head = nn.Linear(observation_shape, self.hidden_dim).to(self.device)
        self.layer_norm = nn.LayerNorm(self.hidden_dim).to(self.device)
        self.ac1 = nn.LeakyReLU().to(self.device)

    def forward(self, x):
        x = self.fc_head(x)
        x = self.layer_norm(x)
        x = self.ac1(x)
        # x = self.block(x)
        return x

class CNNEmbedding(nn.Module):
    def __init__(self, embedding_dim, device):
        super(CNNEmbedding, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.conv1d = nn.Conv1d(1, self.embedding_dim, 1).to(self.device)
        self.layer_norm = nn.LayerNorm(self.embedding_dim).to(self.device)
        self.ac1 = nn.ReLU().to(self.device)

    def forward(self, x):
        x = torch.unsqueeze(x, dim = 1)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        x = self.ac1(x)
        return x


class BweHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_action, device):
        super(BweHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_action = max_action
        self.device = device
        self.fc_head1 = nn.Linear(input_dim, self.hidden_dim).to(self.device)
        self.layer_norm = nn.LayerNorm(self.hidden_dim).to(self.device)
        self.ac1 = nn.LeakyReLU().to(self.device)

        self.fc_head2 = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)
        self.ac2 = nn.LeakyReLU().to(self.device)
        self.tanh = nn.Tanh().to(self.device)

    def forward(self, x):
        x = self.fc_head1(x)
        x = self.layer_norm(x)
        x = self.ac1(x)
        x = self.fc_head2(x)
        x = self.ac2(x)
        return self.tanh(x) * self.max_action * 1e6

class BweHead2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_action, device):
        super(BweHead2, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_action = max_action
        self.device = device
        self.fc_head1 = nn.Linear(input_dim, self.hidden_dim).to(self.device)
        self.layer_norm = nn.LayerNorm(self.hidden_dim).to(self.device)
        self.ac1 = nn.LeakyReLU().to(self.device)
        self.block = ResidualBlock(self.hidden_dim, self.hidden_dim, self.device)
        self.fc_head2 = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)
        self.ac2 = nn.LeakyReLU().to(self.device)
        self.tanh = nn.Tanh().to(self.device)

    def forward(self, x):
        x = self.fc_head1(x)
        x = self.layer_norm(x)
        x = self.ac1(x)
        x = self.block(x)
        x = self.fc_head2(x)
        x = self.ac2(x)
        return self.tanh(x) * self.max_action * 1e6

class BweHead_naorl(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_action, device):
        super(BweHead_naorl, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_action = max_action
        self.device = device
        self.fc_head1 = nn.Linear(input_dim, self.hidden_dim).to(self.device)
        self.layer_norm = nn.LayerNorm(self.hidden_dim).to(self.device)
        self.ac1 = nn.LeakyReLU().to(self.device)

        self.fc_head2 = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)
        self.ac2 = nn.LeakyReLU().to(self.device)
        self.tanh = nn.Tanh().to(self.device)

    def forward(self, x):
        x = self.fc_head1(x)
        x = self.layer_norm(x)
        x = self.ac1(x)
        x = self.fc_head2(x)
        x = self.ac2(x)
        return self.tanh(x) 

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(ResidualBlock, self).__init__()
        self.output_size = output_size
        self.block = nn.Sequential(
            nn.Linear(input_size, self.output_size).to(device),
            nn.LayerNorm(self.output_size).to(device),
            nn.LeakyReLU().to(device),
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out
