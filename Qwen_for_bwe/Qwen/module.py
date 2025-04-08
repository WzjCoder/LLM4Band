# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        # self.block = ResidualBlock(self.hidden_dim, self.hidden_dim)

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

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResidualBlock, self).__init__()
        self.output_size = output_size
        self.block = nn.Sequential(
            nn.Linear(input_size, self.output_size),
            nn.LayerNorm(self.output_size),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out

class CustomParallelEmbedding(nn.Module):
    def __init__(self, input_size, output_size, world_size):
        super(CustomParallelEmbedding, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.world_size = world_size

        self.embedding_size_per_rank = output_size // world_size
        self.local_rank = dist.get_rank()

        # Each rank holds a fraction of the embedding matrix
        self.embedding_matrix = nn.Parameter(torch.randn(input_size, self.embedding_size_per_rank))

    def forward(self, input_indices):
        # 广播嵌入矩阵到所有的进程
        dist.broadcast(self.embedding_matrix, src=0)

        # 创建一个空列表来保存所有进程中的输入索引
        all_input_indices = [torch.zeros(0, dtype=input_indices.dtype, device=input_indices.device) for _ in range(self.world_size)]

        # 从所有进程中收集输入索引（非阻塞）
        gather_op = dist.all_gather(all_input_indices, input_indices)

        # 等待收集操作完成
        torch.distributed.barrier()

        # 从收集到的列表中选择本地进程的输入索引
        local_input_indices = all_input_indices[self.local_rank]

        # 选择本地进程的相应嵌入向量
        local_embeddings = torch.index_select(self.embedding_matrix, dim=0, index=local_input_indices)

        return local_embeddings

# https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/model_parallel/layers.py
