# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import json
import os
import sys
import time
from typing import List, Optional, Tuple, TypedDict
from torch import nn

import torch
import torch.nn.functional as F

from module import StateEncoder, BweHead, CNNEmbedding
from transformers import AutoModelForCausalLM, AutoConfig

Qwen1_5_weight_hf_format_dir = "/your_path_to_pretrain_file_in_huggingface/Qwen1.5/Qwen1.5-0.5B-hf"
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class bwe_agent(nn.Module):
    def __init__(self, max_action, state_dim, action_dim, device):
        super().__init__()
        self.max_action = max_action # Mbps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.float32).to(self.device))
        self.llm = AutoModelForCausalLM.from_pretrained(Qwen1_5_weight_hf_format_dir, use_cache=False, device_map=self.device)
        self.stateencoder = StateEncoder(self.state_dim, 1024, self.device)
        self.bwe_head = BweHead(1024, 512, self.action_dim, self.max_action, self.device)
        self.encoder0 = nn.Parameter(torch.tensor([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 
                                 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1
                                 ], dtype=torch.float32).to(self.device).detach())
        self.encoder0.requires_grad_(False)
    
    def forward(self, x):
        x = x * self.encoder0
        # x = self.cnnembedding(x)
        # x = self.llm.generate(x, max_length = 150)[0]
        # x = self.bwehead(x)
        x = self.stateencoder(x)
        x = torch.unsqueeze(x, dim=1) # ——>[bs, 1, 1024]
        x = self.llm.model(inputs_embeds = x)
        x = torch.squeeze(x.last_hidden_state)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        std = std.expand(x.shape[0], 1)
        # x = self.llm.model.model(x)
        # x = torch.squeeze(x.last_hidden_state[:, 149, :])
        x = self.bwe_head(x)
        return x, std

