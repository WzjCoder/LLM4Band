# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from torch import nn

import torch
import torch.nn.functional as F

from module import StateEncoder, BweHead
from transformers import AutoModelForCausalLM, AutoModel
from peft import LoraConfig, LoraModel

T5_weight_hf_format_dir = "/home/wangzhijian/bandwidth_estimation/T5/T5_hf"
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

Lora_config = LoraConfig(
    init_lora_weights = "gaussian",
    r=16,
    lora_alpha=32,
    target_modules="all-linear", 
    lora_dropout=0.01,
)

class bwe_agent(nn.Module):
    def __init__(self, max_action, state_dim, action_dim, device):
        super().__init__()
        self.max_action = max_action # Mbps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.float32).to(self.device))
        self.llm = LoraModel(AutoModel.from_pretrained(T5_weight_hf_format_dir, use_cache=False, device_map=self.device), 
                            Lora_config, 
                            "default")
        self.stateencoder = StateEncoder(self.state_dim, 768, self.device)
        self.bwe_head = BweHead(768, 512, self.action_dim, self.max_action, self.device)
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
        x = self.stateencoder(x)
        x = torch.unsqueeze(x, dim=1) # ——>[bs, 1, 768]
        x = self.llm.model.decoder(inputs_embeds = x)
        x = torch.squeeze(x.last_hidden_state)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        std = std.expand(x.shape[0], 1)
        x = self.bwe_head(x)
        return x, std


class bwe_agent_base(nn.Module):
    def __init__(self, max_action, state_dim, action_dim, device):
        super().__init__()
        self.max_action = max_action # Mbps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.float32).to(self.device))
        LLM = AutoModel.from_pretrained(T5_weight_hf_format_dir, use_cache=False, device_map=self.device)
        self.llm = LLM.base_model
        self.stateencoder = StateEncoder(self.state_dim, 768, self.device)
        self.bwe_head = BweHead(768, 512, self.action_dim, self.max_action, self.device)
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
        x = self.stateencoder(x)
        x = torch.unsqueeze(x, dim=1) # ——>[bs, 1, 768]
        x = self.llm.decoder(inputs_embeds = x)
        x = torch.squeeze(x.last_hidden_state)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        std = std.expand(x.shape[0], 1)
        x = self.bwe_head(x)
        return x, std

