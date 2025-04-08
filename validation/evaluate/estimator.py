from torch import nn
import torch
import torch.nn.functional as F

from module import StateEncoder, BweHead
from transformers import AutoModelForCausalLM, AutoModel
from peft import LoraConfig, LoraModel

Qwen1_5_weight_hf_format_dir = "/your_path_to_pretrain_file_in_huggingface/Qwen1.5/Qwen1.5-0.5B-hf"
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

Lora_config = LoraConfig(
    init_lora_weights = "gaussian",
    r=16,
    lora_alpha=32,
    target_modules="all-linear", 
    lora_dropout=0.01,
)

def get_Qwen1_5_weight_hf_format_dir(type):
    return '/your_path_to_pretrain_file_in_huggingface/Qwen1.5/Qwen1.5-' + type + '-hf'

class bwe_agent(nn.Module):
    def __init__(self, max_action, state_dim, action_dim, hidden_dim, type, device):
        super().__init__()
        self.max_action = max_action # Mbps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.hidden_dim = hidden_dim
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.float32).to(self.device))
        self.llm = AutoModelForCausalLM.from_pretrained(get_Qwen1_5_weight_hf_format_dir(type), use_cache=False, device_map=self.device)
        self.stateencoder = StateEncoder(self.state_dim, self.hidden_dim, self.device)
        self.bwe_head = BweHead(self.hidden_dim, 512, self.action_dim, self.max_action, self.device)
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
                                 ], dtype=torch.float32).to(self.device).detach()).to(self.device)
        self.encoder0.requires_grad_(False)
    
    def forward(self, x):
        x = x * self.encoder0
        x = self.stateencoder(x)
        x = torch.unsqueeze(x, dim=1) # ——>[bs, 1, 1024]
        x = self.llm.model(inputs_embeds = x)
        x = torch.squeeze(x.last_hidden_state)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        std = std.expand(x.shape[0], 1)
        x = self.bwe_head(x)
        return x, std

Lora_config_qwen = LoraConfig(
    init_lora_weights = "gaussian",
    r=16,
    lora_alpha=32,
    target_modules="all-linear", 
    lora_dropout=0.01,
)

class bwe_agent_lora(nn.Module):
    def __init__(self, max_action, state_dim, action_dim, hidden_dim, type, device):
        super().__init__()
        self.max_action = max_action # Mbps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.hidden_dim = hidden_dim
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.float32).to(self.device))
        self.llm = LoraModel(AutoModelForCausalLM.from_pretrained(get_Qwen1_5_weight_hf_format_dir(type), use_cache=False, device_map=self.device), 
                            Lora_config_qwen, 
                            "default")
        self.stateencoder = StateEncoder(self.state_dim, self.hidden_dim, self.device)
        self.bwe_head = BweHead(self.hidden_dim, 512, self.action_dim, self.max_action, self.device)
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
        x = torch.unsqueeze(x, dim=1) # ——>[bs, 1, 1024]
        x = self.llm.model.model(inputs_embeds = x)
        x = torch.squeeze(x.last_hidden_state)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        std = std.expand(x.shape[0], 1)
        x = self.bwe_head(x)
        return x, std


GPT2_smallest_weight_hf_format_dir = "/your_path_to_pretrain_file_in_huggingface/GPT2/gpt2_smallest_hf"
Lora_config_GPT = LoraConfig(
    init_lora_weights = "gaussian",
    r=16,
    lora_alpha=32,
    target_modules="all-linear", 
    lora_dropout=0.01,
)

class bwe_agent_GPT2(nn.Module):
    def __init__(self, max_action, state_dim, action_dim, device):
        super().__init__()
        self.max_action = max_action # Mbps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.float32).to(self.device))
        self.llm = AutoModelForCausalLM.from_pretrained(GPT2_smallest_weight_hf_format_dir, use_cache=False, device_map=self.device)
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
        x = self.llm.transformer(inputs_embeds = x)
        x = torch.squeeze(x.last_hidden_state)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        std = std.expand(x.shape[0], 1)
        x = self.bwe_head(x)
        return x, std

class bwe_agent_GPT2_lora(nn.Module):
    def __init__(self, max_action, state_dim, action_dim, device):
        super().__init__()
        self.max_action = max_action # Mbps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.float32).to(self.device))
        self.llm = LoraModel(AutoModelForCausalLM.from_pretrained(GPT2_smallest_weight_hf_format_dir, use_cache=False, device_map=self.device), 
                            Lora_config_GPT, 
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
        x = self.llm.model.transformer(inputs_embeds = x)
        x = torch.squeeze(x.last_hidden_state)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        std = std.expand(x.shape[0], 1)
        x = self.bwe_head(x)
        return x, std

T5_weight_hf_format_dir = "/your_path_to_pretrain_file_in_huggingface/T5/T5_hf"
Lora_config_T5 = LoraConfig(
    init_lora_weights = "gaussian",
    r=16,
    lora_alpha=32,
    target_modules="all-linear", 
    lora_dropout=0.01,
)

class bwe_agent_T5(nn.Module):
    def __init__(self, max_action, state_dim, action_dim, device):
        super().__init__()
        self.max_action = max_action # Mbps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.float32).to(self.device))
        self.llm = AutoModel.from_pretrained(T5_weight_hf_format_dir, use_cache=False, device_map=self.device)
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

class bwe_agent_T5_lora(nn.Module):
    def __init__(self, max_action, state_dim, action_dim, device):
        super().__init__()
        self.max_action = max_action # Mbps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.float32).to(self.device))
        self.llm = LoraModel(AutoModel.from_pretrained(T5_weight_hf_format_dir, use_cache=False, device_map=self.device), 
                            Lora_config_T5, 
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


class mlp(nn.Module):
    def __init__(self, max_action, state_dim, action_dim, device):
        super(mlp, self).__init__()
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


class bwe_agent_Qwen_base(nn.Module):
    def __init__(self, max_action, state_dim, action_dim, device):
        super().__init__()
        self.max_action = max_action # Mbps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.float32).to(self.device))
        LLM = AutoModelForCausalLM.from_pretrained(Qwen1_5_weight_hf_format_dir, use_cache=False, device_map=self.device)
        self.llm = LLM.base_model
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
        x = self.stateencoder(x)
        x = torch.unsqueeze(x, dim=1) # ——>[bs, 1, 1024]
        x = self.llm(inputs_embeds = x)
        x = torch.squeeze(x.last_hidden_state)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        std = std.expand(x.shape[0], 1)
        x = self.bwe_head(x)
        return x, std

class bwe_agent_GPT2_base(nn.Module):
    def __init__(self, max_action, state_dim, action_dim, device):
        super().__init__()
        self.max_action = max_action # Mbps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.float32).to(self.device))
        LLM = AutoModelForCausalLM.from_pretrained(GPT2_smallest_weight_hf_format_dir, use_cache=False, device_map=self.device)
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
        x = self.llm(inputs_embeds = x)
        x = torch.squeeze(x.last_hidden_state)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        std = std.expand(x.shape[0], 1)
        x = self.bwe_head(x)
        return x, std

class bwe_agent_T5_base(nn.Module):
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