import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel

Qwen1_5_weight_hf_format_dir = "/home/wangzhijian/bandwidth_estimation/Qwen1.5/Qwen1.5-0.5B-hf"

tokenizer = AutoTokenizer.from_pretrained(Qwen1_5_weight_hf_format_dir)
model = AutoModelForCausalLM.from_pretrained(Qwen1_5_weight_hf_format_dir, use_cache=False, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token

prompt = "Tell me about gravity"
print(tokenizer.decode(model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device), max_length=300)[0]))

# 选择模型和配置，并下载到指定目录
# model_name = "Qwen/Qwen1.5-0.5B"
# output_dir = "/home/wangzhijian/bandwidth_estimation/Qwen1.5/Qwen1.5-0.5B-hf"
# config = AutoConfig.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# # 保存模型和配置
# config.save_pretrained("模型保存路径")
# model.save_pretrained("模型保存路径")

