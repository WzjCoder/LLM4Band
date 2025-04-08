import torch
from torch import nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, LoraModel
from estimator import bwe_agent

def main():
    # Configuration parameters
    max_action = 20
    state_dim = 150
    action_dim = 1
    device = 'cuda:1'
    torch_model_dir = '/home/wangzhijian/bandwidth_estimation/solution/GPT2_for_bwe/GPT2_smallest_Lora/checkpoints_iql/IQL-gpt2-estimator-range_average/checkpoint_1000000.pt'

    # Initialize the agent
    bwe_llm = bwe_agent(max_action, state_dim, action_dim, device)

    # Load the LoRA model weights
    print(f"Loading model weights...")
    bwe_llm.load_state_dict(torch.load(torch_model_dir))

    # Merge the LoRA weights into the base model
    print("Merging LoRA adapter weights into the base model...")
    bwe_llm.llm.merge_adapter()  # This merges LoRA weights into the base model

    base_model_dict = {}
    state_dict = bwe_llm.llm.base_model.state_dict()

    for key, value in state_dict.items():
        if "lora_A" in key or "lora_B" in key:
            continue
        new_key = key.replace(".base_layer", "")
        base_model_dict[new_key] = value

    save_dict = {
        "encoder0": bwe_llm.encoder0,
        "log_std": bwe_llm.log_std,
        "stateencoder": bwe_llm.stateencoder.state_dict(),  # 保存 stateencoder 的权重
        "llm": base_model_dict,  # 保存 llm 的 base_model 权重
        "bwe_head": bwe_llm.bwe_head.state_dict(),  # 保存 bwe_head 的权重
    }
    for key in save_dict["llm"].keys():
        print(key)

    save_path = './merged_model/gpt2_merged_model.pth'
    torch.save(save_dict, save_path)


if __name__ == "__main__":
    main()