import torch
from transformers import AutoModelForCausalLM
from estimator import bwe_agent, bwe_agent_base

def test_merged_model():
    """
    Test if LoRA weights are correctly merged into the base model.
    
    Args:
        original_model_path (str): Path to the original base model (without LoRA weights merged).
        merged_model_path (str): Path to the merged model (with LoRA weights merged).
        test_input_ids (torch.Tensor): Example input IDs to test the model output.
        
    Returns:
        None
    """
    # Load the original base model (before merging LoRA weights)
    device = 'cuda:0'
    torch_model_dir = '/home/wangzhijian/bandwidth_estimation/solution/T5_for_bwe/T5_LoRa/checkpoints_iql/IQL-T5-estimator-range_average/checkpoint_1000000.pt'
    merged_model_path = '/home/wangzhijian/bandwidth_estimation/solution/T5_for_bwe/T5_LoRa/merged_model/T5_merged_model.pth'
    test_input_ids = torch.rand((1,150))

    print("Loading original base model...")
    base_model = bwe_agent(20, 150, 1, device)
    base_model.load_state_dict(torch.load(torch_model_dir, map_location=device))
    base_model.llm.merge_adapter()
    
    # Load the merged model (after merging LoRA weights)
    print("Loading merged model...")
    save_dict = torch.load(merged_model_path)
    merged_model = bwe_agent_base(20, 150, 1, device)
    merged_model.stateencoder.load_state_dict(save_dict["stateencoder"])
    merged_model.llm.load_state_dict(save_dict["llm"])
    merged_model.bwe_head.load_state_dict(save_dict["bwe_head"])
    
    # Set both models to evaluation mode
    base_model.eval()
    merged_model.eval()
    test_input_ids = test_input_ids.to(device)
    
    # Perform inference on the same input
    print("Performing inference with both models...")
    with torch.no_grad():
        base_output, std1 = base_model(test_input_ids)
        merged_output, std2 = merged_model(test_input_ids)

    print(base_output)
    print(merged_output)



# Example usage
if __name__ == "__main__":
    # Run the test
    test_merged_model()