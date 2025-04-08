import torch
import time
import json
import os
import numpy as np
import onnxruntime as ort
import glob
from tqdm import tqdm

from estimator import bwe_agent, bwe_agent_lora, bwe_agent_GPT2, bwe_agent_GPT2_lora, bwe_agent_T5, bwe_agent_T5_lora, bwe_agent_Qwen_base, bwe_agent_GPT2_base, bwe_agent_T5_base

device = 'cuda:0'
device_id = 0

def calculate_time_file_onnx(data_path, onnx_model):

    with open(data_path, "r") as file:
        call_data = json.load(file)

    observations = np.asarray(call_data['observations'], dtype=np.float32)
    hidden_state, cell_state = np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)

    total_time = 0.0
    num_runs = 0

    for t in range(observations.shape[0]):
        obss = observations[t:t+1,:].reshape(1,1,-1)
        feed_dict = {'obs': obss,
                    'hidden_states': hidden_state,
                    'cell_states': cell_state
                    }
        start_time = time.time()
        bw_prediction, hidden_state, cell_state = onnx_model.run(None, feed_dict)
        end_time = time.time()
        total_time += (end_time - start_time)
        num_runs += 1

    average_inference_time = total_time / num_runs
    return average_inference_time

def calculate_time_file_torch(data_path, qwen_model):

    with open(data_path, "r") as file:
        call_data = json.load(file)

    observations = np.asarray(call_data['observations'], dtype=np.float32)

    total_time = 0.0
    num_runs = 0

    for t in range(observations.shape[0]):
        obs = torch.tensor(observations[t:t+1,:]).to(qwen_model.device)
        start_time = time.time()
        with torch.no_grad():
            llm_prediction, std = qwen_model(obs)
        torch.cuda.synchronize()
        end_time = time.time()
        total_time += (end_time - start_time)
        num_runs += 1

    average_inference_time = total_time / num_runs
    return average_inference_time

def calculate_time_dir_torch(data_files, qwen_model):
    qwen_model.eval()
    qwen_model = qwen_model.cuda(device_id)

    total_time = 0.0
    num_runs = 0

    for filename in tqdm(data_files, desc="Processing"):
        total_time += calculate_time_file_torch(filename, qwen_model)
        num_runs += 1

    average_inference_time = total_time / num_runs
    print(f'平均推理时间: {average_inference_time:.6f} 秒')

def calculate_time_dir_onnx(data_files, onnx_model):

    total_time = 0.0
    num_runs = 0

    for filename in tqdm(data_files, desc="Processing"):
        total_time += calculate_time_file_onnx(filename, onnx_model)
        num_runs += 1

    average_inference_time = total_time / num_runs
    print(f'平均推理时间: {average_inference_time:.6f} 秒')

if __name__ == "__main__":

    data_dir = "/your_path/validation/prepare_scenario/high_bw_call"
    onnx_model_path = '/your_path/Schaferct/onnx_model_for_evaluation/baseline.onnx'
    schaferct_range_average_onnx_model_path = "/your_path/Schaferct/code/checkpoints_iql/IQL-range_average/checkpoint_1000000.onnx"

    qwen_0_5B_range_average_model_path = '/your_path/Qwen_for_bwe/Qwen/checkpoints_iql/IQL-qwen0.5B-estimator-range_average/checkpoint_1000000.pt'
    qwen_0_5B_lora_range_average_model_path = '/your_path/Qwen_for_bwe/Qwen_LoRa/checkpoints_iql/IQL-qwen0.5B-estimator-range_average/checkpoint_1000000.pt'
    gpt2_smallest_model_range_average_path = '/your_path/GPT2_for_bwe/GPT2_smallest/checkpoints_iql/IQL-gpt2-estimator-range_average-e0dbdab4/checkpoint_1000000.pt'
    gpt2_smallest_lora_model_range_average_path = '/your_path/GPT2_for_bwe/GPT2_smallest_Lora/checkpoints_iql/IQL-gpt2-estimator-range_average/checkpoint_1000000.pt'
    T5_model_range_average_path = '/your_path/T5_for_bwe/T5/checkpoints_iql/IQL-T5-estimator-range_average-e763176a/checkpoint_1000000.pt'
    T5_model_lora_range_average_path = '/your_path/T5_for_bwe/T5_LoRa/checkpoints_iql/IQL-T5-estimator-range_average/checkpoint_1000000.pt'


    # baseline_model = ort.InferenceSession(onnx_model_path)

    # schaferct_range_average_model = ort.InferenceSession(schaferct_range_average_onnx_model_path)

    # GPT2_smallest_lora = bwe_agent_GPT2_lora(20, 150, 1, device)
    # GPT2_smallest_lora.load_state_dict(torch.load(gpt2_smallest_lora_model_range_average_path, map_location=device))
    # GPT2_smallest_lora.llm.merge_adapter("merge_and_unload")

    # GPT2_smallest = bwe_agent_GPT2(20, 150, 1, device)
    # GPT2_smallest.load_state_dict(torch.load(gpt2_smallest_model_range_average_path, map_location=device))

    # T5_lora = bwe_agent_T5_lora(20, 150, 1, device)
    # T5_lora.load_state_dict(torch.load(T5_model_lora_range_average_path, map_location=device))

    # T5 = bwe_agent_T5(20, 150, 1, device)
    # T5.load_state_dict(torch.load(T5_model_range_average_path, map_location=device))

    # qwen_0_5B = bwe_agent(20, 150, 1, 1024, '0.5B', device)
    # qwen_0_5B.load_state_dict(torch.load(qwen_0_5B_range_average_model_path, map_location=device))

    # qwen_0_5B_lora = bwe_agent_lora(20, 150, 1, 1024, '0.5B', device)
    # qwen_0_5B_lora.load_state_dict(torch.load(qwen_0_5B_lora_range_average_model_path, map_location=device))
    # qwen_0_5B_lora.llm.merge_adapter("merge_and_unload")

    GPT2_lora_merged_model_path = '/your_path/GPT2_for_bwe/GPT2_smallest_Lora/merged_model/gpt2_merged_model.pth'
    T5_lora_merged_model_path = '/your_path/T5_for_bwe/T5_LoRa/merged_model/T5_merged_model.pth'
    qwen_0_5B_lora_merged_model_path = '/your_path/Qwen_for_bwe/Qwen_LoRa/merged_model/Qwen_merged_model.pth'

    save_dict = torch.load(T5_lora_merged_model_path)
    merged_model = bwe_agent_T5_base(20, 150, 1, device)
    merged_model.stateencoder.load_state_dict(save_dict["stateencoder"])
    merged_model.llm.load_state_dict(save_dict["llm"])
    merged_model.bwe_head.load_state_dict(save_dict["bwe_head"])

    data_files = glob.glob(os.path.join(data_dir, f'*.json'), recursive=True)

    # calculate_time_dir_onnx(data_files, baseline_model)
    calculate_time_dir_torch(data_files, merged_model)


