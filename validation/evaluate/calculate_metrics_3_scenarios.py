import torch
import json
import os
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from utils import *
import glob
import sys
import datetime
from estimator import bwe_agent, bwe_agent_lora, bwe_agent_GPT2, bwe_agent_GPT2_lora, bwe_agent_T5, bwe_agent_T5_lora

device = 'cuda:1'

def get_overestimation_rate(x, y):
    l = [max((xx - yy) / yy, np.float32(0)) for xx, yy in zip(x.flat, y.flat)]
    # l = [np.float32(item) for item in l]
    l = np.asarray(l, dtype=np.float32)
    return np.nanmean(l)

def get_mse(x, y):
    l = [(xx - yy) ** 2 for xx, yy in zip(x, y)]
    l = np.asarray(l, dtype=np.float32)
    return np.nanmean(l)

def get_error_rate(x, y):
    # error rate = min(1, |x-y| / y)
    l = [min(1, np.float32(abs(xx - yy) / yy)) for xx, yy in zip(x.flat, y.flat)]
    l = np.asarray(l, dtype=np.float32)
    return np.nanmean(l)

def get_underestimation_rate(x, y):
    # error rate = max(0, (y - x) / y)
    l = [max(np.float32(0), (yy - xx) / yy) for xx, yy in zip(x.flat, y.flat)]
    l = np.asarray(l, dtype=np.float32)
    return np.nanmean(l)

def calculate_metrics(model_names, models, data_files):

    model_mse = {}
    model_or = {} # overestimation error rate
    model_ur = {} # underestimation error rate
    model_er = {} # error rate
    for name in model_names:
        model_mse[name] = []
        model_or[name] = []
        model_ur[name] = []
        model_er[name] = []

    for filename in tqdm(data_files, desc="Processing"):
        with open(filename, "r") as file:
            call_data = json.load(file)

        observations = np.asarray(call_data['observations'], dtype=np.float32)
        true_capacity = np.asarray(call_data['true_capacity'], dtype=np.float32) / 1e6

        predictions = {}
        for name in model_names:
            predictions[name] = []
        
        for t in range(observations.shape[0]):
            obs = torch.tensor(observations[t:t+1,:]).to(device)

            for idx, model in enumerate(models):
                with torch.no_grad():
                    llm_prediction, std = model(obs)
                    
                    predictions[model_names[idx]].append(llm_prediction)
        
        for name in model_names:
            # outputs = np.asarray(predictions[name], dtype=np.float32) / 1e6
            outputs = [tensor.cpu().numpy() for tensor in predictions[name]]
            outputs = np.asarray(outputs, dtype=np.float32) / 1e6

            model_mse[name].append(get_mse(outputs, true_capacity))
            model_or[name].append(get_overestimation_rate(outputs, true_capacity))
            model_er[name].append(get_error_rate(outputs, true_capacity))
            model_ur[name].append(get_underestimation_rate(outputs, true_capacity))
    return model_mse, model_or, model_ur, model_er

def calculate_metrics_onnx(model_names, models, data_files):

    model_mse = {}
    model_or = {} # overestimation error rate
    model_ur = {} # underestimation error rate
    model_er = {} # error rate
    for name in model_names:
        model_mse[name] = []
        model_or[name] = []
        model_ur[name] = []
        model_er[name] = []

    for filename in tqdm(data_files, desc="Processing"):
        with open(filename, "r") as file:
            call_data = json.load(file)

        observations = np.asarray(call_data['observations'], dtype=np.float32)
        true_capacity = np.asarray(call_data['true_capacity'], dtype=np.float32) / 1e6

        predictions = {}
        for name in model_names:
            predictions[name] = []
        hidden_state, cell_state = np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)        
        for t in range(observations.shape[0]):
            obss = observations[t:t+1,:].reshape(1,1,-1)
            feed_dict = {'obs': obss,
                        'hidden_states': hidden_state,
                        'cell_states': cell_state
                        }
            for idx, orts in enumerate(models):
                bw_prediction, hidden_state, cell_state = orts.run(None, feed_dict)
                predictions[model_names[idx]].append(bw_prediction[0,0,0])                    
        
        for name in model_names:
            outputs = np.asarray(predictions[name], dtype=np.float32) / 1e6
            model_mse[name].append(get_mse(outputs, true_capacity))
            model_or[name].append(get_overestimation_rate(outputs, true_capacity))
            model_er[name].append(get_error_rate(outputs, true_capacity))
            model_ur[name].append(get_underestimation_rate(outputs, true_capacity))
    return model_mse, model_or, model_ur, model_er

if __name__ == "__main__":

    # data dir
    high_bw_call_dir = "/your_path/validation/prepare_scenario/high_bw_call"
    low_bw_call_dir = "/your_path/validation/prepare_scenario/low_bw_call"
    high_variance_call_dir = "/your_path/validation/prepare_scenario/high_fluctuating_bw"
    low_variance_call_dir = "/your_path/validation/prepare_scenario/low_fluctuating_bw"
    # model path
    onnx_model_path = 'your_path/Schaferct/onnx_model_for_evaluation/baseline.onnx'
    schaferct_retrain_onnx_model_path = "your_path/Schaferct/code/checkpoints_iql/IQL-range_average/checkpoint_1000000.onnx"
    pioneer_onnx_model_path = 'your_path/Pioneer/onnx_model/SJTU_Medialab_final.onnx'    
    
    qwen_0_5B_range_average_model_path = '/your_path/Qwen_for_bwe/Qwen/checkpoints_iql/IQL-qwen0.5B-estimator-range_average/checkpoint_1000000.pt'
    qwen_0_5B_lora_range_average_model_path = '/your_path/Qwen_for_bwe/Qwen_LoRa/checkpoints_iql/IQL-qwen0.5B-estimator-range_average/checkpoint_1000000.pt'

    gpt2_smallest_model_range_average_path = '/your_path/GPT2_for_bwe/GPT2_smallest/checkpoints_iql/IQL-gpt2-estimator-range_average-e0dbdab4/checkpoint_1000000.pt'
    T5_model_range_average_path = '/your_path/T5_for_bwe/T5/checkpoints_iql/IQL-T5-estimator-range_average-e763176a/checkpoint_1000000.pt'

    gpt2_smallest_lora_range_average_path = '/your_path/GPT2_for_bwe/GPT2_smallest_Lora/checkpoints_iql/IQL-gpt2-estimator-range_average/checkpoint_1000000.pt'
    T5_lora_range_average_path = '/your_path/T5_for_bwe/T5_LoRa/checkpoints_iql/IQL-T5-estimator-range_average/checkpoint_1000000.pt'

    # create model

    baseline_model = ort.InferenceSession(onnx_model_path)
    # pioneer_model = ort.InferenceSession(pioneer_onnx_model_path)

    # schaferct_range_average_model = ort.InferenceSession(schaferct_retrain_onnx_model_path)

    # qwen_0_5B = bwe_agent(20, 150, 1, 1024, '0.5B', device)
    # qwen_0_5B.load_state_dict(torch.load(qwen_0_5B_range_average_model_path, map_location=device))

    # qwen_0_5B_lora = bwe_agent_lora(20, 150, 1, 1024, '0.5B', device)
    # qwen_0_5B_lora.load_state_dict(torch.load(qwen_0_5B_lora_range_average_model_path, map_location=device))

    # onnx_model_names = ['baseline', 'schaferct_retrain']
    # onnx_models = [baseline_model, schaferct_retrain_model]

    # GPT2_smallest_lora = bwe_agent_GPT2_lora(20, 150, 1, device)
    # GPT2_smallest_lora.load_state_dict(torch.load(gpt2_smallest_lora_range_average_path))

    # T5_lora = bwe_agent_T5_lora(20, 150, 1, device)
    # T5_lora.load_state_dict(torch.load(T5_lora_range_average_path, map_location=device))

    # GPT2_smallest = bwe_agent_GPT2(20, 150, 1, device)
    # GPT2_smallest.load_state_dict(torch.load(gpt2_smallest_model_range_average_path, map_location=device))

    # T5 = bwe_agent_T5(20, 150, 1, device)
    # T5.load_state_dict(torch.load(T5_model_range_average_path, map_location=device))

    model_names = []
    models = []
    
    onnx_model_names = ['baseline']
    onnx_models = [baseline_model]

    high_bw_call = glob.glob(os.path.join(high_bw_call_dir, f'*.json'), recursive=True)
    low_bw_call = glob.glob(os.path.join(low_bw_call_dir, f'*.json'), recursive=True)
    fluctuating_call = glob.glob(os.path.join(high_variance_call_dir, f'*.json'), recursive=True) + glob.glob(os.path.join(low_variance_call_dir, f'*.json'), recursive=True)

    scenarios = [high_bw_call, low_bw_call, fluctuating_call]
    scenarios_names = ['high_bw_call', 'low_bw_call', 'fluctuating_call']

    output_file = "/your_path/validation/evaluate/result.txt"

    with open(output_file, 'w') as f:
        sys.stdout = f

        for idx, data_files in enumerate(scenarios):
            
            print(f'----------------{scenarios_names[idx]}----------------')

            current_datetime = datetime.datetime.now()
            print("start time:" + current_datetime.strftime("%Y-%m-%d %H:%M:%S"))

            # onnx models metrics

            model_mse, model_or, model_ur, model_er = calculate_metrics_onnx(onnx_model_names, onnx_models, data_files)
            for name in onnx_model_names:
                print(f'{name} results:')
                print(f'mse:{sum(model_mse[name]) / len(model_mse[name])}')
                print(f'over_rate:{sum(model_or[name]) / len(model_or[name])}')
                print(f'under_rate:{sum(model_ur[name]) / len(model_ur[name])}')
                print(f'error_rate:{sum(model_er[name]) / len(model_er[name])}')

            # torch models metrics

            # model_mse, model_or, model_ur, model_er = calculate_metrics(model_names, models, data_files)
            # for name in model_names:
            #     print(f'{name} results:')
            #     print(f'mse:{sum(model_mse[name]) / len(model_mse[name])}')
            #     print(f'over_rate:{sum(model_or[name]) / len(model_or[name])}')
            #     print(f'under_rate:{sum(model_ur[name]) / len(model_ur[name])}')
            #     print(f'error_rate:{sum(model_er[name]) / len(model_er[name])}')

            current_datetime = datetime.datetime.now()
            print("end time: " + current_datetime.strftime("%Y-%m-%d %H:%M:%S"))
    
    sys.stdout = sys.__stdout__

    



