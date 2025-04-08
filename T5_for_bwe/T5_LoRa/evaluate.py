import glob
import json
import os
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
import matplotlib.pyplot as plt
import torch

from estimator import bwe_agent

current_dir = os.path.split(os.path.abspath(__file__))[0]
project_root_path = current_dir.rsplit('/', 1)[0]
device = 'cuda:0'

plt.rcParams.clear()
plt.rcParams['font.size'] = 10
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.labelsize'] = 17
plt.rcParams['ytick.labelsize'] = 17
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


if __name__ == "__main__":

    data_dir = "/home/wangzhijian/bandwidth_estimation/offline-RL-congestion-control/data" 
    # data_dir = "/home/wangzhijian/bandwidth_estimation/solution/validation/prepare_scenario/high_bw_call"
    # data_dir = "/home/wangzhijian/bandwidth_estimation/solution/validation/prepare_scenario/high_fluctuating_bw"
    onnx_models = ['baseline']
    onnx_models_dir = '/home/wangzhijian/bandwidth_estimation/Schaferct/onnx_model_for_evaluation'
    torch_model = 'Qwen_bwe'
    torch_model_dir = '/home/wangzhijian/bandwidth_estimation/solution/Qwen_for_bwe/Qwen_LoRa/checkpoints_iql/IQL-llama-estimator-1.0/checkpoint_1000000.pt'
    figs_dir = os.path.join(project_root_path, 'evaluation_figs')
    if not os.path.exists(figs_dir):
        os.mkdir(figs_dir)
    data_files = glob.glob(os.path.join(data_dir, f'*.json'), recursive=True)
    ort_sessions = []
    for m in onnx_models:
        m_path = os.path.join(onnx_models_dir, m + '.onnx')
        ort_sessions.append(ort.InferenceSession(m_path))

    bwe_llm = bwe_agent(20, 150, 1, device)
    bwe_llm.load_state_dict(torch.load(torch_model_dir))
    bwe_llm.llm.merge_adapter()
    print("torch model load successfully!")
    for filename in tqdm(data_files, desc="Processing"):
        with open(filename, "r") as file:
            call_data = json.load(file)

        observations = np.asarray(call_data['observations'], dtype=np.float32)
        bandwidth_predictions = np.asarray(call_data['bandwidth_predictions'], dtype=np.float32)
        true_capacity = np.asarray(call_data['true_capacity'], dtype=np.float32)

        baseline_model_predictions = {}
        for m in onnx_models:
            baseline_model_predictions[m] = []
        baseline_model_predictions[torch_model] = []
        hidden_state, cell_state = np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)    
        for t in range(observations.shape[0]):
            obss = observations[t:t+1,:].reshape(1,1,-1)
            feed_dict = {'obs': obss,
                        'hidden_states': hidden_state,
                        'cell_states': cell_state
                        }
            for idx, orts in enumerate(ort_sessions):
                bw_prediction, hidden_state, cell_state = orts.run(None, feed_dict)
                baseline_model_predictions[onnx_models[idx]].append(bw_prediction[0,0,0])
                
            obs = torch.tensor(observations[t:t+1,:]).to(device)
            
            with torch.no_grad():
                llm_prediction, std = bwe_llm(obs)
            baseline_model_predictions[torch_model].append(llm_prediction)

           
        
        for m in onnx_models:
            baseline_model_predictions[m] = np.asarray(baseline_model_predictions[m], dtype=np.float32)
        # baseline_model_predictions[torch_model] = np.asarray(baseline_model_predictions[torch_model].cpu(), dtype=np.float32)
        baseline_model_predictions[torch_model] = [tensor.cpu().numpy() for tensor in baseline_model_predictions[torch_model]]
        baseline_model_predictions[torch_model] = np.asarray(baseline_model_predictions[torch_model], dtype=np.float32)
            
        fig = plt.figure(figsize=(6, 3))
        time_s = np.arange(0, observations.shape[0]*60,60)/1000
        for idx, m in enumerate(onnx_models):
            plt.plot(time_s, baseline_model_predictions[m] / 1000, linestyle='-', label=['Baseline', 'Our model'][idx], color='C' + str(idx))
        plt.plot(time_s, baseline_model_predictions[torch_model] / 1000, linestyle='-', label='Qwen_bwe', color='C' + str(2))
        plt.plot(time_s, bandwidth_predictions/1000, linestyle='--', label='Estimator ' + call_data['policy_id'], color='C' + str(3))
        plt.plot(time_s, true_capacity/1000, label='True Capacity', color='black')
        plt.xlim(0, 125)
        plt.ylim(0)
        plt.ylabel("Bandwidth (Kbps)")
        plt.xlabel("Duration (second)")
        plt.grid(True)
        
        plt.legend(bbox_to_anchor=(0.5, 1.05), ncol=4, handletextpad=0.1, columnspacing=0.5,
                    loc='center', frameon=False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, os.path.basename(filename).replace(".json",".pdf")), dpi=300)
        plt.close()