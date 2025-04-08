import os
import json
import numpy as np
import shutil
from tqdm import tqdm

data_dir = "/your_path/evaluate_dataset"
high_bw_call_dir = "/your_path/validation/prepare_scenario/high_bw_call"
low_bw_call_dir = "/your_path/validation/prepare_scenario/low_bw_call"

def calculate_variance(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        true_capacity = data['true_capacity']
        true_capacity_array = np.array(true_capacity, dtype=np.float32)
        if np.isnan(true_capacity).any():
            avg_c = np.nanmean(true_capacity_array)
            true_capacity_array = np.where(np.isnan(true_capacity_array), avg_c, true_capacity_array)
        return np.var(true_capacity_array), true_capacity_array[0]

def copy_files_with_zero_variance(source_folder, high_bw_call_dir, low_bw_call_dir):
    if not os.path.exists(high_bw_call_dir):
        os.makedirs(high_bw_call_dir)
    
    if not os.path.exists(low_bw_call_dir):
        os.makedirs(low_bw_call_dir)
    
    data_files = [f for f in os.listdir(source_folder) if f.endswith('.json')]

    for filename in tqdm(data_files, desc="Processing"):
        file_path = os.path.join(source_folder, filename)
        variance, true_capacity = calculate_variance(file_path)
        if variance is not None and variance == 0:
            if true_capacity <= 2e6:
                shutil.copy(file_path, os.path.join(low_bw_call_dir, filename))
            elif true_capacity >= 6e6:
                shutil.copy(file_path, os.path.join(high_bw_call_dir, filename))


copy_files_with_zero_variance(data_dir, high_bw_call_dir, low_bw_call_dir)
print(len(os.listdir(high_bw_call_dir)))
print(len(os.listdir(low_bw_call_dir)))
