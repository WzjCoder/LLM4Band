import os
import json
import numpy as np
import math
import shutil
from tqdm import tqdm

data_dir = "/your_path/evaluate_dataset"
high_variance_call_dir = "/your_path/validation/prepare_scenario/high_fluctuating_bw"
low_variance_call_dir = "/your_path/validation/prepare_scenario/low_fluctuating_bw"

def calculate_variance(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        true_capacity = data['true_capacity']
        true_capacity_array = np.array(true_capacity, dtype=np.float32)
        if np.isnan(true_capacity).any():
            avg_c = np.nanmean(true_capacity_array)
            true_capacity_array = np.where(np.isnan(true_capacity_array), avg_c, true_capacity_array)
        return np.var(true_capacity_array)

def process_files(source_folder, high_variance_folder, low_variance_folder):
    if not os.path.exists(high_variance_folder):
        os.makedirs(high_variance_folder)
    if not os.path.exists(low_variance_folder):
        os.makedirs(low_variance_folder)

    variances = []
    data_files = os.listdir(source_folder)
    
    # Calculate variances
    for filename in tqdm(data_files, desc="Calculating Variances"):
        file_path = os.path.join(source_folder, filename)
        variance = calculate_variance(file_path)
        if variance is not None:
            variances.append(variance)
    
    if not variances:
        print("No valid 'bandwidth_prediction' found in any file.")
        return
    
    mean_variance = np.mean(variances)
    max_variance = np.max(variances)
    min_variance = np.min(variances)
    
    print(f"Mean Variance: {mean_variance}")
    print(f"Max Variance: {max_variance}")
    print(f"Min Variance: {min_variance}")

    # Copy files based on variance comparison with mean
    for idx, filename in tqdm(enumerate(data_files), desc="Copying Files"):
        file_path = os.path.join(source_folder, filename)
        variance = variances[idx]
        if variance is not None:
            if variance > mean_variance:
                shutil.copy(file_path, os.path.join(high_variance_folder, filename))
            elif variance != 0:
                shutil.copy(file_path, os.path.join(low_variance_folder, filename))

process_files(data_dir, high_variance_call_dir, low_variance_call_dir)
print(len(os.listdir(high_variance_call_dir)))
print(len(os.listdir(low_variance_call_dir)))
