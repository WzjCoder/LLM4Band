import os
import json
import numpy as np
import shutil
from tqdm import tqdm

high_bw_call_dir = "/your_path/validation/prepare_scenario/high_bw_call"
low_bw_call_dir = "/your_path/validation/prepare_scenario/low_bw_call"
high_variance_call_dir = "/your_path/validation/prepare_scenario/high_fluctuating_bw"
low_variance_call_dir = "/your_path/validation/prepare_scenario/low_fluctuating_bw"
data_dir = low_variance_call_dir
data_files = os.listdir(data_dir)
nan_num = 0

for filename in tqdm(data_files, desc="Processing"):
    file_path = os.path.join(data_dir, filename)
    with open(file_path, 'r') as file:
        data = json.load(file)
        bandwidth_predictions = data['true_capacity']
        if np.isnan(bandwidth_predictions).any():
            nan_num += 1
    
print(nan_num)
