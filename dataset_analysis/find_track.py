import os
import json
import shutil
import numpy as np
from tqdm import tqdm

source_folder = '/your_path/testbed_dataset'
destination_folder = '/your_path/qoe<=6_call'

os.makedirs(destination_folder, exist_ok=True)

count = 0

for filename in tqdm(os.listdir(source_folder), desc='Processing:'):
    if filename.endswith('.json'):
        file_path = os.path.join(source_folder, filename)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        quality_videos = data['video_quality']
        quality_audios = data['audio_quality']
        
        avg_q_v = np.nanmean(np.asarray(quality_videos, dtype=np.float32))
        avg_q_a = np.nanmean(np.asarray(quality_audios, dtype=np.float32))
        
        total_quality = (avg_q_v + avg_q_a)
        
        if total_quality <= 6:
            shutil.copy(file_path, os.path.join(destination_folder, filename))
            count += 1

print(count)