import json
import os
import numpy as np
import math
import pickle
from tqdm import tqdm
import random
from collections import defaultdict

training_dataset_dir_path = '/your_path/training_dataset'

def categorize_true_capacity(true_capacity):
    """Categorize a value of true_capacity into the specified ranges."""
    if true_capacity < 1e6:
        return '0-1MB'
    elif true_capacity < 2e6:
        return '1-2MB'
    elif true_capacity < 3e6:
        return '2-3MB'
    elif true_capacity < 4e6:
        return '3-4MB'
    elif true_capacity < 5e6:
        return '4-5MB'
    elif true_capacity < 6e6:
        return '5-6MB'
    elif true_capacity < 7e6:
        return '6-7MB'
    else:
        return '>7MB'

def load_bwec_dataset(dir_path):
    categorized_data = defaultdict(list)

    for session in tqdm(os.listdir(dir_path), desc='Loading data'):
        session_path = os.path.join(dir_path, session)
        try:
            with open(session_path, 'r', encoding='utf-8') as jf:
                single_session_trajectory = json.load(jf)
        except Exception as e:
            print(f"Error reading {session_path}: {e}")
            continue

        observations = single_session_trajectory['observations']
        actions = single_session_trajectory['bandwidth_predictions']
        quality_videos = single_session_trajectory['video_quality']
        quality_audios = single_session_trajectory['audio_quality']
        true_capacitys = single_session_trajectory['true_capacity']

        avg_q_v = np.nanmean(np.asarray(quality_videos, dtype=np.float32))
        avg_q_a = np.nanmean(np.asarray(quality_audios, dtype=np.float32))

        for idx in range(len(observations)):
            r_v = quality_videos[idx]
            r_a = quality_audios[idx]
            if math.isnan(quality_videos[idx]):
                r_v = avg_q_v
            if math.isnan(quality_audios[idx]):
                r_a = avg_q_a

            reward = r_v + r_a # 1:1
            obs = observations[idx]
            next_obs = observations[idx + 1] if idx + 1 < len(observations) else [-1] * len(observations[0])
            action = [actions[idx]]
            true_capacity = true_capacitys[idx]
            done = idx == len(observations) - 1

            category = categorize_true_capacity(true_capacity)
            categorized_data[category].append((obs, action, next_obs, reward, done, true_capacity))

    return categorized_data

def sample_and_save_data(categorized_data, num_samples_per_category, output_file):
    sampled_data = {
        'observations': [],
        'actions': [],
        'next_observations': [],
        'rewards': [],
        'terminals': [],
        'true_capacitys': [],
    }

    for category, data in categorized_data.items():
        if len(data) > num_samples_per_category:
            sampled_data_indices = random.sample(range(len(data)), num_samples_per_category)
        else:
            sampled_data_indices = range(len(data))

        for idx in sampled_data_indices:
            obs, action, next_obs, reward, done , true_capacity= data[idx]
            sampled_data['observations'].append(obs)
            sampled_data['actions'].append(action)
            sampled_data['next_observations'].append(next_obs)
            sampled_data['rewards'].append(reward)
            sampled_data['terminals'].append(done)
            sampled_data['true_capacitys'].append(true_capacity)

    # 将数据转换为numpy数组
    for key in sampled_data:
        sampled_data[key] = np.array(sampled_data[key])

    # 保存到pickle文件
    with open(output_file, 'wb') as f:
        pickle.dump(sampled_data, f)

if __name__ == '__main__':
    categorized_data = load_bwec_dataset(training_dataset_dir_path)
    print('Sampling and saving training dataset...')
    output_file_path = '/your_path/training_dataset_pickle/range_average_training_supervised_dataset_pickle.pickle'
    sample_and_save_data(categorized_data, 500000, output_file_path)
    print('Done')
