
import json
import os
current_dir = os.path.split(os.path.abspath(__file__))[0]
project_root_path = current_dir.rsplit('/', 1)[0]
import numpy as np
import math
import pickle
from tqdm import tqdm
import random
import shutil


pickle_name = ['avg_qoe_training_dataset_pickle.pickle']

training_dataste_dir_path = os.path.join(project_root_path , 'training_dataset')

def load_bwec_dataset(dir_path):
    obs_ = []
    action_ = []
    next_obs_ = []
    reward_ = []
    done_ = []

    for session in tqdm(os.listdir(dir_path), desc='Making pickle'):
        session_path = os.path.join(dir_path, session)
        with open(session_path, 'r', encoding='utf-8') as jf:
            single_session_trajectory = json.load(jf)
            observations = single_session_trajectory['observations']
            actions = single_session_trajectory['bandwidth_predictions']
            quality_videos = single_session_trajectory['video_quality']
            quality_audios = single_session_trajectory['audio_quality']

            avg_q_v = np.nanmean(np.asarray(quality_videos, dtype=np.float32))
            avg_q_a = np.nanmean(np.asarray(quality_audios, dtype=np.float32))

            obs = []
            next_obs = []
            action = []
            reward = []
            for idx in range(len(observations)):

                r_v = quality_videos[idx]
                r_a = quality_audios[idx]
                if math.isnan(quality_videos[idx]):
                    r_v = avg_q_v
                if math.isnan(quality_audios[idx]):
                    r_a = avg_q_a
                reward.append(r_v + r_a )

                obs.append(observations[idx])
                if idx + 1 >= len(observations):
                    next_obs.append([-1] * len(observations[0]))  # s_terminal
                else:
                    next_obs.append(observations[idx + 1])
                action.append([actions[idx]])
            
            done_bool = [False] * (len(obs) - 1) + [True]

        # check dim
        assert len(obs) == len(next_obs) == len(action) == len(reward) == len(done_bool), 'DIM not match'

        # expaned into x_
        obs_.extend(obs)
        action_.extend(action)
        next_obs_.extend(next_obs)
        reward_.extend(reward)
        done_.extend(done_bool)
        # break

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }

if __name__ == '__main__':
    training_dataset = load_bwec_dataset(training_dataste_dir_path)
    print('training dataset dumping...')
    dataset_file_path = os.path.join(project_root_path, 'dataset', 'training_dataset_pickle', pickle_name[0])
    dataset_file = open(dataset_file_path, 'wb')
    pickle.dump(training_dataset, dataset_file)
    dataset_file.close()
    print('done')