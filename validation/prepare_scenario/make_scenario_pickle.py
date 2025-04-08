
import json
import os
import numpy as np
import math
import pickle
from tqdm import tqdm
import random
import shutil

project_root_path = '/your_path/validation/prepare_scenario'
dir_name = ['high_bw_call', 'high_fluctuating_bw', 'low_bw_call', 'low_fluctuating_bw']
pickle_name = ['high_bw_call.pickle', 'high_fluctuating_bw.pickle', 'low_bw_call.pickle', 'low_fluctuating_bw.pickle']

def load_bwec_dataset(dir_path):
    obs_ = []
    cs_ = []

    for session in tqdm(os.listdir(dir_path), desc='Making pickle'):
        session_path = os.path.join(dir_path, session)
        with open(session_path, 'r', encoding='utf-8') as jf:
            single_session_trajectory = json.load(jf)
            observations = single_session_trajectory['observations']
            capacitys = single_session_trajectory['true_capacity']

            obs = []
            cs = []
            for idx in range(len(observations)):

                obs.append(observations[idx])
                cs.append([capacitys[idx]])

        # check dim
        assert len(obs) == len(cs), 'DIM not match'

        # expaned into x_
        obs_.extend(obs)
        cs_.extend(cs)
        # break

    return {
        'observations': np.asarray(obs_, dtype=np.float32),
        'true_capacity': np.asarray(cs_, dtype=np.float32),
    }

def load_bwec_dataset_tested(dir_path):
    obs_ = []
    action_ = []

    for session in tqdm(os.listdir(dir_path), desc='Making pickle'):
        session_path = os.path.join(dir_path, session)
        with open(session_path, 'r', encoding='utf-8') as jf:
            single_session_trajectory = json.load(jf)
            observations = single_session_trajectory['observations']
            actions = single_session_trajectory['bandwidth_predictions']

            obs = []
            action = []
            for idx in range(len(observations)):

                obs.append(observations[idx])
                action.append(actions[idx])

        # expaned into x_
        obs_.extend(obs)
        action_.extend(action)
        # break

    return {
        'observations': np.asarray(obs_, dtype=np.float32),
        'actions': np.asarray(action_, dtype=np.float32),
    }

if __name__ == '__main__':
    for idx in range(1):
        # scenarios_dataset = load_bwec_dataset(os.path.join(project_root_path, dir_name[idx]))
        scenarios_dataset = load_bwec_dataset_tested(os.path.join(project_root_path, tested_dir_name[idx]))
        print(tested_dir_name[idx] + '_dataset dumping...')
        dataset_file_path = os.path.join(project_root_path, 'scenario_pickle', tested_pickle_name[idx])
        dataset_file = open(dataset_file_path, 'wb')
        pickle.dump(scenarios_dataset, dataset_file)
        dataset_file.close()
    print('done')