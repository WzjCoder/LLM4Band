#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from estimator.heuristic import HeuristicEstimator
from estimator.gcc import GCCEstimator, Estimator
from estimator.hrcc import HRCCEstimator
from estimator.bob import BobEstimator
from estimator.estimator import bwe_agent_Qwen_base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qwen_lora_merged_range_average_path = '/app/model/qwen_merged.pth'



'''
using Bob
'''
# class Estimator(BobEstimator):
#     def __init__(self):
#         super().__init__()


'''
using HRCC
'''
# class Estimator(HRCCEstimator):
#     def __init__(self):
#         super().__init__()

'''
using GCC
'''
class Estimator(GCCEstimator):
    def __init__(self):
        super().__init__()

'''
using qwen_lora_merged
'''
# class Estimator(bwe_agent_Qwen_base):
#     def __init__(self):
#         super().__init__()
#         save_dict = torch.load(qwen_lora_merged_range_average_path, map_location=device)
#         self.stateencoder.load_state_dict(save_dict["stateencoder"])
#         self.llm.load_state_dict(save_dict["llm"])
#         self.bwe_head.load_state_dict(save_dict["bwe_head"])