# LLM4Band
This repository contains the source code for the paper "[LLM4Band: Enhancing Reinforcement Learning with Large Language Models for Accurate Bandwidth Estimation](https://dl.acm.org/doi/10.1145/3712678.3721880)".

# How to use?
## Offline training
- Download the dataset from [RL4BandwidthEstimationChallenge]((https://github.com/microsoft/RL4BandwidthEstimationChallenge)), download the pre-trained model from [huggingface](https://huggingface.co/)([gpt2](https://huggingface.co/openai-community/gpt2), [t5](https://huggingface.co/google-t5/t5-base), [qwen](https://huggingface.co/Qwen/Qwen1.5-0.5B)).
- Split the dataset and preprocess the data (pickle format).
## Offline test
## Online application
# Citation
@inproceedings{wang2025llm4band,
  title={LLM4Band: Enhancing Reinforcement Learning with Large Language Models for Accurate Bandwidth Estimation},
  author={Wang, Zhijian and Lu, Rongwei and Zhang, Zhiyang and Westphal, Cedric and He, Dongbiao and Jiang, Jingyan},
  booktitle={Proceedings of the 35th Workshop on Network and Operating System Support for Digital Audio and Video},
  pages={43--49},
  year={2025}
}
# Acknowledgments
- [RL4BandwidthEstimationChallenge](https://github.com/microsoft/RL4BandwidthEstimationChallenge) - dataset
- [AlphaRTC](https://github.com/OpenNetLab/AlphaRTC) - simulation platform
- [NAORL](https://github.com/bytedance/offline-RL-congestion-control), [CORL](https://github.com/tinkoff-ai/CORL), [HuggingFace](https://huggingface.co/) - tools
- [BoB](https://github.com/NUStreaming/BoB), [Schaferct](https://github.com/n13eho/Schaferct), [HRCC](https://github.com/thegreatwb/HRCC), [Pioneer](https://github.com/sjtu-medialab/Pioneer) - baselines
