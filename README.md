# LLM4Band
This repository contains the source code for the paper "[LLM4Band: Enhancing Reinforcement Learning with Large Language Models for Accurate Bandwidth Estimation](https://dl.acm.org/doi/10.1145/3712678.3721880)".

# How to use?
## Offline training
- Download the dataset from [RL4BandwidthEstimationChallenge](https://github.com/microsoft/RL4BandwidthEstimationChallenge), download the pre-trained model from [huggingface](https://huggingface.co/)([gpt2](https://huggingface.co/openai-community/gpt2), [t5](https://huggingface.co/google-t5/t5-base), [qwen](https://huggingface.co/Qwen/Qwen1.5-0.5B)).
- Split the dataset and preprocess the data (pickle format).
- Replace the model path in the code, train model: run `IQL.py`.
## Offline testing
- Prepare offline testing scenario in `validation/prepare_scenario`, evaluate the model in `validation/evaluate`.
## Online application
- Test environment: [AlphaRTC](https://github.com/OpenNetLab/AlphaRTC)
- Download link for the docker image: [alphartc4band](https://pan.baidu.com/s/1ZlkEEDYT37o0YSfnq1XCKQ?pwd=zdev)
- Download link for the test media: [testmedia](https://pan.baidu.com/s/1Ff_50IjUR2MCe3UZFrx4Ig?pwd=vknw)
- Limit port traffic, run:
  
        modprobe sch_netem

        modprobe sch_htb
  
        docker run --rm -it -v $(pwd)/LLM4Band:/app -w /app -e PYTHONPATH=/usr/lib/python3/dist-packages --name alphartc4band --cap-add=NET_ADMIN alphartc4band
- Entering the container, run:
  
          sudo /root/go/bin/comcast --device lo --target-port 8000 --target-bw 200 --latency 50 --packet-loss 1
  
          peerconnection_serverless receiver_pyinfer.json
- Stop:
  
          comcast --device lo --stop
- Perform the test task in another terminal：
  
        docker exec alphartc4band peerconnection_serverless sender_pyinfer.json
- Calculate the score：
  
          docker run --rm -v `pwd`/LLM4Band:/app -w /app/metrics --name eval alphartc4band python3 eval_network.py --dst_network_log /app/logging/webrtc.log --output /app/result/out_eval_network.json --ground_recv_rate 500 --max_delay 500
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
