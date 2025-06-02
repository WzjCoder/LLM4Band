from torch import nn
import torch
import torch.nn.functional as F

from .module import StateEncoder, BweHead
from transformers import AutoModelForCausalLM, AutoModel
from peft import LoraConfig, LoraModel
from utils.packet_info import PacketInfo
from utils.packet_record import PacketRecord
import collections
import time
import onnxruntime as ort

Qwen1_5_weight_hf_format_dir = "/app/huggingface/Qwen1.5-0.5B-hf"
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
empty_mi = {
            "start_time": 0,
            "end_time": 0,
            "received_packets": [],  # 空的接收包列表
            "lost_packets": 0,       # 丢失的包数量
            "received_bytes": 0,     # 接收到的总字节数
        }

class bwe_agent_Qwen_base(nn.Module):
    def __init__(self, max_action=20, state_dim=150, action_dim=1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.max_action = max_action # Mbps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.float32).to(self.device))
        LLM = AutoModelForCausalLM.from_pretrained(Qwen1_5_weight_hf_format_dir, use_cache=False, device_map=self.device)
        self.llm = LLM.base_model
        self.stateencoder = StateEncoder(self.state_dim, 1024, self.device)
        self.bwe_head = BweHead(1024, 512, self.action_dim, self.max_action, self.device)
        self.encoder0 = nn.Parameter(torch.tensor([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 
                                 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
                                 1,    1,    1,    1,    1,    1,    1,    1,    1,    1
                                 ], dtype=torch.float32).to(self.device).detach())
        self.encoder0.requires_grad_(False)
        # alphartc
        self.short_term_mis = collections.deque(maxlen=5)  # 最近 5 个短期间隔
        self.long_term_mis = collections.deque(maxlen=5)  # 最近 5 个长期间隔
        ## init
        for _ in range(5):
            self.short_term_mis.append(empty_mi)
            self.long_term_mis.append(empty_mi)
        self.mi_duration_short = 60  # 短期间隔的持续时间，单位 ms
        self.mi_duration_long = 600  # 长期间隔的持续时间，单位 ms
        self.current_short_mi = self.initialize_mi(0, self.mi_duration_short)  # 当前短期间隔
        self.current_long_mi = self.initialize_mi(0, self.mi_duration_long)  # 当前长期间隔

    def initialize_mi(self, start_time, duration):
        """初始化一个监控间隔的数据结构"""
        return {
            "start_time": start_time,  # 监控间隔开始时间
            "end_time": start_time + duration,  # 监控间隔结束时间
            "received_packets": [],  # 本间隔内接收到的数据包
            "lost_packets": 0,  # 丢失的数据包数量
            "received_bytes": 0,  # 接收到的总字节数
        }

    def update_mi(self, packet_info, mi_duration, current_mi, mi_queue):
        """
        更新监控间隔的统计信息，支持连续窗口
        """
        now = packet_info.receive_timestamp
        # 如果当前数据包超出监控间隔，则结束当前间隔并开启新间隔
        if now > current_mi["end_time"]:
            # 将当前间隔保存到队列
            mi_queue.append(current_mi)
            # 开启新的间隔：当前间隔的结束时间是新间隔的开始时间
            start_time = current_mi["end_time"]
            current_mi = self.initialize_mi(start_time, mi_duration)
        # 更新当前间隔的统计数据
        current_mi["received_packets"].append(packet_info)
        current_mi["received_bytes"] += packet_info.size

        return current_mi

    def report_states(self, stats: dict):
        """
        接收数据包统计信息并更新监控间隔
        """
        # 将输入数据包转换为 PacketInfo
        packet_info = PacketInfo()
        packet_info.payload_type = stats["payload_type"]
        packet_info.ssrc = stats["ssrc"]
        packet_info.sequence_number = stats["sequence_number"]
        packet_info.send_timestamp = stats["send_time_ms"]
        packet_info.receive_timestamp = stats["arrival_time_ms"]
        packet_info.padding_length = stats["padding_length"]
        packet_info.header_length = stats["header_length"]
        packet_info.payload_size = stats["payload_size"]
        packet_info.size = packet_info.header_length + packet_info.payload_size + packet_info.padding_length

        # 更新短期和长期监控间隔
        self.current_short_mi = self.update_mi(packet_info, self.mi_duration_short, self.current_short_mi, self.short_term_mis)
        self.current_long_mi = self.update_mi(packet_info, self.mi_duration_long, self.current_long_mi, self.long_term_mis)

    
    def forward(self, x):
        x = x * self.encoder0
        x = self.stateencoder(x)
        x = torch.unsqueeze(x, dim=1) # ——>[bs, 1, 1024]
        x = self.llm(inputs_embeds = x)
        x = torch.squeeze(x.last_hidden_state)
        x = self.bwe_head(x)
        return x
    
    def get_estimated_bandwidth(self)->int:
        
        # 计算短期和长期的特征
        observation = get_observation_vector(self.short_term_mis, self.long_term_mis)
        obs = torch.tensor(observation).reshape(1, 150)
        bw = self.forward(obs)
        return int(bw)
    

def calculate_loss_rate(mi):
    """
    Calculate the packet loss rate for a given monitoring interval (MI).
    Args:
        mi (dict): A monitoring interval containing received packet information.
    Returns:
        float: Loss rate for the given monitoring interval (0 <= loss_rate <= 1).
    """
    packets = mi["received_packets"]
    # 如果没有数据包，返回丢包率为 0（没有收到任何包）
    if not packets:
        return 0.0
    # 统计序列号范围以及接收包数量
    min_sequence_number = min(pkt.sequence_number for pkt in packets)
    max_sequence_number = max(pkt.sequence_number for pkt in packets)
    total_packets_expected = max_sequence_number - min_sequence_number + 1  # 理论上的总包数
    total_packets_received = len(packets)  # 实际接收到的包数
    # 计算丢包率
    if total_packets_expected > 0:
        loss_rate = 1 - (total_packets_received / total_packets_expected)
    else:
        loss_rate = 0.0  # 如果没有包发送，则丢包率为 0
    return loss_rate

def calculate_mi_features(mi):
    """
    计算单个监控间隔的 15 个特征值
    """
    packets = mi["received_packets"]
    num_packets = len(packets)
    minimum_seen_delay = float('inf')
    base_delay = 200

    # 如果没有数据包，则返回默认特征
    if num_packets == 0:
        return [0] * 15

    # 特征计算
    receiving_rate = (mi["received_bytes"] * 8) / (mi["end_time"] - mi["start_time"])  # 接收速率，bps
    delays = [
        pkt.receive_timestamp - pkt.send_timestamp - base_delay for pkt in packets
    ]
    min_delay = min(pkt.receive_timestamp - pkt.send_timestamp for pkt in packets)
    minimum_seen_delay = min(minimum_seen_delay, min_delay)  # 更新全局最小延迟
    queuing_delays = [
        pkt.receive_timestamp - pkt.send_timestamp - minimum_seen_delay for pkt in packets
    ]
    interarrival_times = [packets[i].receive_timestamp - packets[i - 1].receive_timestamp for i in range(1, num_packets)]
    loss_rate = calculate_loss_rate(mi)
    avg_lost_packets = (1 - loss_rate) * num_packets
    # 计算特征
    features = [
        receiving_rate,  # 接收速率
        num_packets,  # 接收包数量
        mi["received_bytes"],  # 接收字节数
        sum(queuing_delays) / num_packets,  # 平均排队延迟
        sum(delays) / num_packets,  # 平均延迟 - 基准延迟
        minimum_seen_delay,  # 观察到的最小延迟
        sum(delays) / (min_delay * num_packets),  # 延迟比
        sum(delays) / num_packets - min_delay,  # 延迟差
        sum(interarrival_times) / len(interarrival_times) if interarrival_times else 0,  # 包间隔时间
        (sum([(x - sum(interarrival_times) / len(interarrival_times)) ** 2 for x in interarrival_times]) / len(interarrival_times)) ** 0.5 if interarrival_times else 0,  # 抖动
        loss_rate,  # 丢包率
        avg_lost_packets,  # 丢失包的平均数量
        len([pkt for pkt in packets if pkt.payload_type == 'video']) / num_packets,  # 视频包比例
        len([pkt for pkt in packets if pkt.payload_type == 'audio']) / num_packets,  # 音频包比例
        len([pkt for pkt in packets if pkt.payload_type == 'probe']) / num_packets,  # 探测包比例
    ]
    return features

# def get_observation_vector(short_term_mis, long_term_mis):
#     '''
#     Calculate estimated bandwidth
#     '''
#     # 计算短期和长期的特征
#     short_term_features = [calculate_mi_features(mi) for mi in short_term_mis]  # 5 x 15
#     long_term_features = [calculate_mi_features(mi) for mi in long_term_mis]    # 5 x 15
#     # 转置特征矩阵，使得每个指标的值连续存放
#     # 例如：短期特征从 (5 x 15) 转为 (15 x 5)，长期特征从 (5 x 15) 转为 (15 x 5)
#     short_term_features_transposed = list(zip(*short_term_features))  # 15 x 5
#     long_term_features_transposed = list(zip(*long_term_features))    # 15 x 5

#     # 将短期和长期的特征按指标拼接
#     observation_vector = []
#     for i in range(15):  # 遍历每个指标
#         observation_vector.extend(short_term_features_transposed[i])  # 添加短期的 5 个值
#         observation_vector.extend(long_term_features_transposed[i])   # 添加长期的 5 个值
#     return observation_vector

media_proportion = [0.5624007667367734, 0.525428472491675, 0.5398770595127375, 0.5335882890791457, 0.5281944796582948, 
                    0.5539928326103514, 0.5492280111798816, 0.5472668176812704, 0.5433039898216068, 0.5410861365126697, 
                    0.4329668165684116, 0.41940919988014613, 0.41652523714865447, 0.41276113321926694, 0.4230384000352399, 
                    0.4406922450131523, 0.43879027296869594, 0.4365506371537228, 0.43526221668141996, 0.432693335013149, 
                    0.00461964258628507, 0.0043729238066349254, 0.004236135735652494, 0.004341627231086488, 0.004255546605263082, 
                    0.005299858381903476, 0.005274764207886763, 0.005284779819482612, 0.005265374225193354, 0.005229165976840806]


def get_observation_vector(short_term_mis, long_term_mis):
    '''
    Calculate estimated bandwidth
    '''
    # 计算短期和长期的特征
    short_term_features = [calculate_mi_features(mi) for mi in short_term_mis]  # 5 x 15
    long_term_features = [calculate_mi_features(mi) for mi in long_term_mis]    # 5 x 15
    # 转置特征矩阵，使得每个指标的值连续存放
    # 例如：短期特征从 (5 x 15) 转为 (15 x 5)，长期特征从 (5 x 15) 转为 (15 x 5)
    short_term_features_transposed = list(zip(*short_term_features))  # 15 x 5
    long_term_features_transposed = list(zip(*long_term_features))    # 15 x 5

    # 将短期和长期的特征按指标拼接
    observation_vector = []
    for i in range(15):  # 遍历每个指标
        observation_vector.extend(short_term_features_transposed[i])  # 添加短期的 5 个值
        observation_vector.extend(long_term_features_transposed[i])   # 添加长期的 5 个值

    observation_vector[-30:] = media_proportion

    return observation_vector