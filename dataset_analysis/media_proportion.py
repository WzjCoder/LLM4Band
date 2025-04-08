import os
import json
from tqdm import tqdm

def calculate_observation_position_means(folder_path):
    """
    计算文件夹中所有 observation 的后 30 个位置的逐位置平均值。

    Args:
        folder_path (str): JSON 文件所在文件夹路径。

    Returns:
        list: 长度为 30 的列表，每个值表示所有 observation 在对应位置的平均值。
    """
    position_sums = [0] * 30  # 用于累加每个位置的值（长度为 30）
    total_observations = 0   # 记录有效 observation 的总数量

    # 遍历文件夹中的所有 JSON 文件
    for filename in tqdm(os.listdir(folder_path), desc="Processing files"):
        if filename.endswith(".json"):  # 确保只处理 JSON 文件
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, "r") as f:
                data = json.load(f)  # 加载 JSON 文件
                observations = data.get("observations", [])  # 获取 observations 列表

                # 遍历 observations 列表中的每个 observation
                for observation in observations:
                    last_30_values = observation[-30:]  # 取 observation 的后 30 个值
                    for i, value in enumerate(last_30_values):
                        position_sums[i] += value  # 对每个位置累加值
                    total_observations += 1  # 增加有效 observation 数量

    # 计算每个位置的平均值
    position_means = [position_sum / total_observations for position_sum in position_sums]
    return position_means

# 调用函数
folder_path = "/your_path/training_dataset"  # 替换为你的文件夹路径
result = calculate_observation_position_means(folder_path)

if result is not None:
    print(f"每个位置的平均值为: {result}")
else:
    print("没有有效的 observation 数据。")