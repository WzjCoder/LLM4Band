import os
import json
from collections import defaultdict
from tqdm import tqdm

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
    elif true_capacity < 8e6:
        return '7-8MB'
    else:
        return '>8MB'

def count_true_capacity_ranges(folder_path):
    """Count the occurrences of true_capacity values within specified ranges."""
    range_counts = defaultdict(int)
    
    for filename in tqdm(os.listdir(folder_path), desc = 'processing:'):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                if 'true_capacity' in data:
                    true_capacities = data['true_capacity']
                    for value in true_capacities:
                        category = categorize_true_capacity(value)
                        range_counts[category] += 1
    
    return range_counts

# 使用示例
folder_path = '/your_path/training_dataset'
range_counts = count_true_capacity_ranges(folder_path)

# 打印结果
for range_category, count in range_counts.items():
    print(f'{range_category}: {count}')
