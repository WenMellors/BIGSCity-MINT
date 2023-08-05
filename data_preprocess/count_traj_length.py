# 统计一下轨迹长度的平均值，最大值，最小值，众数
# 要不直接画个图出来吧，可以

import pandas as pd
from tqdm import tqdm
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

traj_file = '../data/Xian/xianshi_partB_mm_test_filtered.csv'
result_file = '../data/Xian/analyze_data_traj_statistics.json'

traj = pd.read_csv(traj_file)
len_cnt = {}

for index, row in tqdm(traj.iterrows(), total=traj.shape[0], desc='count length'):
    rid_list = row['rid_list'].split(',')
    length = len(rid_list)
    if length not in len_cnt:
        len_cnt[length] = 1
    else:
        len_cnt[length] += 1

with open(result_file, 'w') as f:
    json.dump(len_cnt, f)

# 下面根据 len_cnt 进行一些分析
# 为了绘制折线图，需要一个 np 数组
len_keys = list(len_cnt.keys())
min_len = np.min(len_keys)
max_len = np.max(len_keys)

# x 坐标，一共有多少个不同的 length，因为要连续，所以这边就对没有出现的 len 记为 0
x_array = np.arange(0, max_len+1, dtype=int)
# y 坐标，每个对应 len 点有多少条轨迹
y_array = np.zeros((max_len+1), dtype=int)

for i in range(max_len + 1):
    if i in len_cnt:
        y_array[i] = len_cnt[i]

avg_len = np.average(x_array, weights=y_array)
print('min length {}, max length {}, avg length {}'.format(min_len, max_len, avg_len))
# 绘制折线图
data = pd.DataFrame()
data['x'] = x_array
data['y'] = y_array

sns.lineplot(x='x', y='y', data=data)
plt.show()
