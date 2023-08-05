# 分析轨迹数据的时间分布情况
import pandas as pd
from tqdm import tqdm

dataset_name = 'Chengdu'
if dataset_name == 'Xian':
    data_filename = '/mnt/data/jwj/Xian/xianshi_partA_traj_mm_processed.csv'
else:
    assert dataset_name == 'Chengdu'
    data_filename = '/mnt/data/jwj/Chengdu/chengdushi_partA_traj_mm_processed.csv'
data = pd.read_csv(data_filename)

date_set = set()
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    time_list = row['time_list'].split(',')
    for timestamp in time_list:
        date_set.add(timestamp[:10])

print(date_set)
print(len(date_set))
# 从 2013.07.01 到 2014.7.01
