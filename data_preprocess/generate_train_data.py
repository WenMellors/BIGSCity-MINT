# 由于我们是 Multi-agent 所以数据的输入格式跟之前的工作不一样，应该按照时间段来组织。
# 时间步编号 traj_id_list road_id_list, des_id_list
# 按一分钟进行编码
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import json
import argparse
import os


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('bool value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--local', type=str2bool,
                    default=True, help='whether save the trained model')
parser.add_argument('--dataset_name', type=str,
                    default='BJ_Taxi')
parser.add_argument('--data_filename', type=str,
                    default='chaoyang_traj_mm_train.csv')
parser.add_argument('--output_filename', type=str,
                    default='chaoyang_traj_input_train.csv')
args = parser.parse_args()
local = args.local
dataset_name = args.dataset_name
data_filename = args.data_filename
output_filename = args.output_filename

if local:
    data_root = '../data/'
else:
    data_root = '/mnt/data/jwj/'

with open(os.path.join(data_root, dataset_name, 'road_candidate_list.json'), 'r') as f:
    road_candidate_list = json.load(f)

res_dict = {}

# 计算时间跨度
if dataset_name == 'BJ_Taxi':
    start_date = datetime(2015, 11, 1)
    end_date = datetime(2015, 11, 30, 23, 59, 59)
    time_code_range = 30 * 60 * 24
elif dataset_name == 'Porto_Taxi':
    start_date = datetime(2013, 7, 1)
    end_date = datetime(2014, 7, 1, 23, 59, 59)
    time_code_range = 366 * 60 * 24
else:
    # Xian and Chengdu
    start_date = datetime(2018, 10, 31)
    end_date = datetime(2018, 11, 30, 23, 59, 59)
    time_code_range = 31 * 60 * 24

# 读取处理好的 map matching 轨迹
raw_data = pd.read_csv(os.path.join(data_root, dataset_name, data_filename))
for idx, row in tqdm(raw_data.iterrows(), total=raw_data.shape[0]):
    traj_id = row['traj_id']
    road_id_list = [int(x) for x in row['rid_list'].split(',')]
    time_list = row['time_list'].split(',')
    # 按照出发时间进行计算
    # 按一分钟编码，周末与工作日区分开来
    entry_timestamp = time_list[0]
    entry_time = datetime.strptime(entry_timestamp, '%Y-%m-%dT%H:%M:%SZ')
    # 时间合法性检验
    if entry_time < start_date or entry_time > end_date:
        continue
    time_delta = entry_time - start_date
    entry_time_code = time_delta.days * 1440 + time_delta.seconds // 60
    if entry_time_code < 0:
        debug = 1
    # 然后记录下本条轨迹
    des = road_id_list[-1]
    for index, road in enumerate(road_id_list[:-1]):
        if entry_time_code not in res_dict:
            # 计算真实下一跳在 candidate 中的第几个
            candidate_mask = road_candidate_list[str(road)]
            target_id = 0
            for candidate in candidate_mask:
                if candidate == road_id_list[index + 1]:
                    break
                target_id += 1
            res_dict[entry_time_code] = [[traj_id], [road], [des], [target_id]]
        else:
            # 计算真实下一跳在 candidate 中的第几个
            candidate_mask = road_candidate_list[str(road)]
            target_id = 0
            for candidate in candidate_mask:
                if candidate == road_id_list[index + 1]:
                    break
                target_id += 1
            res_dict[entry_time_code][0].append(traj_id)
            res_dict[entry_time_code][1].append(road)
            res_dict[entry_time_code][2].append(des)
            res_dict[entry_time_code][3].append(target_id)
        # 因为我们每一步都是 1 分钟，所以会有点多
        # TODO: 所以这里会有点问题
        entry_time_code += 1
        if entry_time_code >= time_code_range:
            break

# 输出 res_dict
# 按照时间步，一步一行
time_code_keys = sorted(list(res_dict.keys()))
with open(os.path.join(data_root, dataset_name, output_filename), 'w') as f:
    f.write('time_code,traj_id_list,road_id_list,des_id_list,target_id_list\n')
    for time_code in time_code_keys:
        f.write('{},\"{}\",\"{}\",\"{}\",\"{}\"\n'.format(time_code, ','.join([str(x) for x in res_dict[time_code][0]]),
                                                          ','.join([str(x) for x in res_dict[time_code][1]]),
                                                          ','.join([str(x) for x in res_dict[time_code][2]]),
                                                          ','.join([str(x) for x in res_dict[time_code][3]])))

