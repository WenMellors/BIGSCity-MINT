# 由于我们是 Multi-agent 所以数据的输入格式跟之前的工作不一样，应该按照时间段来组织。
# 时间步编号 traj_id_list road_id_list, des_id_list
# 按一分钟进行编码
# 由于强化学习，需要按照 episode 来组织输入，因此我们这里按照 episode_len 来划分 episode （单位为分钟，保证为 60 的倍数）
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
                    default='Xian')
parser.add_argument('--region_name', type=str, default='partA')
parser.add_argument('--episode_len', type=int, default=60)

args = parser.parse_args()
local = args.local
dataset_name = args.dataset_name
region_name = args.region_name
episode_len = args.episode_len

if dataset_name == 'BJ_Taxi':
    data_filename = 'beijing_{}_mm_train.csv'.format(region_name)
    output_filename = 'beijing_{}_episode_{}_train.csv'.format(region_name, episode_len)
elif dataset_name == 'Xian':
    data_filename = 'xianshi_{}_mm_train.csv'.format(region_name)
    output_filename = 'xianshi_{}_episode_{}_train.csv'.format(region_name, episode_len)
else:
    assert dataset_name == 'Chengdu'
    data_filename = 'chengdushi_{}_mm_train.csv'.format(region_name)
    output_filename = 'chengdushi_{}_episode_{}_train.csv'.format(region_name, episode_len)

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
    time_code_range = 23 * 60 * 24

# 读取处理好的 map matching 轨迹
raw_data = pd.read_csv(os.path.join(data_root, dataset_name, data_filename))
for idx, row in tqdm(raw_data.iterrows(), total=raw_data.shape[0]):
    traj_id = row['traj_id']
    road_id_list = [int(x) for x in row['rid_list'].split(',')]
    time_list = row['time_list'].split(',')
    # 按照出发时间进行计算
    entry_timestamp = time_list[0]
    entry_time = datetime.strptime(entry_timestamp, '%Y-%m-%dT%H:%M:%SZ')
    # 时间合法性检验
    if entry_time < start_date or entry_time > end_date:
        continue
    time_delta = entry_time - start_date
    entry_time_code = time_delta.days * 1440 + time_delta.seconds // 60
    # 统计仿真时间步
    simulate_time_list = []
    for index, road in enumerate(road_id_list):
        simulate_time_list.append(entry_time_code)
        # 因为我们每一步都是 1 分钟，所以会有点多
        # TODO: 所以这里会有点问题
        entry_time_code += 1
        if entry_time_code >= time_code_range:
            break
    # 按照 episode_len 来划分 episode，重新组织每一个 episode 中 agent 的目的地
    # 如果这个 agent 的轨迹目的地不能够在当前 episode_len 中达到，那么就将其目的地修改为下一个 episode 的起点，反之不修改
    des_rid_list = []
    cur_des = None
    prev_episode_no = None
    for i in range(len(simulate_time_list) - 2, -1, -1):
        if i == len(simulate_time_list) - 2:
            # 倒数第二个点，无论是在哪个 episode 目的地都是他自己
            cur_episode_no = simulate_time_list[i] // episode_len
            cur_episode_time = simulate_time_list[i] % episode_len
            if cur_episode_time == episode_len - 1:
                # 当前 episode 最后一个时间片段
                # 那么在仿真的时候，这个点是不会被投入模型的，这个点前面的点的目的地应该是这个点
                cur_des = road_id_list[i]
                des_rid_list.append(road_id_list[i])
            else:
                # 当前点不是 episode 不是最后一个时间片段，那么这个点就可以被投入模型，其目的就是下一个点
                cur_des = road_id_list[i + 1]
                des_rid_list.append(cur_des)
        else:
            cur_episode_no = simulate_time_list[i] // episode_len
            cur_episode_time = simulate_time_list[i] % episode_len
            if cur_episode_no != prev_episode_no:
                # 退到新的 episode 了，由于连续性
                assert cur_episode_no == prev_episode_no - 1
                assert cur_episode_time == episode_len - 1
                # 那么在仿真的时候，这个点是不会被投入模型的，这个点前面的点的目的地应该是这个点
                cur_des = road_id_list[i]
                des_rid_list.append(road_id_list[i])
            else:
                # 沿用前面点的目的地就可以了
                des_rid_list.append(cur_des)
        prev_episode_no = cur_episode_no
    # 反转 des_rid_list
    des_rid_list = des_rid_list[::-1]
    assert len(des_rid_list) == len(road_id_list) - 1
    for index, entry_time_code in enumerate(simulate_time_list[:-1]):
        road = road_id_list[index]
        if entry_time_code not in res_dict:
            # 计算真实下一跳在 candidate 中的第几个
            candidate_mask = road_candidate_list[str(road)]
            target_id = 0
            for candidate in candidate_mask:
                if candidate == road_id_list[index + 1]:
                    break
                target_id += 1
            res_dict[entry_time_code] = [[traj_id], [road], [des_rid_list[index]], [target_id]]
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
            res_dict[entry_time_code][2].append(des_rid_list[index])
            res_dict[entry_time_code][3].append(target_id)
# 输出 res_dict
# 按照时间步，一步一行
time_code_keys = sorted(list(res_dict.keys()))
with open(os.path.join(data_root, dataset_name, output_filename), 'w') as f:
    f.write('time_code,traj_id_list,road_id_list,des_id_list,target_id_list,episode_no\n')
    for time_code in time_code_keys:
        f.write('{},\"{}\",\"{}\",\"{}\",\"{}\",{}\n'.format(time_code, ','.join([str(x) for x in res_dict[time_code][0]]),
                                                             ','.join([str(x) for x in res_dict[time_code][1]]),
                                                             ','.join([str(x) for x in res_dict[time_code][2]]),
                                                             ','.join([str(x) for x in res_dict[time_code][3]]),
                                                             time_code // episode_len))

