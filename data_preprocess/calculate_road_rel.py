# 构建 road_surrounding_list 和 road_candidate_list
import json
import pandas as pd
from tqdm import tqdm
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

args = parser.parse_args()
local = args.local
dataset_name = args.dataset_name

if local:
    data_root = '../data/'
else:
    data_root = '/mnt/data/jwj/'

if dataset_name == 'BJ_Taxi':
    road_rel = pd.read_csv(os.path.join(data_root, dataset_name, 'bj_roadmap.rel'))
elif dataset_name == 'Porto':
    road_rel = pd.read_csv(os.path.join(data_root, dataset_name, 'porto.rel'))
elif dataset_name == 'Xian':
    # dataset_name == 'Xian'
    road_rel = pd.read_csv(os.path.join(data_root, dataset_name, 'xian.rel'))
else:
    assert dataset_name == 'Chengdu'
    road_rel = pd.read_csv(os.path.join(data_root, dataset_name, 'chengdu.rel'))

candidate_list = {}
surrounding_list = {}
for index, row in tqdm(road_rel.iterrows(), total=road_rel.shape[0], desc='cal road adjacent list'):
    f_rid = str(row['origin_id'])
    t_rid = row['destination_id']
    if f_rid not in candidate_list:
        candidate_list[f_rid] = [t_rid]
    else:
        candidate_list[f_rid].append(t_rid)
    if f_rid not in surrounding_list:
        surrounding_list[f_rid] = set()
        surrounding_list[f_rid].add(t_rid)
        surrounding_list[f_rid].add(f_rid)
    else:
        surrounding_list[f_rid].add(t_rid)
    if t_rid not in surrounding_list:
        surrounding_list[t_rid] = set()
        surrounding_list[t_rid].add(t_rid)
        surrounding_list[t_rid].add(f_rid)
    else:
        surrounding_list[t_rid].add(f_rid)

for key in surrounding_list:
    surrounding_list[key] = list(surrounding_list[key])

with open(os.path.join(data_root, dataset_name, 'road_candidate_list.json'), 'w') as f:
    json.dump(candidate_list, f)

with open(os.path.join(data_root, dataset_name, 'road_surrounding_list.json'), 'w') as f:
    json.dump(surrounding_list, f)

# with open('../data/BJ_Taxi/region_road_dict.json', 'r') as f:
#     region_road_dict = json.load(f)

# chaoyang_road_set = set(region_road_dict['chaoyang_road_list'])
# # 只计算朝阳区的 candidate_list
# candidate_list = {}
# surrounding_list = {}
# for index, row in tqdm(road_rel.iterrows(), total=road_rel.shape[0], desc='cal road adjacent list'):
#     f_rid = str(row['origin_id'])
#     t_rid = row['destination_id']
#     if int(f_rid) not in chaoyang_road_set or int(t_rid) not in chaoyang_road_set:
#         continue
#     if f_rid not in candidate_list:
#         candidate_list[f_rid] = [t_rid]
#     else:
#         candidate_list[f_rid].append(t_rid)
#     if f_rid not in surrounding_list:
#         surrounding_list[f_rid] = set()
#         surrounding_list[f_rid].add(t_rid)
#         surrounding_list[f_rid].add(f_rid)
#     else:
#         surrounding_list[f_rid].add(t_rid)
#     if t_rid not in surrounding_list:
#         surrounding_list[t_rid] = set()
#         surrounding_list[t_rid].add(t_rid)
#         surrounding_list[t_rid].add(f_rid)
#     else:
#         surrounding_list[t_rid].add(f_rid)
#
# # 要把自己加入 surrounding_list 之中
# for road in chaoyang_road_set:
#     if road not in surrounding_list:
#         surrounding_list[road] = set()
#         surrounding_list[road].add(road)
#     else:
#         surrounding_list[road].add(road)
#
# for key in surrounding_list:
#     surrounding_list[key] = list(surrounding_list[key])
#
# with open('../data/BJ_Taxi/chaoyang_road_candidate_list.json', 'w') as f:
#     json.dump(candidate_list, f)
#
# with open('../data/BJ_Taxi/chaoyang_road_surrounding_list.json', 'w') as f:
#     json.dump(surrounding_list, f)