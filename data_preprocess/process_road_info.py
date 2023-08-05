"""
Preprocess road network information and output intermediate results, e.g. road_adjacent_list
"""
import pandas as pd
from tqdm import tqdm
import json

rid_rel = pd.read_csv('../data/BJ_Taxi/roadmap.rel')
road_adjacent_list = {}
for index, row in tqdm(rid_rel.iterrows(), total=rid_rel.shape[0], desc='cal road adjacent list'):
    f_rid = str(row['origin_id'])
    t_rid = row['destination_id']
    if f_rid not in road_adjacent_list:
        road_adjacent_list[f_rid] = [t_rid]
    else:
        road_adjacent_list[f_rid].append(t_rid)

with open('../data/BJ_Taxi/road_adjacent_list.json', 'w') as f:
    json.dump(road_adjacent_list, f)