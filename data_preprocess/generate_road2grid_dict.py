import json
import math
import argparse
import os
import numpy as np


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

with open(os.path.join(data_root, dataset_name, 'region_road_dict.json'), 'r') as f:
    region_road_dict = json.load(f)

region_A_road_list = region_road_dict['region_A_road_list']
region_B_road_list = region_road_dict['region_B_road_list']
total_region_road_list = region_A_road_list + region_B_road_list

with open(os.path.join(data_root, dataset_name, 'rid_gps.json'), 'r') as f:
    road_gps = json.load(f)

lon_0 = 360
lat_0 = 90
lon_1 = -360
lat_1 = 0
for road in total_region_road_list:
    lon, lat = road_gps[str(road)]
    lon_0 = min(lon, lon_0)
    lat_0 = min(lat, lat_0)
    lon_1 = max(lon, lon_1)
    lat_1 = max(lat, lat_1)

print('lon_0: {}, lon_1: {}, lat_0: {}, lat_1: {}'.format(lon_0, lon_1, lat_0, lat_1))
# 这里主要是为了保留一下小数，让区域不要那么极端
lon_0 = float(input('the lon_0 is: '))  # 104.0586067
lon_1 = float(input('the lon_1 is: '))  # 104.2261992
lat_0 = float(input('the lat_0 is: '))  # 30.54882404
lat_1 = float(input('the lat_1 is: '))  # 30.74048714
img_unit = 0.005  # 这样画出来的大小，大概是 0.42 km * 0.55 km 的格子
img_width = math.ceil((lon_1 - lon_0) / img_unit) + 1  # 图像的宽度
img_height = math.ceil((lat_1 - lat_0) / img_unit) + 1  # 映射出的图像的高度


road2grid = {}
for road in road_gps:
    if int(road) in total_region_road_list:
        gps = road_gps[road]
        x = math.ceil((gps[0] - lon_0) / img_unit)
        y = math.ceil((gps[1] - lat_0) / img_unit)
        road2grid[road] = (x, y)
        assert 0 <= x < img_width
        assert 0 <= y < img_height

with open(os.path.join(data_root, dataset_name, 'road2grid.json'), 'w') as f:
    json.dump(road2grid, f)

