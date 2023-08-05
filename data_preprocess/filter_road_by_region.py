# 根据下载的朝阳区与海淀区的边界 geojson 文件，筛出海淀区的道路与朝阳区道路
from shapely.geometry import shape, Point, LineString
from shapely.ops import unary_union
import json
from tqdm import tqdm
import argparse
import os
import pandas as pd


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
    with open('../data/BJ_Taxi/朝阳区.json', 'r', encoding='utf-8') as f:
        region_A_geo = json.load(f)
        region_A_poly = shape(region_A_geo['features'][0]['geometry'])

    with open('../data/BJ_Taxi/海淀区.json', 'r', encoding='utf-8') as f:
        region_B_geo = json.load(f)
        region_B_poly = shape(region_B_geo['features'][0]['geometry'])

    with open('../data/BJ_Taxi/rid_gps.json', 'r') as f:
        road_gps = json.load(f)
    rid_info = pd.read_csv(os.path.join(data_root, dataset_name, 'roadmap.geo'))
elif dataset_name == 'Xian':
    # dataset_name == 'Xian'
    with open(os.path.join(data_root, dataset_name, '西安市下辖区.json'), 'r', encoding='utf-8') as f:
        city_geojson = json.load(f)
    xincheng = shape(city_geojson['features'][0]['geometry'])
    beilin = shape(city_geojson['features'][1]['geometry'])
    lianhu = shape(city_geojson['features'][2]['geometry'])
    # 这三个区合并为区域 A
    region_A_poly = unary_union([xincheng, beilin, lianhu])
    yanta = shape(city_geojson['features'][5]['geometry'])
    region_B_poly = yanta
    rid_info = pd.read_csv(os.path.join(data_root, dataset_name, 'xian.geo'))
    if not os.path.exists(os.path.join(data_root, dataset_name, 'rid_gps.json')):
        # 计算 road gps
        road_gps = {}
        for index, row in tqdm(rid_info.iterrows(), total=rid_info.shape[0], desc='cal road gps dict'):
            rid = row['geo_id']
            coordinate = eval(row['coordinates'])
            road_line = LineString(coordinates=coordinate)
            center_coord = road_line.centroid
            center_lon, center_lat = center_coord.x, center_coord.y
            road_gps[rid] = (center_lon, center_lat)
        with open(os.path.join(data_root, dataset_name, 'rid_gps.json'), 'w') as f:
            json.dump(road_gps, f)
    else:
        with open(os.path.join(data_root, dataset_name, 'rid_gps.json'), 'r') as f:
            road_gps = json.load(f)
else:
    # Chengdu
    with open(os.path.join(data_root, dataset_name, '成都市下辖区.json'), 'r', encoding='utf-8') as f:
        city_geojson = json.load(f)
    jinjiang = shape(city_geojson['features'][0]['geometry'])
    qingyang = shape(city_geojson['features'][1]['geometry'])
    wuhou = shape(city_geojson['features'][3]['geometry'])
    chenghua = shape(city_geojson['features'][4]['geometry'])
    region_A_poly = unary_union([qingyang, wuhou])
    region_B_poly = unary_union([jinjiang, chenghua])
    rid_info = pd.read_csv(os.path.join(data_root, dataset_name, 'chengdu.geo'))
    if not os.path.exists(os.path.join(data_root, dataset_name, 'rid_gps.json')):
        # 计算 road gps
        road_gps = {}
        for index, row in tqdm(rid_info.iterrows(), total=rid_info.shape[0], desc='cal road gps dict'):
            rid = row['geo_id']
            coordinate = eval(row['coordinates'])
            road_line = LineString(coordinates=coordinate)
            center_coord = road_line.centroid
            center_lon, center_lat = center_coord.x, center_coord.y
            road_gps[rid] = (center_lon, center_lat)
        with open(os.path.join(data_root, dataset_name, 'rid_gps.json'), 'w') as f:
            json.dump(road_gps, f)
    else:
        with open(os.path.join(data_root, dataset_name, 'rid_gps.json'), 'r') as f:
            road_gps = json.load(f)

region_A_road_list = set()
region_B_road_list = set()


# for road in tqdm(road_gps):
#     center_gps = road_gps[road]
#     gps_point = Point(center_gps[0], center_gps[1])
#     if region_A_poly.covers(gps_point):
#         # 这个点在海淀区里
#         region_A_road_list.add(int(road))
#     elif region_B_poly.covers(gps_point):
#         region_B_road_list.add(int(road))
# 更新一下计算方式

for index, row in tqdm(rid_info.iterrows(), total=rid_info.shape[0]):
    road_coord = eval(row['coordinates'])
    road_line = LineString(coordinates=road_coord)
    road_id = row['geo_id']
    if region_A_poly.intersects(road_line) or region_A_poly.covers(road_line):
        region_A_road_list.add(road_id)
    if region_B_poly.intersection(road_line) or region_B_poly.covers(road_line):
        region_B_road_list.add(road_id)


if local:
    # 做一个可视化，来验证一下化的好不好
    geojson = dict()
    geojson['type'] = 'FeatureCollection'
    obj_list = []
    for index, row in rid_info.iterrows():
        rid = row['geo_id']
        coordinate = eval(row['coordinates'])
        if rid in region_A_road_list and rid not in region_B_road_list:
            rid_region_id = 1
        elif rid in region_B_road_list and rid not in region_A_road_list:
            rid_region_id = 2
        elif rid in region_A_road_list and rid in region_B_road_list:
            rid_region_id = 3
        else:
            rid_region_id = 4
        obj = {'type': 'Feature', 'geometry': {}, 'properties': {'region_id': rid_region_id}}
        obj['geometry']['type'] = 'LineString'
        obj['geometry']['coordinates'] = coordinate
        obj_list.append(obj)
    geojson['features'] = obj_list
    with open('../data/{}/region_road_map.json'.format(dataset_name), 'w') as f:
        json.dump(geojson, f)

print(len(region_A_road_list))
print(len(region_B_road_list))

# 保存
res = {
    'region_A_road_list': list(region_A_road_list),
    'region_B_road_list': list(region_B_road_list)
}

with open(os.path.join(data_root, dataset_name, 'region_road_dict.json'), 'w') as f:
    json.dump(res, f)
