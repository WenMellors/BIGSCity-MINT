# 根据下载的朝阳区与海淀区的边界 geojson 文件，筛出海淀区轨迹与朝阳区轨迹
from shapely.geometry import shape, Point
import json
import os
from tqdm import tqdm


with open('../data/BJ_Taxi/海淀区.json', 'r', encoding='utf-8') as f:
    haidian_region = json.load(f)
    haidian_poly = shape(haidian_region['features'][0]['geometry'])

with open('../data/BJ_Taxi/朝阳区.json', 'r', encoding='utf-8') as f:
    chaoyang_region = json.load(f)
    chaoyang_poly = shape(chaoyang_region['features'][0]['geometry'])

debug = False
if debug:
    dataset_path = 'E:\\Continuous-Trajectory-Generation-V2\\data\\BJ_Taxi\\201511\\'
else:
    dataset_path = '/home/mellesj/public/Dataset/BeijingTaxi/201511/'

haidian_traj_id = 0
chaoyang_traj_id = 0
haidian_traj_file = open('../data/BJ_Taxi/haidian_gps_traj_201511.csv', 'w')
haidian_traj_file.write('id;x;y;timestamp\n')
chaoyang_traj_file = open('../data/BJ_Taxi/chaoyang_gps_traj_201511.csv', 'w')
chaoyang_traj_file.write('id;x;y;timestamp\n')


def output_traj(traj_id, traj, fp):
    if len(traj) < 5:
        return
    for point in traj:
        fp.write('{};{};{};{}\n'.format(traj_id, point[0], point[1], point[2]))


file_list = os.listdir(dataset_path)
for filename in tqdm(file_list, total=len(file_list)):
    if len(filename) > 4 and filename[-4:] == '.csv':
        # 是轨迹数据
        with open(os.path.join(dataset_path, filename), 'r') as f:
            cache_traj_haidian = []
            cache_traj_chaoyang = []
            mode = 0  # 0: 还没有找到任何轨迹，1: 已经找到了部分海淀区的轨迹, 2: 已经找到了部分朝阳区的轨迹
            for line in f.readlines():
                items = line.split('|')
                if len(items) != 12:
                    continue
                # 判断是否是有效定位
                if items[7] == '40000000':
                    continue
                # 判断是否是载客
                if items[8] == '0':
                    # 没有载客了，如果当前有匹配到的轨迹需要输出
                    if mode == 1:
                        output_traj(haidian_traj_id, cache_traj_haidian, haidian_traj_file)
                        cache_traj_haidian = []
                        haidian_traj_id += 1
                        mode = 0
                    elif mode == 2:
                        output_traj(chaoyang_traj_id, cache_traj_chaoyang, chaoyang_traj_file)
                        cache_traj_chaoyang = []
                        chaoyang_traj_id += 1
                        mode = 0
                    continue
                try:
                    lat = int('0x' + items[2], 16) / 100000
                    lon = int('0x' + items[3], 16) / 100000
                except ValueError:
                    continue
                try:
                    timestamp = int('0x' + items[0], 16)
                except:
                    continue
                gps_point = Point(lon, lat)
                if mode == 0:
                    if haidian_poly.intersects(gps_point):
                        # 这个点在海淀区里
                        cache_traj_haidian.append((lon, lat, timestamp))
                        mode = 1
                    elif chaoyang_poly.intersects(gps_point):
                        cache_traj_chaoyang.append((lon, lat, timestamp))
                        mode = 2
                elif mode == 1:
                    if haidian_poly.intersects(gps_point):
                        # 这个点还在海淀区里
                        cache_traj_haidian.append((lon, lat, timestamp))
                    elif chaoyang_poly.intersects(gps_point):
                        # 跑到朝阳区了
                        cache_traj_chaoyang.append((lon, lat, timestamp))
                        # 并且要输出之前找到的海淀区轨迹，并清空
                        output_traj(haidian_traj_id, cache_traj_haidian, haidian_traj_file)
                        cache_traj_haidian = []
                        haidian_traj_id += 1
                        mode = 2
                    else:
                        # 输出现在找到的轨迹
                        output_traj(haidian_traj_id, cache_traj_haidian, haidian_traj_file)
                        cache_traj_haidian = []
                        haidian_traj_id += 1
                        mode = 0
                else:
                    # mode == 2
                    if haidian_poly.intersects(gps_point):
                        # 这个点跑到海淀区里去了
                        cache_traj_haidian.append((lon, lat, timestamp))
                        # 输出之前找到的朝阳区轨迹
                        output_traj(chaoyang_traj_id, cache_traj_chaoyang, chaoyang_traj_file)
                        cache_traj_chaoyang = []
                        chaoyang_traj_id += 1
                        mode = 1
                    elif chaoyang_poly.intersects(gps_point):
                        # 还在朝阳区
                        cache_traj_chaoyang.append((lon, lat, timestamp))
                    else:
                        output_traj(chaoyang_traj_id, cache_traj_chaoyang, chaoyang_traj_file)
                        cache_traj_chaoyang = []
                        chaoyang_traj_id += 1
                        mode = 0

haidian_traj_file.close()
chaoyang_traj_file.close()
