import math
import os
from tqdm import tqdm
import numpy as np
from datetime import datetime
# 由于原始轨迹数据是 GPS 的，所以我们这里网格化来统计流量信息
# 分小时统计一个网格化的交通流量矩阵
# 首先我们需要统计整个城市的一个经纬度范围
# 通过 QGIS 我们大致以五环确定一个矩形范围 116.2030482,39.7534922 : 116.568459, 40.043512
lon_0 = 116.2030482
lat_0 = 39.7534922  # 地图最左下角的坐标（即原点坐标）
lon_1 = 116.568459
lat_1 = 40.043512
lon_range = lon_1 - lon_0  # 地图经度的跨度
lat_range = lat_1 - lat_0 # 地图纬度的跨度
img_unit = 0.001  # 这样画出来的大小，大概是 0.11 km * 0.09 km 的格子

img_width = math.ceil(lon_range / img_unit) + 1  # 图像的宽度
img_height = math.ceil(lat_range / img_unit) + 1  # 图像的高度


def gps2grid(lon, lat):
    """
    GPS 经纬度点映射为图像网格
    Args:
        lon: 经度
        lat: 纬度

    Returns:
        x, y: 映射的网格的 x 与 y 坐标
    """
    x = math.floor((lon - lon_0) / img_unit)
    y = math.floor((lat - lat_0) / img_unit)
    return x, y


debug = False
if debug:
    dataset_root = 'E:\\Continuous-Trajectory-Generation-V2\\data\\BJ_Taxi\\'
else:
    dataset_root = '/home/mellesj/public/Dataset/BeijingTaxi/'
month = '201011'
valid_record = 0
total_record = 0

traffic_state_mx = np.zeros((30, img_height, img_width), dtype=np.int64)

if month == '201511':
    # 统计 11.02 号的细粒度交通状态矩阵，来分析轨迹数据集的稀疏性问题
    traffic_state_mx_spec = np.zeros((288, img_height, img_width), dtype=np.int64)
    data_path = dataset_root + month
    file_list = os.listdir(data_path)
    for filename in tqdm(file_list, total=len(file_list)):
        if len(filename) > 4 and filename[-4:] == '.csv':
            # 是轨迹数据
            with open(os.path.join(data_path, filename), 'r') as f:
                prev_x = None
                prev_y = None
                prev_time_slot_num = None
                for line in f.readlines():
                    # 'utc', 'apc', 'lat', 'lon', 'altitude', 'heading',
                    # 'speed', 'tflag', 'vflag', 'ost', 'num', 'phone'
                    total_record += 1
                    items = line.split('|')
                    if len(items) != 12:
                        continue
                    # 判断是否是有效定位
                    if items[7] == '40000000':
                        continue
                    # 判断是否是载客
                    if items[8] == '0':
                        continue
                    # 是否在五环的区域内
                    try:
                        lat = int('0x' + items[2], 16) / 100000
                        lon = int('0x' + items[3], 16) / 100000
                    except ValueError:
                        continue
                    if lat < lat_0 or lat > lat_1 or lon < lon_0 or lon > lon_1:
                        continue
                    try:
                        time = datetime.fromtimestamp(int('0x'+items[0], 16))
                    except:
                        continue
                    # 检查时间是否合格
                    if time.year != 2015 or time.month != 11:
                        continue
                    valid_record += 1
                    # 将当前 GPS 点映射为网格编号
                    x, y = gps2grid(lon, lat)
                    time_slot_num = (time.hour * 60 + time.minute) // 5
                    if prev_x is not None:
                        if x == prev_x and y == prev_y and time_slot_num == prev_time_slot_num:
                            # 不重复计数，车辆并没有离开目前的区域
                            continue
                    traffic_state_mx[time.day - 1][y][x] += 1
                    if time.day == 2:
                        traffic_state_mx_spec[time_slot_num][y][x] += 1
                    prev_x = x
                    prev_y = y
                    prev_time_slot_num = time_slot_num
    np.save('output_res/traffic_state_mx.npy', traffic_state_mx)
    np.save('output_res/traffic_state_mx_spec.npy', traffic_state_mx_spec)
elif month == '201411':
    data_path = dataset_root + month
    file_list = os.listdir(data_path)
    for filename in tqdm(file_list, total=len(file_list)):
        if len(filename) > 4 and filename[-4:] == '.csv':
            # 是轨迹数据
            with open(os.path.join(data_path, filename), 'r', encoding='ISO-8859-1') as f:
                prev_x = None
                prev_y = None
                prev_time_slot_num = None
                for line in f.readlines():
                    # time: 3
                    # lat: 6, lon: 7, 有效: 11, 载客: 12
                    total_record += 1
                    items = line.split(',')
                    if len(items) < 12:
                        continue
                    # 判断是否是有效定位
                    if items[11] != '0':
                        continue
                    # 判断是否是载客
                    if items[12] == '0':
                        continue
                    # 是否在五环的区域内
                    try:
                        lat = int(items[6]) / 100000
                        lon = int(items[7]) / 100000
                    except ValueError:
                        continue
                    if lat < lat_0 or lat > lat_1 or lon < lon_0 or lon > lon_1:
                        continue
                    try:
                        time = datetime.fromtimestamp(int(items[3]))
                    except:
                        continue
                    # 检查时间是否合格
                    if time.year != 2014 or time.month != 11:
                        continue
                    valid_record += 1
                    # 将当前 GPS 点映射为网格编号
                    x, y = gps2grid(lon, lat)
                    time_slot_num = (time.hour * 60 + time.minute) // 5
                    if prev_x is not None:
                        if x == prev_x and y == prev_y and time_slot_num == prev_time_slot_num:
                            # 不重复计数，车辆并没有离开目前的区域
                            continue
                    traffic_state_mx[time.day - 1][y][x] += 1
                    prev_x = x
                    prev_y = y
                    prev_time_slot_num = time_slot_num
    np.save('./traffic_state_mx_{}.npy'.format(month), traffic_state_mx)
elif month == '201311':
    data_path = dataset_root + month
    file_list = os.listdir(data_path)
    for filename in tqdm(file_list, total=len(file_list)):
        if len(filename) > 4 and filename[-4:] == '.csv':
            # 是轨迹数据
            with open(os.path.join(data_path, filename), 'r', encoding='ISO-8859-1') as f:
                prev_x = None
                prev_y = None
                prev_time_slot_num = None
                for line in f.readlines():
                    # time: 3
                    # lat: 5, lon: 6, 有效: 11, 载客: 12
                    total_record += 1
                    items = line.split(',')
                    if len(items) < 12:
                        continue
                    # 判断是否是有效定位
                    if items[11] != '0':
                        continue
                    # 判断是否是载客
                    if items[12] == '0':
                        continue
                    # 是否在五环的区域内
                    try:
                        lat = int(items[5]) / 100000
                        lon = int(items[6]) / 100000
                    except ValueError:
                        continue
                    if lat < lat_0 or lat > lat_1 or lon < lon_0 or lon > lon_1:
                        continue
                    try:
                        time = datetime.fromtimestamp(int(items[3]))
                    except:
                        continue
                    # 检查时间是否合格
                    if time.year != 2013 or time.month != 11:
                        continue
                    valid_record += 1
                    # 将当前 GPS 点映射为网格编号
                    x, y = gps2grid(lon, lat)
                    time_slot_num = (time.hour * 60 + time.minute) // 5
                    if prev_x is not None:
                        if x == prev_x and y == prev_y and time_slot_num == prev_time_slot_num:
                            # 不重复计数，车辆并没有离开目前的区域
                            continue
                    traffic_state_mx[time.day - 1][y][x] += 1
                    prev_x = x
                    prev_y = y
                    prev_time_slot_num = time_slot_num
    np.save('./traffic_state_mx_{}.npy'.format(month), traffic_state_mx)
elif month == '201211':
    data_path = dataset_root + month
    file_list = os.listdir(data_path)
    for filename in tqdm(file_list, total=len(file_list)):
        if len(filename) > 4 and filename[-4:] == '.csv':
            # 是轨迹数据
            with open(os.path.join(data_path, filename), 'r', encoding='ISO-8859-1') as f:
                prev_x = None
                prev_y = None
                prev_time_slot_num = None
                for line in f.readlines():
                    # time: 3
                    # lat: 5, lon: 4
                    total_record += 1
                    items = line.split(',')
                    if len(items) < 6:
                        continue
                    # 是否在五环的区域内
                    try:
                        lat = float(items[5])
                        lon = float(items[4])
                    except ValueError:
                        continue
                    if lat < lat_0 or lat > lat_1 or lon < lon_0 or lon > lon_1:
                        continue
                    try:
                        time = datetime.strptime(items[3], "%Y%m%d%H%M%S")
                    except:
                        continue
                    # 检查时间是否合格
                    if time.year != 2012 or time.month != 11:
                        continue
                    valid_record += 1
                    # 将当前 GPS 点映射为网格编号
                    x, y = gps2grid(lon, lat)
                    time_slot_num = (time.hour * 60 + time.minute) // 5
                    if prev_x is not None:
                        if x == prev_x and y == prev_y and time_slot_num == prev_time_slot_num:
                            # 不重复计数，车辆并没有离开目前的区域
                            continue
                    traffic_state_mx[time.day - 1][y][x] += 1
                    prev_x = x
                    prev_y = y
                    prev_time_slot_num = time_slot_num
    np.save('./traffic_state_mx_{}.npy'.format(month), traffic_state_mx)
elif month == '201111':
    data_path = dataset_root + month
    file_list = os.listdir(data_path)
    for filename in tqdm(file_list, total=len(file_list)):
        if len(filename) > 4 and filename[-4:] == '.csv':
            # 是轨迹数据
            with open(os.path.join(data_path, filename), 'r', encoding='ISO-8859-1') as f:
                prev_x = None
                prev_y = None
                prev_time_slot_num = None
                for line in f.readlines():
                    # time: 3
                    # lat: 5, lon: 4
                    total_record += 1
                    items = line.split(',')
                    if len(items) < 6:
                        continue
                    # 是否在五环的区域内
                    try:
                        lat = float(items[5])
                        lon = float(items[4])
                    except ValueError:
                        continue
                    if lat < lat_0 or lat > lat_1 or lon < lon_0 or lon > lon_1:
                        continue
                    try:
                        time = datetime.strptime(items[3], "%Y%m%d%H%M%S")
                    except:
                        continue
                    # 检查时间是否合格
                    if time.year != 2011 or time.month != 11:
                        continue
                    valid_record += 1
                    # 将当前 GPS 点映射为网格编号
                    x, y = gps2grid(lon, lat)
                    time_slot_num = (time.hour * 60 + time.minute) // 5
                    if prev_x is not None:
                        if x == prev_x and y == prev_y and time_slot_num == prev_time_slot_num:
                            # 不重复计数，车辆并没有离开目前的区域
                            continue
                    traffic_state_mx[time.day - 1][y][x] += 1
                    prev_x = x
                    prev_y = y
                    prev_time_slot_num = time_slot_num
    np.save('./traffic_state_mx_{}.npy'.format(month), traffic_state_mx)
elif month == '201011':
    data_path = dataset_root + month
    file_list = os.listdir(data_path)
    for filename in tqdm(file_list, total=len(file_list)):
        if len(filename) > 4 and filename[-4:] == '.csv':
            # 是轨迹数据
            with open(os.path.join(data_path, filename), 'r', encoding='ISO-8859-1') as f:
                prev_x = None
                prev_y = None
                prev_time_slot_num = None
                for line in f.readlines():
                    # time: 3
                    # lat: 5, lon: 4
                    total_record += 1
                    items = line.split(',')
                    if len(items) < 6:
                        continue
                    # 是否在五环的区域内
                    try:
                        lat = float(items[5])
                        lon = float(items[4])
                    except ValueError:
                        continue
                    if lat < lat_0 or lat > lat_1 or lon < lon_0 or lon > lon_1:
                        continue
                    try:
                        time = datetime.strptime(items[3], "%Y%m%d%H%M%S")
                    except:
                        continue
                    # 检查时间是否合格
                    if time.year != 2010 or time.month != 11:
                        continue
                    valid_record += 1
                    # 将当前 GPS 点映射为网格编号
                    x, y = gps2grid(lon, lat)
                    time_slot_num = (time.hour * 60 + time.minute) // 5
                    if prev_x is not None:
                        if x == prev_x and y == prev_y and time_slot_num == prev_time_slot_num:
                            # 不重复计数，车辆并没有离开目前的区域
                            continue
                    traffic_state_mx[time.day - 1][y][x] += 1
                    prev_x = x
                    prev_y = y
                    prev_time_slot_num = time_slot_num
    np.save('./traffic_state_mx_{}.npy'.format(month), traffic_state_mx)

print('total record {}, valid record {}\n'.format(total_record, valid_record))


