from tqdm import tqdm
import json
from datetime import datetime, timedelta
import pandas as pd
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
                    default=False, help='whether save the trained model')
parser.add_argument('--dataset_name', type=str,
                    default='Xian')
args = parser.parse_args()
local = args.local
dataset_name = args.dataset_name

if local:
    data_root = '../data/'
else:
    data_root = '/mnt/data/jwj/'

if dataset_name == 'BJ_Taxi':
    mm_input_filelist = ['/mnt/data/jwj/BJ_Taxi/chaoyang_gps_traj_201511.csv',
                         '/mnt/data/jwj/BJ_Taxi/haidian_gps_traj_201511.csv']
    mm_output_filelist = ['/mnt/data/jwj/BJ_Taxi/chaoyang_traj_mm_201511.txt',
                          '/mnt/data/jwj/BJ_Taxi/haidian_traj_mm_201511.txt']
    output_filelist = ['/mnt/data/jwj/BJ_Taxi/chaoyang_traj_mm_processed.csv',
                       '/mnt/data/jwj/BJ_Taxi/haidian_traj_mm_processed.csv']
    with open('/mnt/data/jwj/BJ_Taxi/region_road_dict.json', 'r') as f:
        region_road_dict = json.load(f)
        region_A_road_list = set(region_road_dict['chaoyang_road_list'])
        region_B_road_list = set(region_road_dict['haidian_road_list'])
    # 读取路段信息数据
    rid_info = pd.read_csv('/mnt/data/jwj/BJ_Taxi/bj_roadmap.geo')
    rid_length = {}
    for index, row in rid_info.iterrows():
        rid = row['geo_id']
        rid_length[rid] = row['length']
elif dataset_name == 'Xian':
    # dataset_name == 'Xian'
    mm_input_filelist = [os.path.join(data_root, dataset_name, 'xianshi_partA_gps_traj.csv'),
                         os.path.join(data_root, dataset_name, 'xianshi_partB_gps_traj.csv')]
    mm_output_filelist = [os.path.join(data_root, dataset_name, 'xianshi_partA_mm_traj.txt'),
                          os.path.join(data_root, dataset_name, 'xianshi_partB_mm_traj.txt')]
    output_filelist = [os.path.join(data_root, dataset_name, 'xianshi_partA_traj_mm_processed.csv'),
                       os.path.join(data_root, dataset_name, 'xianshi_partB_traj_mm_processed.csv')]
    with open(os.path.join(data_root, dataset_name, 'region_road_dict.json'), 'r') as f:
        region_road_dict = json.load(f)
        region_A_road_list = set(region_road_dict['region_A_road_list'])
        region_B_road_list = set(region_road_dict['region_B_road_list'])
    rid_info = pd.read_csv(os.path.join(data_root, dataset_name, 'xian.geo'))
    rid_length = {}
    for index, row in rid_info.iterrows():
        rid = row['geo_id']
        rid_length[rid] = row['length']
else:
    # Chengdu
    mm_input_filelist = [os.path.join(data_root, dataset_name, 'chengdushi_partA_gps_traj.csv'),
                         os.path.join(data_root, dataset_name, 'chengdushi_partB_gps_traj.csv')]
    mm_output_filelist = [os.path.join(data_root, dataset_name, 'chengdushi_partA_mm_traj.txt'),
                          os.path.join(data_root, dataset_name, 'chengdushi_partB_mm_traj.txt')]
    output_filelist = [os.path.join(data_root, dataset_name, 'chengdushi_partA_traj_mm_processed.csv'),
                       os.path.join(data_root, dataset_name, 'chengdushi_partB_traj_mm_processed.csv')]
    with open(os.path.join(data_root, dataset_name, 'region_road_dict.json'), 'r') as f:
        region_road_dict = json.load(f)
        region_A_road_list = set(region_road_dict['region_A_road_list'])
        region_B_road_list = set(region_road_dict['region_B_road_list'])
    rid_info = pd.read_csv(os.path.join(data_root, dataset_name, 'chengdu.geo'))
    rid_length = {}
    for index, row in rid_info.iterrows():
        rid = row['geo_id']
        rid_length[rid] = row['length']

# 从 mm_input 文件中获取每条轨迹的时间序列
# 这边还需要记录一下时间
# 好现在根据 mm_id2time 以及 mm_id2traj 把路网匹配后的结果也搞定了

min_len = 3
max_len = 100


# 环路检查
def check_loop(cpath):
    visited = set()
    for rid in cpath:
        if rid not in visited:
            visited.add(rid)
        else:
            # 环路了
            return True
    return False


def process_mm_traj(mm_id, cpath, opath):
    if not isinstance(cpath, str):
        error_id['cpath_is_not_str'].append(mm_id)
        return None, None
    try:
        cpath = [int(i) for i in cpath.split(',')]
        opath = [int(i) for i in opath.split(',')]
        if check_loop(cpath=cpath):
            # 不要环路
            error_id['loop'].append(mm_id)
            return None, None
        if len(cpath) < min_len:
            error_id['too_short'].append(mm_id)
            return None, None
        if len(cpath) > max_len:
            error_id['too_long'].append(mm_id)
        true_time_set = mm_id2time[mm_id]
        if len(true_time_set) != len(opath):
            # 舍弃这些轨迹，这可能是由于部分跨天的轨迹，在我采样的时候被截断了
            error_id['opath_not_match_time'].append(mm_id)
            return None, None
        rid2time = {}
        path_with_time = []
        for index, rid in enumerate(opath):
            # 如果一个路段有多个时间，只取第一个时间作为路段的进入时间
            if rid not in rid2time:
                rid2time[rid] = datetime.fromtimestamp(true_time_set[index])
                path_with_time.append(rid)
        # 第一个 rid 和最后一个 rid 一定是有时间的
        if cpath[0] != path_with_time[0] or cpath[-1] != path_with_time[-1]:
            print('first rid and last rid do not have time, pad fail! mm_id: ', mm_id)
            error_id['first_or_end_no_time'].append(mm_id)
            return None, None
        # 统计轨迹的行驶距离，用于计算速度并进一步预测时间
        # 注意因为时间是进入路段，所以这里距离应该是进入路段之前行驶的距离
        path_length = [0]  # 记录每个 rid 的进入时车辆行驶路程
        path_with_time_length = [0]  # 只记录有时间的 rid 的进入时车辆行驶路程
        for i in range(1, len(cpath)):
            path_length.append(path_length[-1] + rid_length[cpath[i - 1]])
            if cpath[i] in rid2time:
                path_with_time_length.append(path_length[-1])
        # 准备时间
        pre_time = rid2time[cpath[0]]
        pre_index = 0
        next_time = rid2time[path_with_time[1]]
        next_index = 1
        time_list = [pre_time.strftime('%Y-%m-%dT%H:%M:%SZ')]
        # 计算两个有时间路段间平均速度，使用该速度去预估没有时间的路段
        time_off = (next_time - pre_time).seconds
        if time_off == 0:
            # 神秘情况，直接弃用
            error_id['no_time_off'].append(mm_id)
            return None, None
        speed_avg = (path_with_time_length[next_index] - path_with_time_length[pre_index]) / time_off  # 单位 m/s
        for i in range(1, len(cpath)):
            # 检查 i 路段是否有时间
            rid = cpath[i]
            if rid == path_with_time[next_index]:
                # 表明 i 路段是有时间的
                # 切换 pre 与 next，重新计算平均速度
                pre_time = next_time
                pre_index = next_index
                if next_index != len(path_with_time) - 1:
                    # 表明不是最后一个 rid
                    next_index += 1
                    next_time = rid2time[path_with_time[next_index]]
                    time_off = (next_time - pre_time).seconds
                    if time_off == 0:
                        # 神秘情况，直接弃用
                        error_id['no_time_off'].append(mm_id)
                        return None, None
                    speed_avg = (path_with_time_length[next_index] - path_with_time_length[pre_index]) / time_off
                # 将 i 的时间放入 time_list 中
                time_list.append(pre_time.strftime('%Y-%m-%dT%H:%M:%SZ'))
            else:
                # 用 speed_avg 来预估时间
                travel_length = path_length[i] - path_with_time_length[pre_index]
                travel_time = travel_length / speed_avg
                rid_time = pre_time + timedelta(seconds=round(travel_time))
                time_list.append(rid_time.strftime('%Y-%m-%dT%H:%M:%SZ'))
        # 返回轨迹 mm_id、路段序列、时间序列
        return cpath, time_list
    except:
        # 不知道还有什么 bug
        # 直接跳过
        error_id['unknown'].append(mm_id)
        return None, None


error_id = {
    'cpath_is_not_str': [],
    'loop': [],
    'too_short': [],
    'too_long': [],
    'opath_not_match_time': [],
    'first_or_end_no_time': [],
    'no_time_off': [],
    'unknown': [],
    'outside': []
}

for index, mm_input_file in enumerate(mm_input_filelist):
    print('process file {}'.format(index))
    valid_cnt = 0
    mm_id2time = {}  # 记录 mm 的原始时间序列
    with open(mm_input_file, 'r') as f:
        f.readline()
        for line in tqdm(f, desc='process mm input file'):
            items = line.split(';')
            mm_id = items[0].strip()
            if mm_id not in mm_id2time:
                mm_id2time[str(mm_id)] = [int(items[3])]
            else:
                mm_id2time[str(mm_id)].append(int(items[3]))

    output = open(output_filelist[index], 'w')
    mm_output_file = mm_output_filelist[index]
    if index == 0:
        region_road_list = region_A_road_list
    else:
        region_road_list = region_B_road_list
    with open(mm_output_file, 'r') as f:
        head = f.readline()
        output.write('traj_id,rid_list,time_list\n')
        for line in tqdm(f, desc='process mm output file'):
            items = line.split(';')
            mm_id = items[0].strip()
            opath = items[1].strip()
            cpath = items[6].strip()
            if mm_id in mm_id2time:
                # 是被筛选的轨迹
                path_list, time_list = process_mm_traj(mm_id, cpath, opath)
                if path_list is not None:
                    # 检查是否有在区域外的
                    is_outside = False
                    for road in path_list:
                        if road not in region_road_list:
                            is_outside = True
                            error_id['outside'].append(mm_id)
                            break
                    if not is_outside:
                        valid_cnt += 1
                        path_list_str = ",".join([str(i) for i in path_list])
                        time_list_str = ",".join(time_list)
                        output.write('{},\"{}\",\"{}\"\n'.format(mm_id, path_list_str, time_list_str))
    output.close()
    for key in error_id:
        print('{} cnt: {}'.format(key, len(error_id[key])))
    # 保存
    with open(os.path.join(data_root, dataset_name, 'error_id_{}.json'.format(index)), 'w') as f:
        json.dump(error_id, f)
    error_id = {
        'cpath_is_not_str': [],
        'loop': [],
        'too_short': [],
        'too_long': [],
        'opath_not_match_time': [],
        'first_or_end_no_time': [],
        'no_time_off': [],
        'unknown': [],
        'outside': []
    }
    print('valid cnt', valid_cnt)
