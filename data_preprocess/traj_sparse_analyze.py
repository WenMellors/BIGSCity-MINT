# 检查轨迹数据的稀疏程度
# 我们需要实现一个可视化，将整个北京的网格化流量可视化出来
import numpy as np
from tqdm import tqdm
from datetime import datetime

traffic_state_mx = np.load('output_res/traffic_state_mx_spec.npy')
time_slot_num = (8 * 60) // 5  # 抽样查看早上八点

print(traffic_state_mx[time_slot_num].max())
# 22
print(traffic_state_mx[time_slot_num].sum())
# 24283
# 这么看下来或许还行，同一时刻有效的出租车数量在 24283 个（但我们也需要考虑 mm 之后的失败数据率）

# 下面找一下之前 mm 过后的数据，同一时间槽行驶在相同道路上的
road_traffic_state = np.zeros((288, 40306), dtype=np.int32)


def parse_time(time_in):
    """
    将 json 中 time_format 格式的 time 转化为 local datatime
    """
    date = datetime.strptime(time_in, '%Y-%m-%dT%H:%M:%SZ')  # 这是 UTC 时间
    return date


with open('E:\\Fine-Grained Trajectory Generation\\data\\201511_week1_mm.dyna', 'r') as f:
    f.readline() # 读取第一行
    for line in tqdm(f.readlines()):
        items = line.split(',')
        time = items[2]
        date = parse_time(time)
        if date.day == 2:
            time_slot_num = (date.hour * 60 + date.minute) // 5
            road = int(items[5])
            road_traffic_state[time_slot_num][road] += 1

print(road_traffic_state[(8 * 60) // 5].max())
# 10
print(road_traffic_state[(8 * 60) // 5].sum())
# 14723
np.save('output_res/road_traffic_state.npy', road_traffic_state)
