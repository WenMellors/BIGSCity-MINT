# 根据 road_adjacent_list.json 统计一共有多少个路口
# 依据：一个路段只有一个进入入口与一个流出入口
import json

with open('../data/BJ_Taxi/road_candidate_list.json', 'r') as f:
    road_adjacent_list = json.load(f)

with open('../data/BJ_Taxi/region_road_dict.json', 'r') as f:
    region_road_dict = json.load(f)

region_road_list = region_road_dict['region_B_road_list']
print(len(region_road_list))


road_in_node_dict = {}  # 记录每个路段对应的入口路口 ID
road_out_node_dict = {}  # 记录每个路段对应的出口路口 ID
in_node_dict = {}  # 记录路口 ID 对应的入口路段 ID 集合
out_node_dict = {}  # 记录路口 ID 对应的出口路段 ID 集合

node_id = 0
for out_road in road_adjacent_list:
    if int(out_road) not in region_road_list:
        continue
    for in_road in road_adjacent_list[out_road]:
        if int(in_road) not in region_road_list:
            continue
        # 查询当前两个路段是否已经有对应的路口 ID
        if int(out_road) in road_out_node_dict and int(in_road) in road_in_node_dict:
            assert road_out_node_dict[int(out_road)] == road_in_node_dict[int(in_road)]
        elif int(out_road) in road_out_node_dict:
            cur_node_id = road_out_node_dict[int(out_road)]
            road_in_node_dict[int(in_road)] = cur_node_id
        elif int(in_road) in road_in_node_dict:
            cur_node_id = road_in_node_dict[int(in_road)]
            road_out_node_dict[int(out_road)] = cur_node_id
        else:
            # 分配一个新的 node_id
            cur_node_id = node_id
            node_id += 1
            road_in_node_dict[int(in_road)] = cur_node_id
            road_out_node_dict[int(out_road)] = cur_node_id

print(node_id)
