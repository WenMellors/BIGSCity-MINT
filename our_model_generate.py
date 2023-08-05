# 模仿学习训练 Agent Policy
import pandas as pd
from model.Agent import AgentPolicyNetV3, AgentManager
from util.logger import get_logger
from util.parser import str2bool
from tqdm import tqdm
import torch
import os
import json
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--local', type=str2bool, default=False)
parser.add_argument('--dataset_name', type=str, default='BJ_Taxi')
parser.add_argument('--region_name', type=str, default='partA')
parser.add_argument('--debug', type=str2bool, default=False)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--model_version', type=str, default='agent_actor_critic_td')
args = parser.parse_args()
local = args.local
dataset_name = args.dataset_name
debug = args.debug
region_name = args.region_name

if local:
    data_root = './data/'
else:
    data_root = '/mnt/data/jwj/'

batch_size = 64  # 数个 agent 一起组成一个 batch 投入决策网络中
device = args.device
torch.cuda.set_device(device)
model_version = args.model_version
save_file = './save/{}/{}.pt'.format(dataset_name, model_version)
if dataset_name == 'BJ_Taxi':
    output_file = open(os.path.join(data_root, dataset_name, '{}_{}_generate.csv'.format(model_version, region_name)), 'w')
    # 数据集的大小
    road_num = 40306  # 这里是全部北京四环内的道路数，因为我们没有对道路编号进行重编码，所以就保持这样吧
    time_size = 2880
    road_pad = road_num
    time_pad = time_size
elif dataset_name == 'Porto_Taxi':
    output_file = open(os.path.join(data_root, dataset_name, '{}_generate.csv'.format(model_version)), 'w')
    road_num = 11095
    time_size = 2880
    road_pad = road_num
    time_pad = time_size
elif dataset_name == 'Xian':
    # Xian
    output_file = open(os.path.join(data_root, dataset_name, '{}_{}_generate.csv'.format(model_version, region_name)), 'w')
    road_num = 17378
    time_size = 2880
    road_pad = road_num
    time_pad = time_size
else:
    assert dataset_name == 'Chengdu'
    output_file = open(os.path.join(data_root, dataset_name, '{}_{}_generate.csv'.format(model_version, region_name)),
                       'w')
    road_num = 28823
    time_size = 2880
    road_pad = road_num
    time_pad = time_size

if dataset_name == 'BJ_Taxi' or dataset_name == 'Porto_Taxi':
    # Agent 策略网络的参数
    policy_config = {
        'road_num': road_num + 1,
        'road_pad': road_pad,
        'road_emb_size': 256,
        'time_num': time_size + 1,
        'time_pad': time_pad,
        'time_emb_size': 64,
        'device': device,
        'preference_size': 16,
        'info_size': 128,
        'hidden_size': 256,
        'head_num': 4,
        'SeqMovingStateNet': {
            'device': device,
            'n_layers': 2
        },
        'dropout_input_p': 0.2,
        'dropout_hidden_p': 0.5
    }
else:
    assert dataset_name == 'Xian' or dataset_name == 'Chengdu'
    policy_config = {
        'road_num': road_num + 1,
        'road_pad': road_pad,
        'road_emb_size': 128,
        'time_num': time_size + 1,
        'time_pad': time_pad,
        'time_emb_size': 32,
        'device': device,
        'preference_size': 8,
        'info_size': 64,
        'hidden_size': 128,
        'head_num': 4,
        'SeqMovingStateNet': {
            'device': device,
            'n_layers': 2
        },
        'dropout_input_p': 0.2,
        'dropout_hidden_p': 0.5
    }

policy_config['SeqMovingStateNet']['input_size'] = policy_config['hidden_size']
policy_config['SeqMovingStateNet']['hidden_size'] = policy_config['hidden_size']

logger = get_logger(name='{}_generate_{}'.format(model_version, dataset_name))
logger.info('read data')
if dataset_name == 'BJ_Taxi':
    # 目前不会实装海淀区
    if region_name == 'partA':
        data = pd.read_csv(os.path.join(data_root, dataset_name, 'beijing_partA_input_test.csv'))
    else:
        data = pd.read_csv(os.path.join(data_root, dataset_name, 'beijing_partB_input_test.csv'))
elif dataset_name == 'Porto_Taxi':
    data = pd.read_csv(os.path.join(data_root, dataset_name, 'porto_input_test.csv'))
elif dataset_name == 'Xian':
    # dataset_name == 'Xian'
    if region_name == 'partA':
        data = pd.read_csv(os.path.join(data_root, dataset_name, 'xianshi_partA_input_test.csv'))
    else:
        data = pd.read_csv(os.path.join(data_root, dataset_name, 'xianshi_partB_input_test.csv'))
else:
    assert dataset_name == 'Chengdu'
    if region_name == 'partA':
        data = pd.read_csv(os.path.join(data_root, dataset_name, 'chengdushi_partA_input_test.csv'))
    else:
        data = pd.read_csv(os.path.join(data_root, dataset_name, 'chengdushi_partB_input_test.csv'))

total_data_num = data.shape[0]
# 加载模型
agent_policy = AgentPolicyNetV3(policy_config=policy_config).to(device)
agent_policy.load_state_dict(torch.load(save_file, map_location=device))


# 加载仿真管理器
with open(os.path.join(data_root, dataset_name, 'road_surrounding_list.json'), 'r') as f:
    road_surrounding_list = json.load(f)

with open(os.path.join(data_root, dataset_name, 'road_candidate_list.json'), 'r') as f:
    road_candidate_list = json.load(f)

with open(os.path.join(data_root, dataset_name, 'region_road_dict.json'), 'r') as f:
    region_road_dict = json.load(f)
if region_name == 'partA':
    region_road_list = region_road_dict['region_A_road_list']
else:
    region_road_list = region_road_dict['region_B_road_list']

agent_manager = AgentManager(road_surrounding_list=road_surrounding_list, road_candidate_list=road_candidate_list,
                             preference_size=policy_config['preference_size'], info_size=policy_config['info_size'],
                             num_layers=policy_config['SeqMovingStateNet']['n_layers'],
                             hidden_size=policy_config['hidden_size'], region_road_list=region_road_list, device=device,
                             dataset_name=dataset_name)

# 开始生成
agent_policy.train(False)
output_file.write('traj_id,rid_list,time_list\n')
start_time_code = data.iloc[0]['time_code']
if dataset_name == 'BJ_Taxi':
    end_time_code = 30 * 1440
elif dataset_name == 'Porto_Taxi':
    end_time_code = 366 * 60 * 24
else:
    # Xian
    assert dataset_name == 'Xian' or dataset_name == 'Chengdu'
    end_time_code = 31 * 60 * 24

data_index = 0
for time_code in tqdm(range(start_time_code, end_time_code)):
    with torch.no_grad():
        # 可能这个时间戳没有输入，但是环境中还是有可能有车在跑
        if data_index < data.shape[0] and time_code == data.iloc[data_index]['time_code']:
            # 有输入fix
            # 取出这一轮的仿真输入信息
            input_row = data.iloc[data_index]
            agent_id_list = [int(x) for x in input_row['traj_id_list'].split(',')]
            road_id_list = [int(x) for x in input_row['road_id_list'].split(',')]
            des_id_list = [int(x) for x in input_row['des_id_list'].split(',')]
            data_index += 1
        else:
            agent_id_list = []
            road_id_list = []
            des_id_list = []
        env_agent_id_list, input_data = agent_manager.simulate_organize_input(current_time=time_code,
                                                                              agent_id_list=agent_id_list,
                                                                              road_id_list=road_id_list,
                                                                              des_id_list=des_id_list)
        # 按照 batch_size 数量进行组织，投入策略网络中进行训练
        input_data_len = len(input_data[0])
        if input_data_len == 0:
            # 环境中没有在跑的 agent 了
            continue
        start_input_index = 0
        batch_out_info = []
        batch_next_history_h = []
        batch_next_h = []
        batch_next_c = []
        batch_next_step_list = []
        while start_input_index < input_data_len:
            next_input_index = start_input_index + batch_size
            # 获取数据
            batch_input_loc = input_data[0][start_input_index:next_input_index]
            batch_input_time = input_data[1][start_input_index:next_input_index]
            batch_input_des = input_data[2][start_input_index:next_input_index]
            batch_input_inter_info = input_data[3][start_input_index:next_input_index]
            batch_input_prefer = input_data[4][start_input_index:next_input_index]
            batch_input_candidate_mask = input_data[5][start_input_index:next_input_index]
            batch_input_history_h = input_data[6][start_input_index:next_input_index]
            batch_input_current_h = input_data[7][start_input_index:next_input_index]
            batch_input_current_c = input_data[8][start_input_index:next_input_index]
            # 该转成 tensor 的转
            batch_input_loc = torch.LongTensor(batch_input_loc).to(device)
            batch_input_time = torch.LongTensor(batch_input_time).to(device)
            batch_input_des = torch.LongTensor(batch_input_des).to(device)
            batch_input_prefer = torch.cat(batch_input_prefer, dim=0).to(device)
            batch_input_current_h = torch.cat(batch_input_current_h, dim=1).to(device)
            batch_input_current_c = torch.cat(batch_input_current_c, dim=1).to(device)
            # 输入策略网络
            candidate_prob, out_info, next_history_h, \
            next_h, next_c, agent_state = agent_policy(batch_input_loc, batch_input_time, batch_input_des,
                                                       batch_input_inter_info, batch_input_prefer,
                                                       batch_input_candidate_mask, batch_input_history_h,
                                                       batch_input_current_h, batch_input_current_c)
            # 验证下一跳是否命中
            for i in range(len(candidate_prob)):
                arg_max_candidate = torch.softmax(candidate_prob[i], dim=0).argmax().item()
                batch_next_step_list.append(batch_input_candidate_mask[i][arg_max_candidate])
            # 开始下一个循环
            start_input_index = next_input_index
            batch_out_info.append(out_info)
            batch_next_history_h.extend(next_history_h)
            batch_next_h.extend(next_h)
            batch_next_c.extend(next_c)
        # 当前仿真步，所有 agent 都决策完了，那么可以开始更新每个 agent 的状态了
        batch_out_info = torch.cat(batch_out_info, dim=0)  # (agent_num, info_size)
        agent_manager.simulate_update(env_agent_id_list, time_code, batch_out_info, batch_next_history_h,
                                      batch_next_h, batch_next_c, batch_next_step_list, output_file, max_step=100,
                                      in_region=True)
        if debug and time_code == start_time_code + 10:
            break
# 还有可能有没仿真完的，全部都输出了吧，那就
# 因为已经出了仿真的时间范围了
agent_manager.simulate_end(output_file)
output_file.close()
