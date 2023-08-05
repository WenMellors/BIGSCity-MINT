# 模仿学习训练 Agent Policy
import pandas as pd
from model.Agent import AgentPolicyNetV3, AgentManager
from util.logger import get_logger
from util.parser import str2bool
from datetime import datetime
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
parser.add_argument('--save_filename', type=str, default='agent_imitate_learning_v3_drop.pt')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
local = args.local
dataset_name = args.dataset_name
debug = args.debug
region_name = args.region_name


if local:
    data_root = './data/'
else:
    data_root = '/mnt/data/jwj/'

# 训练相关参数
if debug:
    max_epoch = 2
else:
    max_epoch = 20
batch_size = 64  # 数个 agent 一起组成一个 batch 投入决策网络中
device = args.device
torch.cuda.set_device(device)
train_rate = 0.9
learning_rate = 0.0005
weight_decay = 0.00001
lr_patience = 2
lr_decay_ratio = 0.1
save_folder = './save/{}'.format(dataset_name)
save_file_name = args.save_filename
temp_folder = './temp/marl/{}/'.format(int(datetime.now().timestamp()))
early_stop_lr = 1e-6
train = True
clip = 5.0
if dataset_name == 'BJ_Taxi':
    # 数据集的大小
    road_num = 40306  # 这里是全部北京四环内的道路数，因为我们没有对道路编号进行重编码，所以就保持这样吧
    time_size = 2880
elif dataset_name == 'Porto_Taxi':
    # dataset_name == 'Porto_Taxi'
    road_num = 11095
    time_size = 2880
elif dataset_name == 'Xian':
    # dataset_name == 'Xian'
    road_num = 17378
    time_size = 2880
else:
    assert dataset_name == 'Chengdu'
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

logger = get_logger(name='imitate_learning_{}'.format(dataset_name))
logger.info('read data')
if dataset_name == 'BJ_Taxi':
    if region_name == 'partA':
        train_data = pd.read_csv(os.path.join(data_root, dataset_name, 'beijing_partA_input_train.csv'))
        test_data = pd.read_csv(os.path.join(data_root, dataset_name, 'beijing_partA_input_test.csv'))
    else:
        train_data = pd.read_csv(os.path.join(data_root, dataset_name, 'beijing_partB_input_train.csv'))
        test_data = pd.read_csv(os.path.join(data_root, dataset_name, 'beijing_partB_input_test.csv'))
elif dataset_name == 'Xian':
    # dataset_name == 'Xian'
    if region_name == 'partA':
        train_data = pd.read_csv(os.path.join(data_root, dataset_name, 'xianshi_partA_input_train.csv'))
        test_data = pd.read_csv(os.path.join(data_root, dataset_name, 'xianshi_partA_input_test.csv'))
    else:
        train_data = pd.read_csv(os.path.join(data_root, dataset_name, 'xianshi_partB_input_train.csv'))
        test_data = pd.read_csv(os.path.join(data_root, dataset_name, 'xianshi_partB_input_test.csv'))
else:
    assert dataset_name == 'Chengdu'
    if region_name == 'partA':
        train_data = pd.read_csv(os.path.join(data_root, dataset_name, 'chengdushi_partA_input_train.csv'))
        test_data = pd.read_csv(os.path.join(data_root, dataset_name, 'chengdushi_partA_input_test.csv'))
    else:
        train_data = pd.read_csv(os.path.join(data_root, dataset_name, 'chengdushi_partB_input_train.csv'))
        test_data = pd.read_csv(os.path.join(data_root, dataset_name, 'chengdushi_partB_input_test.csv'))

# 训练集再划分为验证和训练的
total_data_num = train_data.shape[0]
# 划分数据集
if debug:
    train_num = 16
    total_data_num = 24
else:
    train_num = int(total_data_num * train_rate)
# 加载模型
agent_policy = AgentPolicyNetV3(policy_config=policy_config).to(device)
logger.info('init agent policy net')
logger.info(agent_policy)

optimizer = torch.optim.Adam(agent_policy.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=lr_patience,
                                                          factor=lr_decay_ratio)
loss_func = torch.nn.CrossEntropyLoss().to(device)
# 加载仿真管理器
with open(os.path.join(data_root, dataset_name, 'road_surrounding_list.json'), 'r') as f:
    road_surrounding_list = json.load(f)

with open(os.path.join(data_root, dataset_name, 'road_candidate_list.json'), 'r') as f:
    road_candidate_list = json.load(f)

agent_manager = AgentManager(road_surrounding_list=road_surrounding_list, road_candidate_list=road_candidate_list,
                             preference_size=policy_config['preference_size'], info_size=policy_config['info_size'],
                             num_layers=policy_config['SeqMovingStateNet']['n_layers'],
                             hidden_size=policy_config['hidden_size'], device=device, dataset_name=dataset_name)

# 开始训练
if debug:
    torch.autograd.set_detect_anomaly(True)
if train:
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    metrics = []
    for epoch in range(max_epoch):
        # train
        logger.info('start train epoch {}'.format(epoch))
        agent_policy.train(True)
        train_loss = 0
        for index in tqdm(range(train_num)):
            # 取出这一轮的仿真输入信息
            input_row = train_data.iloc[index]
            agent_id_list = [int(x) for x in input_row['traj_id_list'].split(',')]
            road_id_list = [int(x) for x in input_row['road_id_list'].split(',')]
            des_id_list = [int(x) for x in input_row['des_id_list'].split(',')]
            # train 的目标
            target_id_list = [int(x) for x in input_row['target_id_list'].split(',')]
            target_id_list = torch.LongTensor(target_id_list).to(device)
            input_data = agent_manager.imitate_learning_organize_input(current_time=input_row['time_code'],
                                                                       agent_id_list=agent_id_list,
                                                                       road_id_list=road_id_list,
                                                                       des_id_list=des_id_list)
            # 按照 batch_size 数量进行组织，投入策略网络中进行训练
            input_data_len = len(input_data[0])
            start_input_index = 0
            assert target_id_list.shape[0] == input_data_len
            batch_out_info = []
            batch_next_history_h = []
            batch_next_h = []
            batch_next_c = []
            batch_candidate_prob = []
            optimizer.zero_grad()  # 全部算完再 backward
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
                candidate_prob, out_info, next_history_h,\
                    next_h, next_c, agent_state = agent_policy(batch_input_loc, batch_input_time, batch_input_des,
                                                               batch_input_inter_info, batch_input_prefer,
                                                               batch_input_candidate_mask, batch_input_history_h,
                                                               batch_input_current_h, batch_input_current_c)
                # 开始下一个循环
                start_input_index = next_input_index
                batch_candidate_prob.extend(candidate_prob)
                batch_out_info.append(out_info)
                batch_next_history_h.extend(next_history_h)
                batch_next_h.extend(next_h)
                batch_next_c.extend(next_c)
            loss = loss_func(batch_candidate_prob[0].unsqueeze(0), target_id_list[0].unsqueeze(0))
            for i in range(1, len(batch_candidate_prob)):
                loss += loss_func(batch_candidate_prob[i].unsqueeze(0), target_id_list[i].unsqueeze(0))
            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(agent_policy.parameters(), clip)
            optimizer.step()
            # 当前仿真步，所有 agent 都决策完了，那么可以开始更新每个 agent 的状态了
            batch_out_info = torch.cat(batch_out_info, dim=0)  # (agent_num, info_size)
            agent_manager.imitate_learning_update(agent_id_list, batch_out_info, batch_next_history_h, batch_next_h,
                                                  batch_next_c, des_id_list, input_data[5],
                                                  target_id_list.tolist())
        # val
        val_hit = 0
        val_num = 0
        agent_policy.train(False)
        for index in tqdm(range(train_num, total_data_num)):
            with torch.no_grad():
                # 取出这一轮的仿真输入信息
                input_row = train_data.iloc[index]
                agent_id_list = [int(x) for x in input_row['traj_id_list'].split(',')]
                road_id_list = [int(x) for x in input_row['road_id_list'].split(',')]
                des_id_list = [int(x) for x in input_row['des_id_list'].split(',')]
                # train 的目标
                target_id_list = [int(x) for x in input_row['target_id_list'].split(',')]
                input_data = agent_manager.imitate_learning_organize_input(current_time=input_row['time_code'],
                                                                           agent_id_list=agent_id_list,
                                                                           road_id_list=road_id_list,
                                                                           des_id_list=des_id_list)
                # 按照 batch_size 数量进行组织，投入策略网络中进行训练
                input_data_len = len(input_data[0])
                start_input_index = 0
                assert len(target_id_list) == input_data_len
                batch_out_info = []
                batch_next_history_h = []
                batch_next_h = []
                batch_next_c = []
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
                        if arg_max_candidate == target_id_list[start_input_index + i]:
                            val_hit += 1
                        val_num += 1
                    # 开始下一个循环
                    start_input_index = next_input_index
                    batch_out_info.append(out_info)
                    batch_next_history_h.extend(next_history_h)
                    batch_next_h.extend(next_h)
                    batch_next_c.extend(next_c)
                # 当前仿真步，所有 agent 都决策完了，那么可以开始更新每个 agent 的状态了
                batch_out_info = torch.cat(batch_out_info, dim=0)  # (agent_num, info_size)
                agent_manager.imitate_learning_update(agent_id_list, batch_out_info, batch_next_history_h, batch_next_h,
                                                      batch_next_c, des_id_list, input_data[5],
                                                      target_id_list)
        val_ac = val_hit / val_num
        metrics.append(val_ac)
        lr_scheduler.step(val_ac)
        # store temp model
        torch.save(agent_policy.state_dict(), os.path.join(temp_folder, 'agent_policy_{}.pt'.format(epoch)))
        lr = optimizer.param_groups[0]['lr']
        logger.info('==> Train Epoch {}: Train Loss {:.6f}, val ac {}, lr {}'.format(epoch, train_loss, val_ac, lr))
        if lr < early_stop_lr:
            logger.info('early stop')
            break
        if epoch != max_epoch - 1:
            # 还会重头再训练一次，需要重置 agent_manager
            # 需要重置 agent_manager
            agent_manager.imitate_learning_reset()
    # load best epoch
    best_epoch = np.argmax(metrics)
    load_temp_file = 'agent_policy_{}.pt'.format(best_epoch)
    logger.info('load best from {}'.format(best_epoch))
    agent_policy.load_state_dict(torch.load(os.path.join(temp_folder, load_temp_file)))

else:
    agent_policy.load_state_dict(torch.load(os.path.join(save_folder, save_file_name), map_location=device))
# 开始评估
agent_policy.train(False)
test_hit = 0
test_total_num = 0
agent_manager.imitate_learning_reset()
for index, input_row in tqdm(test_data.iterrows(), total=test_data.shape[0]):
    with torch.no_grad():
        # 取出这一轮的仿真输入信息
        agent_id_list = [int(x) for x in input_row['traj_id_list'].split(',')]
        road_id_list = [int(x) for x in input_row['road_id_list'].split(',')]
        des_id_list = [int(x) for x in input_row['des_id_list'].split(',')]
        # train 的目标
        target_id_list = [int(x) for x in input_row['target_id_list'].split(',')]
        input_data = agent_manager.imitate_learning_organize_input(current_time=input_row['time_code'],
                                                                   agent_id_list=agent_id_list,
                                                                   road_id_list=road_id_list,
                                                                   des_id_list=des_id_list)
        # 按照 batch_size 数量进行组织，投入策略网络中进行训练
        input_data_len = len(input_data[0])
        start_input_index = 0
        assert len(target_id_list) == input_data_len
        batch_out_info = []
        batch_next_history_h = []
        batch_next_h = []
        batch_next_c = []
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
                if arg_max_candidate == target_id_list[start_input_index + i]:
                    test_hit += 1
                test_total_num += 1
            # 开始下一个循环
            start_input_index = next_input_index
            batch_out_info.append(out_info)
            batch_next_history_h.extend(next_history_h)
            batch_next_h.extend(next_h)
            batch_next_c.extend(next_c)
        # 当前仿真步，所有 agent 都决策完了，那么可以开始更新每个 agent 的状态了
        batch_out_info = torch.cat(batch_out_info, dim=0)  # (agent_num, info_size)
        agent_manager.imitate_learning_update(agent_id_list, batch_out_info, batch_next_history_h, batch_next_h,
                                              batch_next_c, des_id_list, input_data[5],
                                              target_id_list)
        if debug and index == 5:
            break
test_ac = test_hit / test_total_num
logger.info('==> Test Result: test ac {}'.format(test_ac))
# 保存模型
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
torch.save(agent_policy.state_dict(), os.path.join(save_folder, save_file_name))
# 删除 temp 文件
for rt, dirs, files in os.walk(temp_folder):
    for name in files:
        remove_path = os.path.join(rt, name)
        os.remove(remove_path)
