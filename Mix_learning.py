from model.MixLearnerV2 import MixLearnerV2
from model.Agent import AgentManager, AgentPolicyNetV3
from model.EpisodePool import EpisodePool
import pandas as pd
from util.logger import get_logger
from util.parser import str2bool
import torch
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--local', type=str2bool, default=False)
parser.add_argument('--dataset_name', type=str, default='BJ_Taxi')
parser.add_argument('--region_name', type=str, default='partA')
parser.add_argument('--debug', type=str2bool, default=False)
parser.add_argument('--pretrain_filename', type=str, default='agent_imitate_learning_v3_drop.pt')
parser.add_argument('--save_agent_file', type=str, default='mix_learning_agent_policy.pt')
parser.add_argument('--save_critic_file', type=str, default='mix_learning_critic.pt')
parser.add_argument('--episode_size', type=int, default=60)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
local = args.local
dataset_name = args.dataset_name
debug = args.debug
region_name = args.region_name
pretrain_filename = args.pretrain_filename
pretrain_file = './save/{}/{}'.format(dataset_name, pretrain_filename)

episode_size = args.episode_size  # 一个小时一个 episode

if local:
    data_root = './data/'
else:
    data_root = '/mnt/data/jwj/'


device = args.device
torch.cuda.set_device(device)
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
# 定义 agent policy config
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

# 定义 Critic config
critic_config = {
    'agent_state_size': policy_config['SeqMovingStateNet']['hidden_size'],
    'info_size': policy_config['info_size'],
    'hidden_size': policy_config['hidden_size'],
    'conv_kernel_size': 3,
    'stride': 1,
    'conv_padding': 1,
    'attn_type': 'general_qmix'
}

# 定义 config
config = {
    'dataset_name': dataset_name,
    'max_step': 2 if debug else 100,
    'agent_lr': 0.0005,
    'critic_lr': 0.0005,
    'optim_alpha': 0.99,
    'optim_eps': 1e-8,
    'weight_decay': 0.00001,
    'batch_size': 64,
    'global_gamma': 1.0,
    'global_alpha': 0.1,
    'local_gamma': 0.99,
    'td_lambda': 0.9,
    'grad_norm_clip': 5.0,  # 这个参数可调
    'critic_pretrain_step': 2 if debug else 3000,
    'save_agent_file': args.save_agent_file,
    'save_critic_file': args.save_critic_file,
    'target_update_interval': 40,  # 因为我们每一个 episode 是 60 步，所以这里是 40 * 60
    'learner_log_interval': 20,
    'critic_config': critic_config,
    'soft_update': True
}
logger = get_logger(name='Mix_learning')
logger.info('read data')
if dataset_name == 'BJ_Taxi':
    # 目前不会实装海淀区
    if region_name == 'partA':
        train_data = pd.read_csv(
            os.path.join(data_root, dataset_name, 'beijing_partA_episode_{}_train.csv'.format(episode_size)))
    else:
        train_data = pd.read_csv(
            os.path.join(data_root, dataset_name, 'beijing_partB_episode_{}_train.csv'.format(episode_size)))
elif dataset_name == 'Xian':
    # dataset_name == 'Xian'
    if region_name == 'partA':
        train_data = pd.read_csv(
            os.path.join(data_root, dataset_name, 'xianshi_partA_episode_{}_train.csv'.format(episode_size)))
    else:
        train_data = pd.read_csv(
            os.path.join(data_root, dataset_name, 'xianshi_partB_episode_{}_train.csv'.format(episode_size)))
else:
    assert dataset_name == 'Chengdu'
    if region_name == 'partA':
        train_data = pd.read_csv(os.path.join(data_root, dataset_name, 'chengdushi_partA_episode_{}_train.csv'.format(episode_size)))
    else:
        train_data = pd.read_csv(os.path.join(data_root, dataset_name, 'chengdushi_partB_episode_{}_train.csv'.format(episode_size)))

# 加载模型
agent_policy = AgentPolicyNetV3(policy_config=policy_config).to(device)
agent_policy.load_state_dict(torch.load(pretrain_file, map_location=device))
logger.info('init agent policy net')
logger.info(agent_policy)

# 加载仿真管理器
with open(os.path.join(data_root, dataset_name, 'road_surrounding_list.json'), 'r') as f:
    road_surrounding_list = json.load(f)

with open(os.path.join(data_root, dataset_name, 'road_candidate_list.json'), 'r') as f:
    road_candidate_list = json.load(f)
# 加载区域 road list 与 road2grid
with open(os.path.join(data_root, dataset_name, 'region_road_dict.json'), 'r') as f:
    region_road_dict = json.load(f)
if region_name == 'partA':
    region_road_list = region_road_dict['region_A_road_list']
else:
    region_road_list = region_road_dict['region_B_road_list']
with open(os.path.join(data_root, dataset_name, 'road2grid.json'), 'r') as f:
    road2grid = json.load(f)
# 定义 agent_manager
agent_manager = AgentManager(road_surrounding_list=road_surrounding_list, road_candidate_list=road_candidate_list,
                             preference_size=policy_config['preference_size'], info_size=policy_config['info_size'],
                             num_layers=policy_config['SeqMovingStateNet']['n_layers'],
                             hidden_size=policy_config['hidden_size'], device=device, dataset_name=dataset_name,
                             region_road_list=region_road_list, road2grid=road2grid)

critic_config['grid_height'] = agent_manager.img_height
critic_config['grid_width'] = agent_manager.img_width
# 定义 episode pool
episode_pool = EpisodePool(episode_size=episode_size, data=train_data, logger=logger)
# 初始化 liir 学习器
mix_learner = MixLearnerV2(config, agent_manager, agent_policy, episode_pool, logger, device, debug)
# 开始学习
mix_learner.learning()

