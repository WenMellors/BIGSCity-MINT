# 建模 agent 个体的出行策略
import torch
import torch.nn as nn
import numpy as np
import math
from datetime import datetime, timedelta
from util.similarity_funcs import cosine_similarity
import torch.nn.functional as F


def time_encode(simulate_time_code, dataset_name):
    if dataset_name == 'BJ_Taxi':
        simulate_time_date = (simulate_time_code // 1440) + 1
        if simulate_time_date in [1, 7, 8, 14, 15, 21, 22, 28, 29]:
            return simulate_time_code % 1440 + 1440
        else:
            return simulate_time_code % 1440
    elif dataset_name == 'Xian' or dataset_name == 'Chengdu':
        start_date = datetime(2018, 10, 31)
        simulate_time_date = start_date + timedelta(minutes=int(simulate_time_code))
        if simulate_time_date.weekday() in [5, 6]:
            return simulate_time_code % 1440 + 1440
        else:
            return simulate_time_code % 1440
    else:
        raise NotImplementedError


class Agent(object):
    """
    代理实体
    """

    def __init__(self, start, des, preference_size, start_time, info_size, num_layers, hidden_size, device):
        self.loc = start
        self.des = des
        self.prefer = torch.randn((1, preference_size)).cpu()
        self.trace_loc = [start]
        self.trace_time = [start_time]
        # LSTM 相关缓存信息
        self.history_lstm = None
        self.current_c = np.zeros((num_layers, 1, hidden_size))
        self.current_h = np.zeros((num_layers, 1, hidden_size))
        # agent 的 info
        self.info = torch.zeros((1, info_size)).to(device)


class AgentManager(object):
    """
    代理实体管理器
    主要负责维护当前在环境中运行的代理列表（他们的偏好向量，信息，目的地，当前位置这些）
    """

    def __init__(self, road_surrounding_list, road_candidate_list, preference_size, info_size, num_layers, hidden_size,
                 device, region_road_list=None, road2grid=None, dataset_name=None):
        self.id2agent = {}
        self.simulate_visit_id = set()
        self.road_surrounding_list = road_surrounding_list  # 这个是无向图的连通性 (包括自环)
        self.road_candidate_list = road_candidate_list  # 这个是有向图
        self.preference_size = preference_size
        self.info_size = info_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.dataset_name = dataset_name
        if region_road_list is not None:
            self.region_road_idx = {}
            for index, road in enumerate(region_road_list):
                self.region_road_idx[road] = index
        if road2grid is not None:
            if dataset_name == 'BJ_Taxi':
                # 计算 grid 用于计算奖励（学习目标）
                lon_0 = 116.2501510
                lon_1 = 116.5003217
                lat_0 = 39.8087290
                lat_1 = 39.9999468
            elif dataset_name == 'Xian':
                # if region_name == 'partA':
                #     lon_0 = 108.8093988
                #     lon_1 = 109.0499449
                #     lat_0 = 34.17026046
                #     lat_1 = 34.29639324
                # else:
                #     lon_0 = 108.8093988
                #     lon_1 = 109.0499449
                #     lat_0 = 34.17026047
                #     lat_1 = 34.25241200
                # 这里应该使用 A + B 区域合并的网格经纬度
                lon_0 = 108.8093988
                lon_1 = 109.0499449
                lat_0 = 34.17026046
                lat_1 = 34.29639324
            else:
                assert dataset_name == 'Chengdu'
                # 这里应该使用 A + B 区域合并的网格经纬度
                lon_0 = 103.9024498
                lon_1 = 104.2261992
                lat_0 = 30.52970854
                lat_1 = 30.74048714
            img_unit = 0.005  # 这样画出来的大小，大概是 0.42 km * 0.55 km 的格子
            self.img_width = math.ceil((lon_1 - lon_0) / img_unit) + 1  # 图像的宽度
            self.img_height = math.ceil((lat_1 - lat_0) / img_unit) + 1  # 映射出的图像的高度
            self.road2grid = road2grid

    def imitate_learning_organize_input(self, current_time, agent_id_list, road_id_list, des_id_list):
        """
        组织模仿学习的输入，以 list 的形式返回
        Args:
            current_time:
            agent_id_list:
            road_id_list:
            des_id_list:

        Returns:

        """
        input_loc = []
        # 这里需要把仿真时间步，转化为时间编码
        input_time = [time_encode(current_time, dataset_name=self.dataset_name)] * len(agent_id_list)
        input_des = []
        input_inter_info = []
        input_prefer = []
        input_candidate_mask = []
        input_history_h = []
        input_current_h = []
        input_current_c = []
        # agent_id_list 就是此时此刻，所有在环境中的全量车辆
        road2agent = {}
        # 先遍历一遍，构建 road2agent 字典
        for index, agent_id in enumerate(agent_id_list):
            if agent_id not in self.id2agent:
                # 新的一个 agent
                agent_start = road_id_list[index]
                agent_des = des_id_list[index]
                new_agent = Agent(start=agent_start, des=agent_des, start_time=current_time,
                                  preference_size=self.preference_size, info_size=self.info_size,
                                  num_layers=self.num_layers, hidden_size=self.hidden_size, device=self.device)
                self.id2agent[agent_id] = new_agent
                # 添加入输入队列
                input_loc.append(agent_start)
                input_des.append(agent_des)
                input_prefer.append(new_agent.prefer)
                input_history_h.append(new_agent.history_lstm)
                input_current_h.append(torch.FloatTensor(new_agent.current_h))
                input_current_c.append(torch.FloatTensor(new_agent.current_c))
                if agent_start not in road2agent:
                    road2agent[agent_start] = [agent_id]
                else:
                    road2agent[agent_start].append(agent_id)
            else:
                # 不是新的 agent 需要跟新它当前所在的位置
                old_agent = self.id2agent[agent_id]
                cur_loc = road_id_list[index]
                cur_des = des_id_list[index]
                old_agent.loc = cur_loc
                old_agent.trace_loc.append(cur_loc)
                old_agent.trace_time.append(current_time)
                if cur_loc not in road2agent:
                    road2agent[cur_loc] = [agent_id]
                else:
                    road2agent[cur_loc].append(agent_id)
                # 添加入输入队列
                input_loc.append(cur_loc)
                input_des.append(cur_des)
                input_prefer.append(old_agent.prefer)
                input_history_h.append(old_agent.history_lstm)
                input_current_h.append(torch.FloatTensor(old_agent.current_h))
                input_current_c.append(torch.FloatTensor(old_agent.current_c))
        assert len(self.id2agent) == len(agent_id_list)
        for index, agent_id in enumerate(agent_id_list):
            # 构建信息输入
            cur_loc = road_id_list[index]
            cur_des = des_id_list[index]
            assert cur_loc != cur_des
            cur_agent = self.id2agent[agent_id]
            inter_info = [cur_agent.info]
            for surrounding_road in self.road_surrounding_list[str(cur_loc)]:
                if surrounding_road in road2agent:
                    for surrounding_agent in road2agent[surrounding_road]:
                        if surrounding_agent != agent_id:
                            surrounding_agent_object = self.id2agent[surrounding_agent]
                            inter_info.append(surrounding_agent_object.info)
            inter_info = torch.cat(inter_info, dim=0)  # (info_num, info_size)
            input_inter_info.append(inter_info)
            # 计算 candidate_mask
            candidate_mask = self.road_candidate_list[str(cur_loc)]
            input_candidate_mask.append(candidate_mask)
        return [input_loc, input_time, input_des, input_inter_info, input_prefer, input_candidate_mask, input_history_h,
                input_current_h, input_current_c]

    def imitate_learning_update(self, agent_id_list, out_info, next_history_h, next_h, next_c, des_id_list,
                                candidate_mask, target_id_list):
        """
        模仿学习中每模仿完一步，就 update 一下所有 agent 的信息（info, lstm 相关向量），以及删除到达目的地的 agent
        """
        for index, agent_id in enumerate(agent_id_list):
            # 判断这个 agent 是否需要删除
            if des_id_list[index] == candidate_mask[index][target_id_list[index]]:
                # 到达了目的地
                del self.id2agent[agent_id]
                continue
            # 更新 agent 的信息
            agent_object = self.id2agent[agent_id]
            agent_object.info = out_info[index].reshape(1, self.info_size)
            agent_object.history_lstm = next_history_h[index]
            agent_object.current_h = next_h[index]
            agent_object.current_c = next_c[index]

    def imitate_learning_reset(self):
        self.id2agent = {}
        self.simulate_visit_id = set()

    def simulate_organize_input(self, current_time, agent_id_list, road_id_list, des_id_list):
        """
        组织仿真生成的输入，以 list 的形式返回
        Args:
            current_time:
            agent_id_list: 需要注意，不是所有的 agent_id 都是要处理的，仿真的时候，就是新的 id 表示出行需求输入，后面的都是生成
            road_id_list:
            des_id_list:

        Returns:

        """
        input_loc = []
        # 这里需要把仿真时间步，转化为时间编码
        input_des = []
        input_inter_info = []
        input_prefer = []
        input_candidate_mask = []
        input_history_h = []
        input_current_h = []
        input_current_c = []
        # agent_id_list 只是作为出行需求的输入，并不代表当前仿真环境中所有出行的车辆
        # 先遍历一遍 agent_id_list，判断新的出行需求，加入环境之中
        for index, agent_id in enumerate(agent_id_list):
            if agent_id not in self.id2agent and agent_id not in self.simulate_visit_id:
                # 新的一个出行需求，需要构建 agent
                agent_start = road_id_list[index]
                agent_des = des_id_list[index]
                new_agent = Agent(start=agent_start, des=agent_des, start_time=current_time,
                                  preference_size=self.preference_size, info_size=self.info_size,
                                  num_layers=self.num_layers, hidden_size=self.hidden_size, device=self.device)
                self.id2agent[agent_id] = new_agent
        # 遍历 id2agent，构建 road2agent
        road2agent = {}
        # 增加 sorted 函数保证有序性
        env_agent_id_list = sorted(list(self.id2agent.keys()))  # 环境中的 agent_id 列表
        for agent_id in env_agent_id_list:
            old_agent = self.id2agent[agent_id]
            cur_des = old_agent.des
            cur_loc = old_agent.loc
            if cur_loc not in road2agent:
                road2agent[cur_loc] = [agent_id]
            else:
                road2agent[cur_loc].append(agent_id)
            # 添加入输入队列
            input_loc.append(cur_loc)
            input_des.append(cur_des)
            input_prefer.append(old_agent.prefer)
            input_history_h.append(old_agent.history_lstm)
            input_current_h.append(torch.FloatTensor(old_agent.current_h))
            input_current_c.append(torch.FloatTensor(old_agent.current_c))
        input_time = [time_encode(current_time, self.dataset_name)] * len(input_loc)
        for agent_id in env_agent_id_list:
            # 构建信息输入
            cur_agent = self.id2agent[agent_id]
            cur_des = cur_agent.des
            cur_loc = cur_agent.loc
            try:
                assert cur_loc != cur_des
            except AssertionError:
                debug = 1
            inter_info = [cur_agent.info]
            for surrounding_road in self.road_surrounding_list[str(cur_loc)]:
                if surrounding_road in road2agent:
                    for surrounding_agent in road2agent[surrounding_road]:
                        if surrounding_agent != agent_id:
                            surrounding_agent_object = self.id2agent[surrounding_agent]
                            inter_info.append(surrounding_agent_object.info)
            inter_info = torch.cat(inter_info, dim=0)  # (info_num, info_size)
            input_inter_info.append(inter_info)
            # 计算 candidate_mask
            candidate_mask = self.road_candidate_list[str(cur_loc)]
            input_candidate_mask.append(candidate_mask)
        return env_agent_id_list, [input_loc, input_time, input_des, input_inter_info, input_prefer,
                                   input_candidate_mask, input_history_h,
                                   input_current_h, input_current_c]

    def simulate_update(self, env_agent_id_list, current_time, out_info, next_history_h, next_h, next_c,
                        next_step_list, output_file=None, max_step=100, in_region=False, is_finish=False):
        """
        仿真生成中每生成完一步，就 update 一下所有 agent 的信息（info, lstm 相关向量），以及删除到达目的地的 agent
        对于到达目的地和达到仿真最大步数的 agent 的轨迹进行输出
        """
        cur_local_reward = []
        terminal_list = []
        for index, agent_id in enumerate(env_agent_id_list):
            # 判断这个 agent 是否需要删除
            agent_object = self.id2agent[agent_id]
            if agent_object.des == next_step_list[index]:
                # 到达了目的地
                if not in_region or (next_step_list[index] in self.region_road_idx):
                    agent_object.trace_loc.append(next_step_list[index])
                    agent_object.trace_time.append(current_time)
                if output_file is not None:
                    output_file.write('{},\"{}\",\"{}\"\n'.format(str(agent_id),
                                                                  ','.join([str(loc) for loc in agent_object.trace_loc]),
                                                                  ','.join([str(x) for x in agent_object.trace_time])))
                # 输出轨迹
                del self.id2agent[agent_id]
                # 加入已经访问过的 agent 列表
                self.simulate_visit_id.add(agent_id)
                # 给予较大的 local reward
                cur_local_reward.append(5.0)
                terminal_list.append(agent_id)
            elif len(agent_object.trace_loc) == max_step - 1 or \
                    (str(next_step_list[index]) not in self.road_candidate_list) or \
                    (in_region and (next_step_list[index] not in self.region_road_idx)):
                # 最大仿真步数或者走到死路了, 走出去了
                # 更新完轨迹之后就可以输出了
                # 如果是走出去的情况，那么就不要把这个区域外的路段加入生成的轨迹中了，这样会导致我们评估过程出错
                if not in_region or (next_step_list[index] in self.region_road_idx):
                    agent_object.trace_loc.append(next_step_list[index])
                    agent_object.trace_time.append(current_time)
                if output_file is not None:
                    output_file.write('{},\"{}\",\"{}\"\n'.format(str(agent_id),
                                                                  ','.join([str(loc) for loc in agent_object.trace_loc]),
                                                                  ','.join([str(x) for x in agent_object.trace_time])))
                # 输出轨迹
                del self.id2agent[agent_id]
                # 加入已经访问过的 agent 列表
                self.simulate_visit_id.add(agent_id)
                # 给予负数的 reward
                cur_local_reward.append(-5.0)
                terminal_list.append(agent_id)
            else:
                # agent 还在环境中
                # 更新 agent 的信息
                agent_object.info = out_info[index].reshape(1, self.info_size)
                agent_object.history_lstm = next_history_h[index]
                agent_object.current_h = next_h[index]
                agent_object.current_c = next_c[index]
                # 更新位置信息
                agent_object.loc = next_step_list[index]
                if in_region:
                    assert agent_object.loc in self.region_road_idx
                agent_object.trace_loc.append(next_step_list[index])
                agent_object.trace_time.append(current_time)
                if not is_finish:
                    cur_local_reward.append(0.0)
                else:
                    cur_local_reward.append(-1.0)
                    terminal_list.append(agent_id)
        return cur_local_reward, terminal_list

    def get_road_freq(self, region=None, road_id_list=None):
        """
        统计当前环境中每条道路上的车辆状态向量
        Args:
            region:
            road_id_list:
        Returns:

        """
        if region == 'chaoyang':
            road_freq = np.zeros((len(self.region_road_idx)), dtype=np.int32)
            if road_id_list is None:
                for agent_id in self.id2agent:
                    agent_object = self.id2agent[agent_id]
                    current_loc = agent_object.loc
                    if current_loc in self.region_road_idx:
                        road_freq[self.region_road_idx[current_loc]] += 1
            else:
                for road in road_id_list:
                    if road in self.region_road_idx:
                        road_freq[self.region_road_idx[road]] += 1
            return road_freq

    def get_grid_freq(self, road_id_list=None):
        """
        统计当前环境中每条道路上的车辆状态向量
        Args:
            road_id_list:  这是真实情况下才会使用的
        Returns:

        """
        grid_freq = np.zeros((self.img_width, self.img_height), dtype=np.int32)
        if road_id_list is None:
            for agent_id in self.id2agent:
                agent_object = self.id2agent[agent_id]
                current_loc = str(agent_object.loc)
                if current_loc in self.road2grid:
                    x, y = self.road2grid[current_loc][0], self.road2grid[current_loc][1]
                    grid_freq[x][y] += 1
        else:
            for road in road_id_list:
                road = str(road)
                if road in self.road2grid:
                    x, y = self.road2grid[road][0], self.road2grid[road][1]
                    grid_freq[x][y] += 1
        return grid_freq

    def simulate_end(self, output_file):
        for agent_id in self.id2agent:
            # 输出所有 agent
            agent_object = self.id2agent[agent_id]
            output_file.write('{},\"{}\",\"{}\"\n'.format(str(agent_id),
                                                          ','.join([str(loc) for loc in agent_object.trace_loc]),
                                                          ','.join([str(x) for x in agent_object.trace_time])))
            # 加入已经访问过的 agent 列表
            self.simulate_visit_id.add(agent_id)

    @staticmethod
    def calculate_external_rewards(real_grid_states, simulate_grid_states):
        """
        计算外部奖励值
        目前的设计是，在终止状态下再给一个总体的 Reward
        Args:
            real_grid_states (list of np.array): 真实的网格状态
            simulate_grid_states (list of np.array): 仿真的网格状态

        Returns:
            external_rewards: 外部奖励值
        """
        # 计算每一时刻真实网格状态与仿真网格状态的余弦相似度作为 reward
        external_rewards = []
        for real_grid_state, simulate_grid_state in zip(real_grid_states, simulate_grid_states):
            reward = cosine_similarity(real_grid_state.reshape(-1), simulate_grid_state.reshape(-1))
            external_rewards.append(reward)
        return sum(external_rewards)

    @staticmethod
    def calculate_external_reward(real_grid_state, simulate_grid_state):
        """
        计算外部奖励值
        Args:
            real_grid_state (np.array): 真实的网格状态
            simulate_grid_state (np.array): 仿真的网格状态

        Returns:
            external_reward: 外部奖励值
        """
        reward = cosine_similarity(real_grid_state.reshape(-1), simulate_grid_state.reshape(-1))
        return reward


class AgentPolicyNet(nn.Module):
    """
    代理实体的出行策略网络
    因为我们需要做参数共享，所以其实不同代理还是共用一个策略网络
    此外，由于我们的 agent 数目是变化的，因此我们只能通过一些随机采样来引入多样性参数
    如，随机采样偏好向量，作为 noise 噪音来实现多样化。
    """

    def __init__(self, policy_config):
        """
        Args:
            policy_config: 策略网络的参数 (dict)
        """
        super(AgentPolicyNet, self).__init__()
        # param
        self.road_num = policy_config['road_num']
        self.road_pad = policy_config['road_pad']
        self.road_emb_size = policy_config['road_emb_size']
        self.time_num = policy_config['time_num']
        self.time_pad = policy_config['time_pad']
        self.time_emb_size = policy_config['time_emb_size']
        self.device = policy_config['device']
        self.preference_size = policy_config['preference_size']
        self.info_size = policy_config['info_size']
        self.hidden_size = policy_config['hidden_size']
        # Embedding module
        self.road_emb = nn.Embedding(num_embeddings=self.road_num, embedding_dim=self.road_emb_size,
                                     padding_idx=self.road_pad)
        self.time_emb = nn.Embedding(num_embeddings=self.time_num, embedding_dim=self.time_emb_size,
                                     padding_idx=self.time_pad)
        self.seq_moving_state_net = SeqMovingSateNet(policy_config['SeqMovingStateNet'])
        self.info_encoder = InfoAttn(info_size=self.info_size, device=self.device)
        # 线性输出层
        self.out_fc = nn.Linear(in_features=self.hidden_size, out_features=self.road_num)
        self.info_fc = nn.Linear(in_features=self.hidden_size, out_features=self.info_size)

    def forward(self, loc, tim, des, inter_info, prefer, candidate_mask, history_h=None, current_h=None, current_c=None):
        """

        Args:
            loc: 当前每个 agent 所处的位置, (batch_size)
            tim: 当前仿真的时刻, (batch_size) 所有的值应该是一样的
            des: agent 的目的地, (batch_size)
            inter_info: 每个 agent 收集到的周围信息集合, (info_num, info_size) 我们约束第一个元素是 agent 自己的信息 调整成一个 list
            prefer: agent 的偏好向量, (batch_size, prefer_size)
            candidate_mask: 每个 agent 的 candidate mask, 只挑选 mask 中的路段作为可能的一下步 list of (candidate_size)
            history_h: lstm 历史最后一层输出的状态向量, (batch_size, seq_len, hidden_size) 调整成一个 list
            current_h: lstm 上一次输出的 h 值, (num_layers, batch_size, hidden_size)
            current_c: lstm 上一次输出的 c 值, (num_layers, batch_size, hidden_size)

        Returns:
            candidate_prob: 每个 agent 的候选道路的选取概率 list of (candidate_size)
            output_info: 每个 agent 当前输出的通信信息 (batch_size, info_size)
            next_history_h: 下一次 lstm 历史最后一层输出, (seq_len + 1, hidden_size) 调整成一个 list
            next_h: next h of the lstm, (num_layers, hidden_size)  调整成一个 list
            next_c: next c of the lstm, (num_layers, hidden_size)  调整成一个 list
            agent_state: 每个 agent 的状态向量, (batch_size, hidden_size)
        """
        # encode input_traj first
        road_emb = self.road_emb(loc).unsqueeze(1)  # (batch_size, 1, road_emb_size)
        time_emb = self.time_emb(tim).unsqueeze(1)
        # encode des
        des_emb = self.road_emb(des).unsqueeze(1)  # (batch_size, 1, road_emb_size)
        # encode inter_info
        # (batch_size, 1, info_size)
        inter_info_encode = self.info_encoder(inter_info).unsqueeze(1)
        # (batch_size, 1, lstm_input_size)
        lstm_input = torch.cat([road_emb, time_emb, prefer.unsqueeze(1), des_emb, inter_info_encode], dim=2)
        moving_state, next_history_h, next_h, next_c = self.seq_moving_state_net(lstm_input, current_h, current_c,
                                                                                 history_h)
        # 根据 moving_state 进行下一步预测
        score = self.out_fc(moving_state)  # (batch_size, road_num)
        # 根据 candidate_mask 挑选
        candidate_prob = []
        for i in range(len(candidate_mask)):
            candidate_prob.append(torch.gather(score[i], dim=0, index=torch.LongTensor(candidate_mask[i]).to(
                self.device)))
        # 传达信息
        out_info = self.info_fc(moving_state.detach())  # info_fc 的训练由下一轮的其他 agent 的反馈来迭代
        return candidate_prob, out_info, next_history_h, next_h, next_c, moving_state


class AgentPolicyNetV2(nn.Module):
    """
    在 Policy Net v1 的基础上，改进通信模块，使用 Multi-head attention 来处理 info，并且添加残差链接与 Norm
    """

    def __init__(self, policy_config):
        """
        Args:
            policy_config: 策略网络的参数 (dict)
        """
        super(AgentPolicyNetV2, self).__init__()
        # param
        self.road_num = policy_config['road_num']
        self.road_pad = policy_config['road_pad']
        self.road_emb_size = policy_config['road_emb_size']
        self.time_num = policy_config['time_num']
        self.time_pad = policy_config['time_pad']
        self.time_emb_size = policy_config['time_emb_size']
        self.device = policy_config['device']
        self.preference_size = policy_config['preference_size']
        self.info_size = policy_config['info_size']
        self.hidden_size = policy_config['hidden_size']
        self.head_num = policy_config['head_num']
        # Embedding module
        self.road_emb = nn.Embedding(num_embeddings=self.road_num, embedding_dim=self.road_emb_size,
                                     padding_idx=self.road_pad)
        self.time_emb = nn.Embedding(num_embeddings=self.time_num, embedding_dim=self.time_emb_size,
                                     padding_idx=self.time_pad)
        # LSTM 输入前的映射层
        self.input_size = self.road_emb_size * 2 + self.time_emb_size + self.info_size * 2 + self.preference_size
        self.lstm_input_fc = nn.Linear(in_features=self.input_size, out_features=self.hidden_size)
        self.seq_moving_state_net = SeqMovingSateNet(policy_config['SeqMovingStateNet'])
        self.info_encoder = InfoMultiAttn(info_size=self.info_size, head_num=self.head_num, device=self.device)
        # 线性输出层
        self.out_fc = nn.Linear(in_features=self.hidden_size, out_features=self.road_num)
        # self.info_fc1 = nn.Linear(in_features=self.hidden_size + self.info_size*2, out_features=self.info_size)
        self.info_fc1 = nn.Linear(in_features=self.hidden_size, out_features=self.info_size)
        # self.info_fc2 = nn.Linear(in_features=self.hidden_size, out_features=self.info_size)
        # self.info_mlp = nn.Sequential(info_fc1, nn.ReLU(), info_fc2)
        # norm
        self.layer_norm1 = nn.LayerNorm(self.info_size*2)
        # self.layer_norm2 = nn.LayerNorm(self.hidden_size + self.info_size*2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, loc, tim, des, inter_info, prefer, candidate_mask, history_h=None, current_h=None, current_c=None):
        """

        Args:
            loc: 当前每个 agent 所处的位置, (batch_size)
            tim: 当前仿真的时刻, (batch_size) 所有的值应该是一样的
            des: agent 的目的地, (batch_size)
            inter_info: 每个 agent 收集到的周围信息集合, (info_num, info_size) 我们约束第一个元素是 agent 自己的信息 调整成一个 list
            prefer: agent 的偏好向量, (batch_size, prefer_size)
            candidate_mask: 每个 agent 的 candidate mask, 只挑选 mask 中的路段作为可能的一下步 list of (candidate_size)
            history_h: lstm 历史最后一层输出的状态向量, (batch_size, seq_len, hidden_size) 调整成一个 list
            current_h: lstm 上一次输出的 h 值, (num_layers, batch_size, hidden_size)
            current_c: lstm 上一次输出的 c 值, (num_layers, batch_size, hidden_size)

        Returns:
            candidate_prob: 每个 agent 的候选道路的选取概率 list of (candidate_size)
            output_info: 每个 agent 当前输出的通信信息 (batch_size, info_size)
            next_history_h: 下一次 lstm 历史最后一层输出, (seq_len + 1, hidden_size) 调整成一个 list
            next_h: next h of the lstm, (num_layers, hidden_size)  调整成一个 list
            next_c: next c of the lstm, (num_layers, hidden_size)  调整成一个 list
            agent_state: 每个 agent 的状态向量, (batch_size, hidden_size)
        """
        # encode input_traj first
        road_emb = self.road_emb(loc).unsqueeze(1)  # (batch_size, 1, road_emb_size)
        time_emb = self.time_emb(tim).unsqueeze(1)
        # encode des
        des_emb = self.road_emb(des).unsqueeze(1)  # (batch_size, 1, road_emb_size)
        # encode inter_info
        # (batch_size, info_size)
        inter_info_encode, self_info = self.info_encoder(inter_info)
        # Add&Norm
        # (batch_size, 1, info_size*2)
        info_encode = self.layer_norm1(torch.cat([self_info, inter_info_encode], dim=1)).unsqueeze(1)
        # (batch_size, 1, lstm_input_size)
        lstm_input = torch.cat([road_emb, time_emb, prefer.unsqueeze(1), des_emb, info_encode], dim=2)
        lstm_input = self.relu1(self.lstm_input_fc(lstm_input))
        moving_state, next_history_h, next_h, next_c = self.seq_moving_state_net(lstm_input, current_h, current_c,
                                                                                 history_h)
        # 根据 moving_state 进行下一步预测
        score = self.out_fc(moving_state)  # (batch_size, road_num)
        # 根据 candidate_mask 挑选
        candidate_prob = []
        for i in range(len(candidate_mask)):
            candidate_prob.append(torch.gather(score[i], dim=0, index=torch.LongTensor(candidate_mask[i]).to(
                self.device)))
        # 传达信息
        # info_fc 的训练由下一轮的其他 agent 的反馈来迭代
        # out_info_input = self.layer_norm2(torch.cat([moving_state, info_encode.squeeze(1)], dim=1)).detach()
        out_info = self.info_fc1(moving_state.detach())
        # out_info = self.info_fc2(out_info_hidden)
        return candidate_prob, out_info, next_history_h, next_h, next_c, moving_state


class AgentPolicyNetV3(nn.Module):
    """
    在 Policy Net v1 的基础上，改进通信模块，使用 GAT 来处理 info，由于我们的通信图是动态的，所以可以叫做 dynamic GAT
    """

    def __init__(self, policy_config):
        """
        Args:
            policy_config: 策略网络的参数 (dict)
        """
        super(AgentPolicyNetV3, self).__init__()
        # param
        self.road_num = policy_config['road_num']
        self.road_pad = policy_config['road_pad']
        self.road_emb_size = policy_config['road_emb_size']
        self.time_num = policy_config['time_num']
        self.time_pad = policy_config['time_pad']
        self.time_emb_size = policy_config['time_emb_size']
        self.device = policy_config['device']
        self.preference_size = policy_config['preference_size']
        self.info_size = policy_config['info_size']
        self.hidden_size = policy_config['hidden_size']
        self.head_num = policy_config['head_num']
        self.dropout_input_p = policy_config['dropout_input_p']
        self.dropout_hidden_p = policy_config['dropout_hidden_p']
        # Embedding module
        self.road_emb = nn.Embedding(num_embeddings=self.road_num, embedding_dim=self.road_emb_size,
                                     padding_idx=self.road_pad)
        self.time_emb = nn.Embedding(num_embeddings=self.time_num, embedding_dim=self.time_emb_size,
                                     padding_idx=self.time_pad)
        # LSTM 输入前的映射层
        self.input_size = self.road_emb_size * 2 + self.time_emb_size + self.info_size + self.preference_size
        self.lstm_input_fc = nn.Linear(in_features=self.input_size, out_features=self.hidden_size)
        self.seq_moving_state_net = SeqMovingSateNet(policy_config['SeqMovingStateNet'])
        self.info_encoder = InfoGatAttn(info_size=self.info_size, head_num=self.head_num, device=self.device)
        # 线性输出层
        self.out_fc = nn.Linear(in_features=self.hidden_size, out_features=self.road_num)
        self.info_fc1 = nn.Linear(in_features=self.hidden_size, out_features=self.info_size)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout_input = nn.Dropout(p=self.dropout_input_p)
        self.dropout_hidden = nn.Dropout(p=self.dropout_hidden_p)

    def forward(self, loc, tim, des, inter_info, prefer, candidate_mask, history_h=None, current_h=None, current_c=None):
        """

        Args:
            loc: 当前每个 agent 所处的位置, (batch_size)
            tim: 当前仿真的时刻, (batch_size) 所有的值应该是一样的
            des: agent 的目的地, (batch_size)
            inter_info: 每个 agent 收集到的周围信息集合, (info_num, info_size) 我们约束第一个元素是 agent 自己的信息 调整成一个 list
            prefer: agent 的偏好向量, (batch_size, prefer_size)
            candidate_mask: 每个 agent 的 candidate mask, 只挑选 mask 中的路段作为可能的一下步 list of (candidate_size)
            history_h: lstm 历史最后一层输出的状态向量, (batch_size, seq_len, hidden_size) 调整成一个 list
            current_h: lstm 上一次输出的 h 值, (num_layers, batch_size, hidden_size)
            current_c: lstm 上一次输出的 c 值, (num_layers, batch_size, hidden_size)

        Returns:
            candidate_prob: 每个 agent 的候选道路的选取概率 list of (candidate_size)
            output_info: 每个 agent 当前输出的通信信息 (batch_size, info_size)
            next_history_h: 下一次 lstm 历史最后一层输出, (seq_len + 1, hidden_size) 调整成一个 list
            next_h: next h of the lstm, (num_layers, hidden_size)  调整成一个 list
            next_c: next c of the lstm, (num_layers, hidden_size)  调整成一个 list
            agent_state: 每个 agent 的状态向量, (batch_size, hidden_size)
        """
        # encode input_traj first
        road_emb = self.road_emb(loc).unsqueeze(1)  # (batch_size, 1, road_emb_size)
        time_emb = self.time_emb(tim).unsqueeze(1)
        # encode des
        des_emb = self.road_emb(des).unsqueeze(1)  # (batch_size, 1, road_emb_size)
        # encode inter_info
        # (batch_size, info_size)
        inter_info_encode, self_info = self.info_encoder(inter_info)
        # (batch_size, 1, lstm_input_size)
        lstm_input = torch.cat([road_emb, time_emb, prefer.unsqueeze(1), des_emb, inter_info_encode.unsqueeze(1)], dim=2)
        lstm_input = self.dropout_input(lstm_input)
        lstm_input = self.relu1(self.lstm_input_fc(lstm_input))
        moving_state, next_history_h, next_h, next_c = self.seq_moving_state_net(lstm_input, current_h, current_c,
                                                                                 history_h)
        moving_state = self.dropout_hidden(moving_state)
        # 根据 moving_state 进行下一步预测
        score = self.out_fc(moving_state)  # (batch_size, road_num)
        # 根据 candidate_mask 挑选
        candidate_prob = []
        for i in range(len(candidate_mask)):
            candidate_prob.append(torch.gather(score[i], dim=0, index=torch.LongTensor(candidate_mask[i]).to(
                self.device)))
        # 传达信息
        # info_fc 的训练由下一轮的其他 agent 的反馈来迭代
        out_info = self.info_fc1(moving_state.detach())
        return candidate_prob, out_info, next_history_h, next_h, next_c, moving_state


class LstmAttn(nn.Module):
    """ scaled dot-product attention
    """

    def __init__(self, hidden_size):
        super(LstmAttn, self).__init__()
        # param
        self.scale_d = np.sqrt(hidden_size)

    def forward(self, query, key):
        """前馈

        Args:
            query (tensor): 当前轨迹经过 LSTM 后的隐藏层向量序列 (hidden_size)
            key (tensor): 轨迹向量序列的最后一个状态 (seq_len, hidden_size)
        Return:
            attn_hidden (tensor): shape (hidden_size)
        """
        attn_weight = torch.mm(key, query.unsqueeze(1)).squeeze(1) / self.scale_d  # shape (seq_len)
        attn_weight = torch.softmax(attn_weight, dim=0).unsqueeze(1)  # (seq_len, 1)
        attn_hidden = torch.sum(attn_weight * key, dim=0)
        return attn_hidden


class InfoAttn(nn.Module):
    """ general-1 dot-product attention
    """

    def __init__(self, info_size, device):
        super(InfoAttn, self).__init__()
        # param
        self.scale_d = np.sqrt(info_size)
        self.info_size = info_size
        self.device = device
        # network
        self.w_q = nn.Linear(in_features=self.info_size, out_features=self.info_size, bias=False)
        self.w_v = nn.Linear(in_features=self.info_size, out_features=self.info_size, bias=False)
        self.w_k = nn.Linear(in_features=self.info_size, out_features=self.info_size, bias=False)

    def forward(self, inter_info):
        """前馈

        Args:
            inter_info: 每个 agent 收集到的周围信息集合, (info_num, info_size) 我们约束第一个元素是 agent 自己的信息 调整成一个 list
        Return:
            attn_hidden (tensor): shape (agent_num, info_size)
        """
        batch_attn_hidden = []
        for i in range(len(inter_info)):
            query = inter_info[i][0]  # (info_size)
            key = inter_info[i]  # (info_num, info_size)
            value = self.w_v(key)  # (info_num, info_size)
            attn_weight = torch.mm(self.w_k(key), self.w_q(query).unsqueeze(1)).squeeze(1)  # shape (info_num)
            attn_weight = torch.softmax(attn_weight, dim=0).unsqueeze(1)  # (info_num, 1)
            attn_hidden = torch.sum(attn_weight * value, dim=0).unsqueeze(0)  # (1, info_size)
            batch_attn_hidden.append(attn_hidden)
        batch_attn_hidden = torch.cat(batch_attn_hidden, dim=0)  # (agent_num, info_size)
        return batch_attn_hidden


class InfoMultiAttn(nn.Module):
    """
    encode Information with multi-head attention
    """

    def __init__(self, info_size, head_num, device):
        super(InfoMultiAttn, self).__init__()
        if info_size % head_num != 0:
            raise ValueError('info_size must be divisible by head_num')
        # param
        self.info_size = info_size
        self.head_num = head_num
        self.device = device
        # network
        self.multi_head_attn = nn.MultiheadAttention(embed_dim=self.info_size, num_heads=self.head_num)

    def forward(self, inter_info):
        batch_attn_hidden = []
        batch_query_info = []
        for i in range(len(inter_info)):
            info_vectors = inter_info[i].unsqueeze(1)  # 就是 Key 和 Value
            query = info_vectors[0].unsqueeze(0)  # (1, 1, info_size)
            attn_output, attn_output_weights = self.multi_head_attn(query, info_vectors, info_vectors)
            batch_attn_hidden.append(attn_output.squeeze(1))
            batch_query_info.append(query.squeeze(1))
        batch_attn_hidden = torch.cat(batch_attn_hidden, dim=0)  # (agent_num, info_size)
        batch_query_info = torch.cat(batch_query_info, dim=0)  # (agent_num, info_size)
        return batch_attn_hidden, batch_query_info


class InfoGatAttn(nn.Module):
    """
    encode Information with gat attention
    由于我们在外部就根据图邻接关系处理好了输入，所以这里不用构建每个 agent 的子图
    """

    def __init__(self, info_size, head_num, device):
        super(InfoGatAttn, self).__init__()
        if info_size % head_num != 0:
            raise ValueError('info_size must be divisible by head_num')
        # param
        self.info_size = info_size
        self.head_num = head_num
        self.head_hidden_size = self.info_size // self.head_num
        self.device = device
        # GAT 的 Attention 机制有两个参数：映射 Q,K,V 的 W，以及将 [WQ, WK] 映射为一个标量系数的 a
        self.W = nn.Linear(in_features=self.info_size, out_features=self.info_size, bias=False)
        # 本来的 a 应该是下面两个张量的合并，但是分开实现会简单一点
        self.cur_node_a = nn.Parameter(torch.Tensor(1, self.head_num, self.head_hidden_size))
        self.neighbor_a = nn.Parameter(torch.Tensor(1, self.head_num, self.head_hidden_size))
        self.leaky_relu = nn.LeakyReLU()
        self.out_fc = nn.Linear(in_features=self.info_size, out_features=self.info_size, bias=False)

    def forward(self, inter_info):
        batch_attn_hidden = []
        batch_query_info = []
        for i in range(len(inter_info)):
            info_vectors = inter_info[i]  # (neighbor_num + 1, info_size) 当前节点的邻居和他自己
            cur_node_info = info_vectors[0].unsqueeze(0)  # (1, info_size)
            neighbor_info = info_vectors  # (neighbor_num + 1, info_size)
            cur_node_info_proj = self.W(cur_node_info)  # (1, info_size)
            neighbor_info_proj = self.W(neighbor_info)  # (neighbor_num, info_size)
            # 按照 head_hidden_size 重新 reshape
            cur_node_info_proj = cur_node_info_proj.view(-1, self.head_num, self.head_hidden_size)  # (1, head_num, head_hidden_size)
            neighbor_info_proj = neighbor_info_proj.view(-1, self.head_num, self.head_hidden_size)  # (neighbor_num + 1, head_num, head_hidden_size)
            # 计算 attention 得分
            cur_node_info_score = (cur_node_info_proj * self.cur_node_a).sum(dim=-1)  # (1, head_num)
            neighbor_info_score = (neighbor_info_proj * self.neighbor_a).sum(dim=-1)  # (neighbor_num + 1, head_num)
            # 计算 attention 权重
            cur_edge_score = self.leaky_relu(cur_node_info_score + neighbor_info_score)  # (neighbor_num + 1, head_num)
            cur_edge_score = torch.softmax(cur_edge_score, dim=0).unsqueeze(2)  # (neighbor_num + 1, head_num, 1)
            # 计算 attention 后的信息
            attn_info = (neighbor_info_proj * cur_edge_score).sum(dim=0)  # (head_num, info_size)
            attn_info = attn_info.view(-1, self.info_size)  # (1, info_size)
            # attn_output, attn_output_weights = self.multi_head_attn(query, info_vectors, info_vectors)
            batch_attn_hidden.append(attn_info)
            batch_query_info.append(cur_node_info)
        batch_attn_hidden = torch.cat(batch_attn_hidden, dim=0)  # (agent_num, info_size)
        batch_attn_output = self.out_fc(batch_attn_hidden)  # (agent_num, info_size)
        batch_query_info = torch.cat(batch_query_info, dim=0)  # (agent_num, info_size)
        return batch_attn_output, batch_query_info


class SeqMovingSateNet(nn.Module):
    """
    Model sequential moving state
    """

    def __init__(self, config):
        super(SeqMovingSateNet, self).__init__()
        # parameter
        # the size of the embedded trajectory (concatenate with preference vector and destination)
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.device = config['device']
        self.n_layers = config['n_layers']
        # self.dropout_p = config['dropout_p']
        # network
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, batch_first=True)
        # self_attention
        self.self_attn = LstmAttn(hidden_size=self.hidden_size)
        # Dropout
        # self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, current_input, current_h, current_c, history_h):
        """
        Calculate current moving state.

        Args:
            current_input: lstm 当前的输入, (batch_size, 1, lstm_input_size)
            current_h: lstm 当前输入的 h 值, (num_layers, batch_size, hidden_size)
            current_c: lstm 当前输入的 c 值, (num_layers, batch_size, hidden_size)
            history_h: lstm 历史最后一层输出的状态向量, (seq_len, hidden_size)  调整为 list

        Returns:
            moving_state: current moving state (batch_size, hidden_size)
            next_history_h: the new history last layer h of the lstm, (seq_len + 1, hidden_size) 调整为 list
            next_h: next h of the lstm, (num_layers, hidden_size) 调整为 list
            next_c: next c of the lstm, (num_layers, hidden_size) 调整为 list
        """
        batch_moving_state = []
        batch_next_history_h = []
        batch_size = current_input.shape[0]
        batch_next_h = []
        batch_next_c = []
        lstm_out, (next_h, next_c) = self.lstm(current_input, (current_h, current_c))
        next_h = next_h.detach().cpu().numpy()
        next_c = next_c.detach().cpu().numpy()
        for i in range(batch_size):
            if history_h[i] is not None:
                next_history_h = torch.cat([torch.FloatTensor(history_h[i]).to(self.device), lstm_out[i]], dim=0)
                # (1, hidden_size)
                moving_state = self.self_attn(query=lstm_out[i].squeeze(0), key=next_history_h).unsqueeze(0)
                batch_moving_state.append(moving_state)
                batch_next_history_h.append(next_history_h.detach().cpu().numpy())
                batch_next_h.append(next_h[:, i, :].reshape(-1, 1, self.hidden_size))
                batch_next_c.append(next_c[:, i, :].reshape(-1, 1, self.hidden_size))
            else:
                next_history_h = lstm_out[i]
                moving_state = lstm_out[i]
                batch_moving_state.append(moving_state)
                batch_next_history_h.append(next_history_h.detach().cpu().numpy())
                batch_next_h.append(next_h[:, i, :].reshape(-1, 1, self.hidden_size))
                batch_next_c.append(next_c[:, i, :].reshape(-1, 1, self.hidden_size))
        batch_moving_state = torch.cat(batch_moving_state, dim=0)
        return batch_moving_state, batch_next_history_h, batch_next_h, batch_next_c
