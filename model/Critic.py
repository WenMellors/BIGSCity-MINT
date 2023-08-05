# Critic Network
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    """
    根据当前的城市交通状态预测状态价值（即状态价值函数网络）
    难点：如何定义当前的仿真状态，将其组织为输入输入进该网络中。
    """
    def __init__(self, critic_config):
        super(ValueNetwork, self).__init__()
        self.time_num = critic_config['time_num']
        self.time_pad = critic_config['time_pad']
        self.time_emb_size = critic_config['time_emb_size']
        self.time_emb = nn.Embedding(num_embeddings=self.time_num, embedding_dim=self.time_emb_size,
                                     padding_idx=self.time_pad)
        self.road_freq_size = critic_config['road_freq_size']
        self.road_freq_emb_size = critic_config['road_freq_emb_size']
        self.road_freq_fc = nn.Linear(in_features=self.road_freq_size, out_features=self.road_freq_emb_size)
        # 后续的 MLP 网络
        self.fc1 = nn.Linear(in_features=self.time_emb_size + self.road_freq_emb_size,
                             out_features=critic_config['fc1_size'])
        self.fc2 = nn.Linear(in_features=critic_config['fc1_size'], out_features=critic_config['fc2_size'])
        self.out_fc = nn.Linear(in_features=critic_config['fc2_size'], out_features=1)

    def forward(self, current_time, road_freq):
        """

        Args:
            current_time (tensor): 当前的时间 (1)
            road_freq (tensor): 当前每个路段上的行车数量 (一个向量) 会不会太稀疏？后面可以试试其他方案，先把 pipeline 建立起来 (road_freq_size)

        Returns:
            value: 预测的状态价值
        """
        current_time_emb = self.time_emb(current_time)  # (1, time_emb_size)
        road_freq_emb = self.road_freq_fc(road_freq).unsqueeze(0)  # (1, road_freq_emb_size)
        hidden_state = torch.relu(self.fc1(torch.cat([current_time_emb, road_freq_emb], dim=1)))
        hidden_state = torch.relu(self.fc2(hidden_state))
        value = torch.relu(self.out_fc(hidden_state)).squeeze(1)
        return value


class Attention(nn.Module):
    """
    实现一个 scaled dot-product Attention 网络
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # param
        self.scale_d = np.sqrt(hidden_size)

    def forward(self, query, key):
        """前馈

        Args:
            query (tensor): 全局城市交通状态 state 向量 (hidden_size)
            key (tensor): 全体 Agent 的 state 向量集合 (agent_nums, hidden_size)
        Return:
            attn_hidden (tensor): shape (hidden_size)
        """
        attn_weight = torch.mm(key, query.unsqueeze(1)).squeeze(1) / self.scale_d  # shape (seq_len)
        attn_weight = torch.softmax(attn_weight, dim=0).unsqueeze(1)  # (seq_len, 1)
        attn_hidden = torch.sum(attn_weight * key, dim=0)
        return attn_hidden


class LIIRCritic(nn.Module):
    """
    基于 LIIR 学习框架的 Critic 评论家网络
    reference: LIIR: Learning Individual Intrinsic Reward in Multi-Agent Reinforcement Learning
    网络的输入为：全体 agent 的 state 和全局宏观的城市交通 state (以网格热力图的形式表征)
    """

    def __init__(self, critic_config):
        """

        :param critic_config: 网络的参数配置
        """
        super(LIIRCritic, self).__init__()
        # 定义 Agent state 的处理网络参数
        self.agent_state_size = critic_config['agent_state_size']  # agent 的 state 隐层状态
        self.hidden_size = critic_config['hidden_size']  # agent 的 state 的第一个全连接层的隐层状态
        # 定义处理输入的网格热力图序列的 CNN + GRU 的网络参数
        self.grid_height = critic_config['grid_height']  # 网格热力图的高度
        self.grid_width = critic_config['grid_width']  # 网格热力图的宽度
        self.conv_kernel_size = critic_config['conv_kernel_size']  # 卷积核的大小
        self.stride = critic_config['stride']  # 卷积核的步长
        self.conv_padding = critic_config['conv_padding']  # 卷积核的 padding
        # 这里需要计算一下 grid_conv 的输出维度，与输入的网格热力图的 height 和 width 有关
        self.conv_output_height = int((self.grid_height - self.conv_kernel_size + 2 * self.conv_padding) / self.stride + 1)
        self.conv_output_width = int((self.grid_width - self.conv_kernel_size + 2 * self.conv_padding) / self.stride + 1)
        # 定义后续的 MLP 网络参数
        self.r_in_fc1_size = critic_config['r_in_fc1_size']  # 预测 r_in 的 MLP 的第一个线性层的隐层状态
        # 下面需要定义一个二维卷积层和一个GRU层来处理输入的网格热力图序列
        self.grid_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=self.conv_kernel_size,
                                   stride=self.stride, padding=self.conv_padding)
        self.grid_gru = nn.GRU(input_size=self.conv_output_height * self.conv_output_width,
                               hidden_size=self.hidden_size, batch_first=True)
        # 接下来需要再定义一个 Linear 网络来处理输入的 agent 的 state 序列
        self.agent_fc1 = nn.Linear(in_features=self.agent_state_size, out_features=self.hidden_size)
        # 先定义一个二层 MLP 网络来处理输入的 agent 的 state 和网格热力图的 state, 并预测 r_in
        self.r_in_fc1 = nn.Linear(in_features=self.hidden_size * 2,
                                  out_features=self.r_in_fc1_size)
        self.r_in_out = nn.Linear(in_features=self.r_in_fc1_size, out_features=1)
        self.v_mix_output = nn.Linear(in_features=self.r_in_fc1_size, out_features=1)
        # 再定义一个 Attention 网络和一个线性层来预测 v_ex
        self.state_attn = Attention(hidden_size=self.hidden_size)
        self.v_ex_output = nn.Linear(in_features=self.hidden_size * 2, out_features=1)

    def forward(self, agent_state, grid_state):
        """

        :param agent_state: agent 状态隐层向量集合, (agent_size, hidden_size)
        :param grid_state: 网格热力图状态隐层向量序列, (episode_len, grid_height, grid_width) 从刚开始仿真到现在时间步的一个热力图
        :return: v_ex (1),v_mix (agent_size), r_in (agent_size)
        """
        # 先处理 agent 的 state 序列
        agent_state = torch.relu(self.agent_fc1(agent_state))  # (agent_size, hidden_size)
        # 再处理网格热力图的 state 序列
        grid_state = grid_state.unsqueeze(1)  # (episode_len, 1, grid_height, grid_width)
        grid_state = torch.relu(self.grid_conv(grid_state))  # (episode_len, 1, conv_output_height, conv_output_width)
        # (1, episode_len, conv_output_height * conv_output_width)
        grid_state = grid_state.reshape(1, grid_state.size(0), -1)
        grid_state, _ = self.grid_gru(grid_state)  # (1, episode_len, hidden_size)
        last_grid_state = grid_state[:, -1, :]  # (1, hidden_size)
        # 将 agent_state 和 last_grid_state 拼接然后预测 r_in
        # 需要先将 last_grid_state broadcast 到 agent_state 的维度
        # (agent_size, hidden_size)
        agent_grid_state = last_grid_state.repeat(agent_state.shape[0], 1)
        # (agent_size, hidden_size * 2)
        agent_concat_state = torch.cat([agent_state, agent_grid_state], dim=1)
        r_in_fc1_hidden = torch.relu(self.r_in_fc1(agent_concat_state))  # (agent_size, r_in_fc1_size)
        v_mix = self.v_mix_output(r_in_fc1_hidden)  # (agent_size, 1) mix_v 不一定是一个非负数，因为我的 r_in 可以是负数
        r_in = self.r_in_out(r_in_fc1_hidden)  # (agent_size, 1)  做一个梯度打断，因为这部分不是随着前面层来 BP 的
        # 对 r_in 做一些修改，参考 LIIR 中的操作
        r_in = torch.tanh(r_in / 10)  # r_in 可以是负数
        r_in = r_in * 10  # (agent_size, 1)
        # 接下来需要计算 Attention 后的网格热力图的 state，然后预测 v_ex 与 v_mix
        # 这里也要做一个计算图的打断
        # (1, hidden_size)
        attn_grid_state = self.state_attn(last_grid_state.squeeze(0), agent_state).unsqueeze(0)
        grid_concat_state = torch.cat([last_grid_state, attn_grid_state], dim=1)  # (1, hidden_size * 2)
        v_ex = self.v_ex_output(grid_concat_state).squeeze(1)  # (1)
        return v_ex, v_mix.squeeze(1), r_in.squeeze(1)


class LocalCritic(nn.Module):
    """
    参考 QMIX 的设计思路，我们实现一个仅考虑 Agent 局部状态的 Critic 网络，以预测每个 agent 的价值函数
    """

    def __init__(self, critic_config):
        """
        现在是一个三层 MLP 结构，因为输入的 agent_state 已经包含 lstm 的输出信息了，所以感觉不需要再加 LSTM
        Args:
            critic_config:
        """
        super(LocalCritic, self).__init__()
        self.agent_state_size = critic_config['agent_state_size']
        self.info_size = critic_config['info_size']
        self.hidden_size = critic_config['hidden_size']
        # 先把输入的 agent_state 和 agent_info 进行一次映射
        self.agent_fc = nn.Linear(in_features=self.agent_state_size, out_features=self.hidden_size)
        self.info_fc = nn.Linear(in_features=self.info_size, out_features=self.hidden_size)
        # 再把 agent_state 和 agent_info 拼接起来，然后预测 v
        self.v_fc = nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size)
        self.v_output = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, agent_state, agent_info):
        """

        Args:
            agent_state: 每个 agent 的隐层状态向量
            agent_info: 每个 agent 的通信向量

        Returns:
            v: 每个 agent 的局部价值函数
            agent_concat_state: 每个 agent 的局部状态向量，用于支持 global critic 的计算
        """
        agent_state = torch.relu(self.agent_fc(agent_state))
        agent_info = torch.relu(self.info_fc(agent_info))
        agent_concat_state = torch.cat([agent_state, agent_info], dim=1)
        v = torch.relu(self.v_fc(agent_concat_state))
        v = self.v_output(v)
        return v, agent_concat_state


class GlobalAttention(nn.Module):
    """
    基于 attention 机制计算全局价值
    """

    def __init__(self, hidden_size, attn_type='scaled dot'):
        super(GlobalAttention, self).__init__()
        # param
        self.attn_type = attn_type
        self.hidden_size = hidden_size
        if self.attn_type == 'scaled dot':
            self.scale_d = np.sqrt(hidden_size)
        elif self.attn_type == 'additive':
            self.w_k = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
            self.w_q = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
            self.weight = nn.Linear(in_features=hidden_size, out_features=1, bias=False)
        elif self.attn_type == 'general_qmix':
            self.w_k = nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size)
            self.w_v = nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size)
            self.w = nn.Parameter(torch.randn(hidden_size, hidden_size))
            self.w_global = nn.Linear(in_features=hidden_size, out_features=hidden_size)
            self.output = nn.Linear(in_features=hidden_size * 2, out_features=1)
            # self.output_bias = nn.Sequential(
            #     nn.Linear(in_features=hidden_size, out_features=hidden_size),
            #     nn.ReLU(),
            #     nn.Linear(in_features=hidden_size, out_features=1)
            # )
        else:
            raise ValueError('attn_type must be scaled dot or additive')

    def forward(self, query, key, value):
        """前馈

        Args:
            query (tensor): 全局城市交通状态 state 向量 (hidden_size)
            key (tensor): 全体 Agent 的 state 向量集合 (agent_nums, hidden_size * 2)
            value (tensor): 全体 Agent 的局部价值 (agent_nums, hidden_size * 2)
        Return:
            global_v (tensor): 全局价值 (1)
        """
        if self.attn_type == 'scaled dot':
            attn_weight = torch.mm(key, query.unsqueeze(1)).squeeze(1) / self.scale_d  # shape (agent_nums)
            attn_weight = torch.softmax(attn_weight, dim=0)  # (agent_nums)
            global_v = torch.sum(attn_weight * value)  # (1)
        elif self.attn_type == 'general_qmix':
            # 参考 QMIX 的预估方式
            attn_weight = torch.mm(self.w_k(key), torch.mm(self.w, query.unsqueeze(1))).squeeze(1)  # (agent_nums)
            # 这里没有想好是 softmax 还是 abs
            attn_weight = torch.softmax(attn_weight, dim=0)  # (agent_nums)
            # attn_weight = torch.abs(attn_weight)
            # 融入 global 全局状态
            global_state = torch.relu(self.w_global(query)).unsqueeze(0)  # (1, hidden_size)
            hidden_state = torch.cat([torch.sum(attn_weight.unsqueeze(1) * self.w_v(value), dim=0, keepdim=True), global_state], dim=1)  # (1, hidden_size * 2)
            global_v = self.output(hidden_state).squeeze(1)  # (1)
        else:
            key = torch.relu(self.w_k(key))  # (agent_nums, hidden_size)
            query = torch.relu(self.w_q(query))  # (hidden_size)
            attn_energy = torch.tanh(key + query.unsqueeze(0))  # (agent_nums, hidden_size)
            attn_weight = torch.abs(self.weight(attn_energy).squeeze(1))  # (agent_nums)
            global_v = torch.sum(attn_weight * value)  # (1)
        return global_v, attn_weight


class GlobalCritic(nn.Module):
    """
    参考 QMIX 的设计思路，我们实现一个仅考虑全局的 Critic 网络，以预测环境整体的价值函数
    """

    def __init__(self, critic_config):
        super(GlobalCritic, self).__init__()
        self.hidden_size = critic_config['hidden_size']  # agent 的 state 的第一个全连接层的隐层状态
        # 定义处理输入的网格热力图序列的 CNN + GRU 的网络参数
        self.grid_height = critic_config['grid_height']  # 网格热力图的高度
        self.grid_width = critic_config['grid_width']  # 网格热力图的宽度
        self.conv_kernel_size = critic_config['conv_kernel_size']  # 卷积核的大小
        self.stride = critic_config['stride']  # 卷积核的步长
        self.conv_padding = critic_config['conv_padding']  # 卷积核的 padding
        self.attn_type = critic_config['attn_type']  # 最后计算全局价值的注意力机制类型
        # 这里需要计算一下 grid_conv 的输出维度，与输入的网格热力图的 height 和 width 有关
        self.conv_output_height = int(
            (self.grid_height - self.conv_kernel_size + 2 * self.conv_padding) / self.stride + 1)
        self.conv_output_width = int(
            (self.grid_width - self.conv_kernel_size + 2 * self.conv_padding) / self.stride + 1)
        # 下面需要定义一个二维卷积层和一个GRU层来处理输入的网格热力图序列
        self.grid_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=self.conv_kernel_size,
                                   stride=self.stride, padding=self.conv_padding)
        self.grid_gru = nn.GRU(input_size=self.conv_output_height * self.conv_output_width,
                               hidden_size=self.hidden_size, batch_first=True)
        # 由于我们场景中 agent 数目并不固定，所以我们使用 attention 机制来完成对 agent 局部价值的加权求和
        self.attn = GlobalAttention(attn_type=self.attn_type, hidden_size=self.hidden_size)

    def forward(self, agent_state, grid_state):
        """
        Args:
            agent_v: 每个 agent 的局部价值函数
            agent_state: 每个 agent 的局部价值隐层向量
            grid_state: 网格热力图的 state 序列 (全局状态信息)

        Returns:
            v: 环境的全局价值函数
        """
        # 处理网格热力图的 state 序列
        grid_state = grid_state.unsqueeze(1)  # (episode_len, 1, grid_height, grid_width)
        grid_state = torch.relu(self.grid_conv(grid_state))  # (episode_len, 1, conv_output_height, conv_output_width)
        # (1, episode_len, conv_output_height * conv_output_width)
        grid_state = grid_state.reshape(1, grid_state.size(0), -1)
        grid_state, _ = self.grid_gru(grid_state)  # (1, episode_len, hidden_size)
        last_grid_state = grid_state[:, -1, :]  # (1, hidden_size)
        global_v, attn_weight = self.attn(query=last_grid_state.squeeze(0), key=agent_state, value=agent_state)  # (1)
        return global_v, attn_weight


class MixCritic(nn.Module):
    """
    调用 global 和 critic，简单包装一层的网络
    """
    def __init__(self, critic_config):
        super(MixCritic, self).__init__()
        self.global_critic = GlobalCritic(critic_config)
        self.local_critic = LocalCritic(critic_config)
        # 构建一个嵌入层来编码 agent 的到达目的地后对应的 agent_state 与 agent_info
        # self.agent_terminal_state = nn.Embedding(1, critic_config['agent_state_size'])
        # self.info_terminal_state = nn.Embedding(1, critic_config['info_size'])

    def forward(self, agent_state, agent_info, grid_state):
        """

        Args:
            agent_state:
            agent_info:
            grid_state:

        Returns:
            agent_v: 每个 agent 的价值
            global_v: 全局的价值
        """
        # 先计算每个 agent 局部的价值
        agent_v, agent_hidden_state = self.local_critic(agent_state, agent_info)
        # 再计算全局价值
        global_v, agent_weight = self.global_critic(agent_hidden_state.detach(), grid_state)
        return agent_v, global_v, agent_weight.detach()
