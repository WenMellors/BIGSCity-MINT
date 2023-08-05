import numpy as np
import pandas as pd


class EpisodePool(object):
    """
    Episode 输入池
    接受训练数据, 将其处理为多个 episode, 并将其存储在 episode 池中，待 LIIRLearner 调用其采样函数后，输出一个 episode 作为 agent 的输入
    该类参考 DQN 中的经验回放机制，为 AC 算法提供输入
    """

    def __init__(self, episode_size, data, logger):
        """
        初始化 Episode 输入池

        Args:
            episode_size (int): 每个 episode 的时间跨度
            data (pandas.DataFrame): 训练数据
            logger (logging.Logger): 日志记录器

        """
        self.episode_size = episode_size  # 每个 episode 的大小
        self.episode_list = []  # episode 列表，其实只需要记录此 episode 的数据中开始时间步与结束时间步即可
        self.data = data
        self.logger = logger
        self.episode_list = []
        prev_episode_no = None
        episode_start_index = None
        for index, row in self.data.iterrows():
            if index == 0:
                prev_episode_no = row['episode_no']
                episode_start_index = index
            else:
                cur_episode_no = row['episode_no']
                if cur_episode_no != prev_episode_no:
                    # 新的 episode
                    end_episode_index = index - 1
                    if end_episode_index - episode_start_index + 1 >= 10:
                        self.episode_list.append((episode_start_index, end_episode_index))
                    prev_episode_no = cur_episode_no
                    episode_start_index = index
        self.logger.info('EpisodePool: episode_list size: {}'.format(len(self.episode_list)))
        self.episode_index_list = list(range(len(self.episode_list)))
        # 给前 24 个 epsiode 赋予高权重，剩下的低权重。因为前 24 个 episode 是我们已知的训练数据，后面是只知道出行需求的
        self.episode_weight_list = [10.0] * 24 + [0.1] * (len(self.episode_list) - 24)
        self.episode_weight_list = np.array(self.episode_weight_list)
        self.episode_weight_list = self.episode_weight_list / np.sum(self.episode_weight_list)
        # 检查 data 的合法性
        # for episode_index in range(len(self.episode_list)):
        #     start_index, end_index = self.episode_list[episode_index]
        #     agent_dict = {}
        #     for index, row in self.data.iloc[start_index:end_index + 1].iterrows():
        #         agent_id_list = [int(x) for x in row['traj_id_list'].split(',')]
        #         road_id_list = [int(x) for x in row['road_id_list'].split(',')]
        #         des_id_list = [int(x) for x in row['des_id_list'].split(',')]
        #         for idx, agent_id in enumerate(agent_id_list):
        #             if agent_id not in agent_dict:
        #                 agent_dict[agent_id] = {'trace_loc': [road_id_list[idx]], 'des': des_id_list[idx]}
        #             else:
        #                 assert agent_dict[agent_id]['des'] == des_id_list[idx]
        #                 agent_dict[agent_id]['trace_loc'].append(road_id_list[idx])
        #     # 检查所有 agent 的轨迹最后一个点就是目的地
        #     for agent_id in agent_dict:
        #         assert agent_dict[agent_id]['trace_loc'][-1] == agent_dict[agent_id]['des']

    def sample(self):
        """
        从 episode_list 中随机选择一个 episode 出来，然后根据开始时间步与结束时间步，从 data 中获取对应的数据
        Returns:
            pandas.DataFrame: 一个 episode 的数据
        """
        # episode_index = np.random.choice(self.episode_index_list, p=self.episode_weight_list)
        episode_index = np.random.randint(0, len(self.episode_list))
        start_index, end_index = self.episode_list[episode_index]
        sample_data = self.data.iloc[start_index:end_index + 1]
        assert sample_data.shape[0] >= 10
        return sample_data, episode_index
