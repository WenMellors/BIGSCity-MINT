from model.Critic import MixCritic
import copy
import torch
import numpy as np
from util.rl_utils import td_lambda_return_forward_view, build_local_lambda_return
from datetime import datetime
import os
import json
from tqdm import tqdm
import pickle


class MixLearnerV2(object):
    """
    Leaner算法流程为：
        1. 从 episode 池中抽取一个 episode 作为当前 agent 的输入（出行需求）
        2. 根据当前 agent 的输入，进行仿真，得到当前 episode 中 agents 的生成的轨迹集合
        3. 根据当前 episode 中 agents 的生成的轨迹集合，计算每个 agent 的局部 reward (到达目的地 reward 为 1，其他为 0)
        4. 将全局的 team reward 作为监督信号，利用 Actor-Critic 框架对 Agent 进行训练，并更新 Critic 的参数
        5. 重复 1-4 步，直到满足终止条件 (设定一个固定步数)
    MixLeaner 基于 QMIX 的思想，分别使用 local critic 和 global critic 来实现 actor-critic 框架
    """

    def __init__(self, config, agent_manager, agent_policy, episode_pool, logger, device, debug, pretrain_critic_file=None):
        """
        初始化 Mix Learner

        Args:
            config (dict): 参数
            agent_manager (AgentManager): agent 管理器, 负责 agent 的每一步仿真
            agent_policy (AgentPolicyNet): agent 策略网络
            episode_pool (EpisodePool): episode 输入池
            logger (Logger): 日志记录器
            device (string): 使用的设备，cpu or cuda:0, etc.
            debug (bool): 是否为 debug 模式
        """
        self.episode_pool = episode_pool
        self.agent_manager = agent_manager
        self.agent_policy = agent_policy
        self.logger = logger
        self.device = device
        self.dataset_name = config['dataset_name']
        self.debug = debug
        # 定义训练相关超参数
        self.max_step = config['max_step']
        self.agent_lr = config['agent_lr']
        self.critic_lr = config['critic_lr']
        self.optim_alpha = config['optim_alpha']
        self.optim_eps = config['optim_eps']
        self.weight_decay = config['weight_decay']
        self.batch_size = config['batch_size']
        self.global_gamma = config['global_gamma']
        self.local_gamma = config['local_gamma']
        self.global_alpha = config['global_alpha']
        self.td_lambda = config['td_lambda']
        self.grad_norm_clip = config['grad_norm_clip']
        self.critic_pretrain_step = config['critic_pretrain_step']
        self.episode_size = self.episode_pool.episode_size
        self.tau = 0.005
        self.soft_update = config['soft_update']
        self.temp_folder = './temp/marl/{}/'.format(int(datetime.now().timestamp()))
        self.save_folder = './save/{}'.format(self.dataset_name)
        self.save_agent_file = config['save_agent_file']
        self.save_critic_file = config['save_critic_file']
        self.save_log_file = './log/local_learner_metrics_log_{}.pkl'.format(datetime.now().strftime('%b-%d-%Y_%H-%M-%S'))
        # target_critic 更新的步长
        self.target_update_interval = config['target_update_interval']
        self.learner_log_interval = config['learner_log_interval']
        # 初始化 Critic 网络
        self.critic = MixCritic(config['critic_config'])
        if pretrain_critic_file is not None:
            self.critic.load_state_dict(torch.load(pretrain_critic_file, map_location=device))
        self.target_critic = copy.deepcopy(self.critic)  # 目标 Critic 网络
        # 上 device
        self.agent_policy = self.agent_policy.to(self.device)
        self.critic = self.critic.to(self.device)
        self.target_critic = self.target_critic.to(self.device)
        # 定义 optimizer
        self.agent_optimizer = torch.optim.RMSprop(params=self.agent_policy.parameters(), lr=self.agent_lr,
                                                   alpha=self.optim_alpha, eps=self.optim_eps,
                                                   weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.RMSprop(params=self.critic.parameters(), lr=self.critic_lr,
                                                    alpha=self.optim_alpha, eps=self.optim_eps,
                                                    weight_decay=self.weight_decay)
        # 控制目标网络更新的
        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_last_epoch = 0
        self.early_stop = 5  # 如果连续 5 轮 learner_log_interval 的 reward 都没有提升，那么就提前终止训练
        self.tolerance = self.early_stop

    def new_region_learning(self):
        """
        在新区域的训练中，我们是无法从历史轨迹中获取到 global reward 的。
        即我们不能使用 global reward 来训练 agent 才能符合我们的实验假设
        Returns:

        """
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)
        # 这里就直接加载的之前在已知区域上训练的 critic
        prev_best_metric = None
        if self.debug:
            torch.autograd.set_detect_anomaly(True)
        log_metrics = {
            'agent loss': [],
            'critic loss': [],
            'critic retrain loss': [],
            'mean local reward': [],
            'arrive rate': [],
            'mean local return': [],  # 统计平均的 agent 开始的 return 值
            'single global return': []
        }
        for epoch in range(self.max_step):
            # self.logger.info('Start LIIR training epoch {}'.format(epoch))
            # 从 episode 池中抽取一个 episode 作为当前 agent 的输入（出行需求）
            episode, episode_index = self.episode_pool.sample()
            new_region = True
            # 采样轨迹
            global_rewards, local_rewards, env_agent_list, agent_states, agent_info, simulate_grid_states, log_pi_taken = self.simulate_episode(
                episode)
            # 首先根据仿真返回的局部奖励来依据 TD loss 来训练 self.critic_params
            local_v, local_return, start_local_return, global_v, \
            target_global_return, agent_global_weight, critic_loss = self.train_critic(global_rewards, local_rewards,
                                                                                       env_agent_list,
                                                                                       agent_states, agent_info,
                                                                                       simulate_grid_states,
                                                                                       new_region=new_region)
            # 然后训练 agent 的 policy 网络
            agent_loss = self.train_agent_policy(local_v, local_return, global_v, target_global_return,
                                                 agent_global_weight, log_pi_taken, new_region=new_region)
            # 重新训练 critic 以适应 agent 策略的变化
            retrain_critic_loss = self.retrain_critic(10, new_region=True)
            # 输出一些训练记录
            log_metrics['agent loss'].append(agent_loss.cpu().item())
            log_metrics['critic loss'].append(critic_loss)
            log_metrics['critic retrain loss'].append(retrain_critic_loss)
            # 统计一些个体指标
            local_rewards_np = np.concatenate(local_rewards)
            no_zero_local_rewards = local_rewards_np[local_rewards_np != 0]
            log_metrics['mean local reward'].append(np.mean(no_zero_local_rewards))
            postive_local_rewards = no_zero_local_rewards[no_zero_local_rewards > 0]
            log_metrics['arrive rate'].append(len(postive_local_rewards) / len(no_zero_local_rewards))
            log_metrics['mean local return'].append(np.mean(start_local_return))
            # 统计全局指标
            log_metrics['single global return'].append(np.sum(global_rewards))
            if epoch % self.learner_log_interval == 0 or self.debug:
                # 这里修正一下评估方式：是拿当前 learning_log_interval 里面的结果指标去计算平均值
                # 这样尽可能拿到当前最新 agent policy 的一个表现
                # 缓存当前步骤训练得到的 agent_policy
                cur_metric = np.mean(log_metrics['arrive rate'][-self.learner_log_interval:])
                if prev_best_metric is None or cur_metric > prev_best_metric:
                    # 只保存 best 的结果
                    torch.save(self.agent_policy.state_dict(),
                               os.path.join(self.temp_folder, 'best_temp_agent.pt'))
                    torch.save(self.critic.state_dict(), os.path.join(self.temp_folder, 'best_temp_critic.pt'))
                    prev_best_metric = cur_metric
                # mean_ex_return = ex_return.mean().cpu().item()
                # mean_ex_v = v_ex.mean().cpu().item()
                # mean_r_in = torch.cat(r_in).mean().cpu().item()
                # mean_mix_return = torch.cat(mix_return).mean().cpu().item()
                output_str = ''
                for metrics in log_metrics:
                    output_str += '{}: {:.4f}, '.format(metrics, np.mean(log_metrics[metrics][-self.learner_log_interval:]))
                self.logger.info('LIIR Epoch {}, the cumulative metrics are: {}'.format(epoch, output_str))
                # 判断是否早停
                if epoch > self.learner_log_interval:
                    if np.mean(log_metrics['arrive rate'][-self.learner_log_interval:]) < np.mean(
                            log_metrics['arrive rate'][-self.learner_log_interval * 2:-self.learner_log_interval]):
                        self.tolerance -= 1
                    else:
                        self.tolerance = self.early_stop
                if self.tolerance == 0:
                    self.logger.info('Early stop at epoch {}'.format(epoch))
                    break
        # 保存模型
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        load_agent_temp_file = 'best_temp_agent.pt'
        load_critic_temp_file = 'best_temp_critic.pt'
        self.logger.info('load best and correspond arrive rate is {:.4f}'.format(prev_best_metric))
        self.agent_policy.load_state_dict(torch.load(os.path.join(self.temp_folder, load_agent_temp_file)))
        self.critic.load_state_dict(torch.load(os.path.join(self.temp_folder, load_critic_temp_file)))
        torch.save(self.agent_policy.state_dict(), os.path.join(self.save_folder, self.save_agent_file))
        torch.save(self.critic.state_dict(), os.path.join(self.save_folder, self.save_critic_file))
        # 保存 log metrics
        with open(self.save_log_file, 'w') as f:
            pickle.dump(log_metrics, f)
        # 删除 temp 文件
        for rt, dirs, files in os.walk(self.temp_folder):
            for name in files:
                remove_path = os.path.join(rt, name)
                os.remove(remove_path)

    def learning(self):
        """
        LIIR 算法流程
        """
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)
        # 先开始预训练 critic 一段时间
        self.pretrain_critic()
        # 开始正式的训练
        self.logger.info('Finish pretraining critic, and start LIIR training')
        prev_best_metric = None
        if self.debug:
            torch.autograd.set_detect_anomaly(True)
        log_metrics = {
            'agent loss': [],
            'critic loss': [],
            'critic retrain loss': [],
            'mean local reward': [],
            'arrive rate': [],
            'mean local return': [],  # 统计平均的 agent 开始的 return 值
            'single global return': []
        }
        for epoch in range(self.max_step):
            # self.logger.info('Start LIIR training epoch {}'.format(epoch))
            # 从 episode 池中抽取一个 episode 作为当前 agent 的输入（出行需求）
            episode, episode_index = self.episode_pool.sample()
            # 采样轨迹
            global_rewards, local_rewards, env_agent_list, agent_states, agent_info, simulate_grid_states, log_pi_taken = self.simulate_episode(episode)
            # 首先根据仿真返回的局部奖励来依据 TD loss 来训练 self.critic_params
            local_v, local_return, start_local_return, global_v, \
            target_global_return, agent_global_weight, critic_loss = self.train_critic(global_rewards, local_rewards, env_agent_list,
                                                                                       agent_states, agent_info,
                                                                                       simulate_grid_states)
            # 然后训练 agent 的 policy 网络
            agent_loss = self.train_agent_policy(local_v, local_return, global_v, target_global_return, agent_global_weight, log_pi_taken)
            # 重新训练 critic 以适应 agent 策略的变化
            retrain_critic_loss = self.retrain_critic(5)
            # 输出一些训练记录
            log_metrics['agent loss'].append(agent_loss.cpu().item())
            log_metrics['critic loss'].append(critic_loss)
            log_metrics['critic retrain loss'].append(retrain_critic_loss)
            # 统计一些个体指标
            local_rewards_np = np.concatenate(local_rewards)
            no_zero_local_rewards = local_rewards_np[local_rewards_np != 0]
            log_metrics['mean local reward'].append(np.mean(no_zero_local_rewards))
            postive_local_rewards = no_zero_local_rewards[no_zero_local_rewards > 0]
            log_metrics['arrive rate'].append(len(postive_local_rewards) / len(no_zero_local_rewards))
            log_metrics['mean local return'].append(np.mean(start_local_return))
            # 统计全局指标
            log_metrics['single global return'].append(np.sum(global_rewards))
            if epoch % self.learner_log_interval == 0 or self.debug:
                # 这里修正一下评估方式：是拿当前 learning_log_interval 里面的结果指标去计算平均值
                # 这样尽可能拿到当前最新 agent policy 的一个表现
                # 缓存当前步骤训练得到的 agent_policy
                cur_metric = np.mean(log_metrics['arrive rate'][-self.learner_log_interval:])
                if prev_best_metric is None or cur_metric > prev_best_metric:
                    # 只保存 best 的结果
                    torch.save(self.agent_policy.state_dict(),
                               os.path.join(self.temp_folder, 'best_temp_agent.pt'))
                    torch.save(self.critic.state_dict(), os.path.join(self.temp_folder, 'best_temp_critic.pt'))
                    prev_best_metric = cur_metric
                # mean_ex_return = ex_return.mean().cpu().item()
                # mean_ex_v = v_ex.mean().cpu().item()
                # mean_r_in = torch.cat(r_in).mean().cpu().item()
                # mean_mix_return = torch.cat(mix_return).mean().cpu().item()
                output_str = ''
                for metrics in log_metrics:
                    output_str += '{}: {:.4f}, '.format(metrics,
                                                        np.mean(log_metrics[metrics][-self.learner_log_interval:]))
                self.logger.info('LIIR Epoch {}, the cumulative metrics are: {}'.format(epoch, output_str))
                # 判断是否早停
                if epoch > self.learner_log_interval:
                    if np.mean(log_metrics['arrive rate'][-self.learner_log_interval:]) < np.mean(
                            log_metrics['arrive rate'][-self.learner_log_interval * 2:-self.learner_log_interval]):
                        self.tolerance -= 1
                    else:
                        self.tolerance = self.early_stop
                if self.tolerance == 0:
                    self.logger.info('Early stop at epoch {}'.format(epoch))
                    break
        # 保存模型
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        load_agent_temp_file = 'best_temp_agent.pt'
        load_critic_temp_file = 'best_temp_critic.pt'
        self.logger.info('load best and correspond arrive rate is {:.4f}'.format(prev_best_metric))
        self.agent_policy.load_state_dict(torch.load(os.path.join(self.temp_folder, load_agent_temp_file)))
        self.critic.load_state_dict(torch.load(os.path.join(self.temp_folder, load_critic_temp_file)))
        torch.save(self.agent_policy.state_dict(), os.path.join(self.save_folder, self.save_agent_file))
        torch.save(self.critic.state_dict(), os.path.join(self.save_folder, self.save_critic_file))
        # 保存 log metrics
        with open(self.save_log_file, 'w') as f:
            pickle.dump(log_metrics, f)
        # 删除 temp 文件
        for rt, dirs, files in os.walk(self.temp_folder):
            for name in files:
                remove_path = os.path.join(rt, name)
                os.remove(remove_path)

    def pretrain_critic(self):
        self.logger.info('Start pre-training critic')
        log_metrics = {
            'critic_loss': [],
            'global_reward': []
        }
        # 早停
        stop_rounds = 10
        for epoch in range(self.critic_pretrain_step):
            # self.logger.info('Start pretraining critic epoch {}'.format(epoch))
            # 从 episode 池中抽取一个 episode 作为当前 agent 的输入（出行需求）
            episode, episode_index = self.episode_pool.sample()
            with torch.no_grad():
                # 采样轨迹
                global_rewards, local_rewards, env_agent_list, agent_states, agent_info, simulate_grid_states, log_pi_taken = self.simulate_episode(episode)
            # 首先根据仿真返回的全局奖励来依据 TD loss 来训练 self.critic_params
            local_v, local_return, _, v_global, target_global_return, agent_global_weight, critic_loss = self.train_critic(global_rewards, local_rewards, env_agent_list, agent_states, agent_info, simulate_grid_states)
            log_metrics['global_reward'].append(np.mean(global_rewards))
            log_metrics['critic_loss'].append(critic_loss)
            # 记录日志
            if (epoch + 1) % self.learner_log_interval == 0 or self.debug:
                output_str = ''
                for metrics in log_metrics:
                    output_str += '{}: {:.4f}, '.format(metrics, np.mean(log_metrics[metrics]))
                self.logger.info('Critic Pretrain Epoch {}, the cumulative metrics are: {}'.format(epoch, output_str))
            if self.soft_update:
                self._soft_update_targets()
            elif self.critic_training_steps % self.target_update_interval == 0:
                self._update_targets()
            if epoch > 1:
                if log_metrics['critic_loss'][-1] > log_metrics['critic_loss'][-2] - 1e-3:
                    stop_rounds -= 1
                    if stop_rounds == 0:
                        self.logger.info('Early stop at epoch {}'.format(epoch))
                        break
                else:
                    stop_rounds = 10

    def retrain_critic(self, epochs, new_region=False):
        # 早停
        log_metrics = {
            'critic_loss': []
        }
        stop_rounds = 10
        for epoch in range(epochs):
            # 每次 agent 策略更新后，价值也应该更新，所以这里再训练 50 个 epoch 看看效果会不会提升
            # self.logger.info('Start pretraining critic epoch {}'.format(epoch))
            # 从 episode 池中抽取一个 episode 作为当前 agent 的输入（出行需求）
            episode, episode_index = self.episode_pool.sample()
            with torch.no_grad():
                # 采样轨迹
                global_rewards, local_rewards, env_agent_list, agent_states, agent_info, simulate_grid_states, log_pi_taken = self.simulate_episode(episode)
            # 首先根据仿真返回的全局奖励来依据 TD loss 来训练 self.critic_params
            local_v, local_return, _, v_global, target_global_return, agent_global_weight, critic_loss = self.train_critic(global_rewards, local_rewards, env_agent_list, agent_states, agent_info, simulate_grid_states, new_region=new_region)
            log_metrics['critic_loss'].append(critic_loss)
            # 记录日志
            if (epoch + 1) % self.learner_log_interval == 0 or self.debug:
                output_str = ''
                for metrics in log_metrics:
                    output_str += '{}: {:.4f}, '.format(metrics, np.mean(log_metrics[metrics]))
                self.logger.info('Critic Retrain Epoch {}, the cumulative metrics are: {}'.format(epoch, output_str))
            if self.soft_update:
                self._soft_update_targets()
            elif self.critic_training_steps % self.target_update_interval == 0:
                self._update_targets()
            if epoch > 1:
                if log_metrics['critic_loss'][-1] > log_metrics['critic_loss'][-2] - 1e-3:
                    stop_rounds -= 1
                    if stop_rounds == 0:
                        break
                else:
                    stop_rounds = 10
        return np.mean(log_metrics['critic_loss'])

    def simulate_episode(self, episode):
        """
        根据当前 agent 的输入，进行仿真，得到当前 episode 中 agents 的生成的轨迹集合
        Args:
            episode: (pandas.DataFrame) 当前 episode 的输入数据（出行需求）

        Returns:
            global_rewards: (list) 每一个时间步的外部奖励
            agent_states: (list) 每一个时间步 agent 隐层状态
            agent_info: (list) 每一个时间步 agent 的输出信息
            simulate_grid_states: (list) 每一个时间步城市网格热力状态，长度为 len(global_rewards) + 1
                                 因为 global_rewards 没有计算初始状态的 1
        """
        # self.logger.info('Start Simulating episode...')
        assert len(self.agent_manager.id2agent) == 0
        assert len(self.agent_manager.simulate_visit_id) == 0
        # 获取开始 time_code 与结束 time_code
        start_time_code = episode.iloc[0]['time_code']
        if self.debug:
            end_time_code = start_time_code + 3
        else:
            end_time_code = start_time_code // self.episode_size * self.episode_size + self.episode_size - 1
        data_index = 0
        # 创建结果变量
        global_rewards = []
        local_rewards = []
        env_agent_list = []
        agent_states = []
        agent_info = []
        simulate_grid_states = []
        log_pi_taken = []
        total_global_reward = 0
        # 开始仿真
        # 记住这里就应该是仿真到 end_time_code - 1 就结束了，因为到了 end_time_code 的时候 agent 做出选择，但是没有外部奖励了
        # 因为 episode 结束了
        for time_code in range(start_time_code, end_time_code):
            # 可能这个时间戳没有输入，但是环境中还是有可能有车在跑
            if data_index < episode.shape[0] and time_code == episode.iloc[data_index]['time_code']:
                # 有输入
                # 取出这一轮的仿真输入信息
                input_row = episode.iloc[data_index]
                agent_id_list = [int(x) for x in input_row['traj_id_list'].split(',')]
                road_id_list = [int(x) for x in input_row['road_id_list'].split(',')]
                des_id_list = [int(x) for x in input_row['des_id_list'].split(',')]
                data_index += 1
            else:
                # 没有输入，表示此时，城市中没有出租车在开
                # 但是这不代表，我们仿真里面没有车了
                agent_id_list = []
                road_id_list = []
                des_id_list = []
            env_agent_id_list, input_data = self.agent_manager.simulate_organize_input(current_time=time_code,
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
            batch_next_step_probs = []
            batch_next_step_idx_list = []
            batch_agent_state = []
            while start_input_index < input_data_len:
                next_input_index = start_input_index + self.batch_size
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
                batch_input_loc = torch.LongTensor(batch_input_loc).to(self.device)
                batch_input_time = torch.LongTensor(batch_input_time).to(self.device)
                batch_input_des = torch.LongTensor(batch_input_des).to(self.device)
                batch_input_prefer = torch.cat(batch_input_prefer, dim=0).to(self.device)
                batch_input_current_h = torch.cat(batch_input_current_h, dim=1).to(self.device)
                batch_input_current_c = torch.cat(batch_input_current_c, dim=1).to(self.device)
                # 输入策略网络
                candidate_prob, out_info, next_history_h, next_h, \
                    next_c, agent_state = self.agent_policy(batch_input_loc, batch_input_time, batch_input_des,
                                                            batch_input_inter_info, batch_input_prefer,
                                                            batch_input_candidate_mask, batch_input_history_h,
                                                            batch_input_current_h, batch_input_current_c)
                # 进行决策
                for i in range(len(candidate_prob)):
                    candidate_i_prob_soft = torch.softmax(candidate_prob[i], dim=0)
                    arg_max_candidate = candidate_i_prob_soft.argmax().item()
                    batch_next_step_list.append(batch_input_candidate_mask[i][arg_max_candidate])
                    batch_next_step_probs.append(candidate_i_prob_soft[arg_max_candidate].unsqueeze(0))
                    batch_next_step_idx_list.append(arg_max_candidate)
                # 开始下一个循环
                start_input_index = next_input_index
                batch_out_info.append(out_info)
                batch_next_history_h.extend(next_history_h)
                batch_next_h.extend(next_h)
                batch_next_c.extend(next_c)
                batch_agent_state.append(agent_state)
            # 计算 log_pi_taken
            agent_action_prob = torch.cat(batch_next_step_probs)
            agent_action_prob_log = torch.log(agent_action_prob)
            log_pi_taken.append(agent_action_prob_log)
            # 当前仿真步，所有 agent 都决策完了，那么可以开始更新每个 agent 的状态了
            batch_out_info = torch.cat(batch_out_info, dim=0)  # (agent_num, info_size)
            batch_agent_state = torch.cat(batch_agent_state, dim=0)  # (agent_num, hidden_size)
            # 计算当前仿真步之前的 grid_state
            current_grid_state = self.agent_manager.get_grid_freq()
            # 加入结果变量中
            simulate_grid_states.append(current_grid_state)
            # 判断是不是最后一个仿真步了
            is_finish = (time_code == end_time_code - 1)
            cur_local_reward, terminal_list = self.agent_manager.simulate_update(env_agent_id_list, time_code, batch_out_info, batch_next_history_h,
                                                                                 batch_next_h, batch_next_c, batch_next_step_list, max_step=100,
                                                                                 is_finish=is_finish)
            agent_states.append(batch_agent_state.detach())
            agent_info.append(batch_out_info.detach())
            # 计算当前仿真步所有 agent 决策后的 grid_state
            new_grid_state = self.agent_manager.get_grid_freq()
            if data_index < episode.shape[0] and time_code + 1 == episode.iloc[data_index]['time_code']:
                # 有车
                # 取出这一轮的仿真输入信息
                true_input_row = episode.iloc[data_index]
                true_road_id_list = [int(x) for x in true_input_row['road_id_list'].split(',')]
                true_grid_state = self.agent_manager.get_grid_freq(road_id_list=true_road_id_list)
            else:
                # 没有车
                true_grid_state = np.zeros((self.agent_manager.img_width, self.agent_manager.img_height))
            # 计算 external reward
            global_reward = self.agent_manager.calculate_external_reward(true_grid_state, new_grid_state)
            total_global_reward += global_reward
            if is_finish:
                # 调整为最后给一个累计奖励值
                global_rewards.append(total_global_reward)
            else:
                global_rewards.append(0)
            local_rewards.append(cur_local_reward)
            env_agent_list.append(env_agent_id_list)
        # 最后仿真步走完后的 current_grid_state 没有加入 simulate_grid_states 中
        current_grid_state = self.agent_manager.get_grid_freq()
        simulate_grid_states.append(current_grid_state)
        # 清空仿真器
        self.agent_manager.imitate_learning_reset()
        # 做一个 check 看是不是所有 agent 都被赋予了一个 local reward 了
        # local_rewards_np = np.concatenate(local_rewards)
        # no_zero_local_rewards = local_rewards_np[local_rewards_np != 0]
        # distinct_agent_id = np.unique(np.concatenate(env_agent_list))
        # assert len(distinct_agent_id) == len(no_zero_local_rewards)
        return global_rewards, local_rewards, env_agent_list, agent_states, agent_info, simulate_grid_states, log_pi_taken

    def train_critic(self, global_rewards, local_rewards, env_agent_list, agent_states, agent_info, simulate_grid_states, new_region=False):
        """
        训练 critic 网络，基于 global_reward 监督信号
        Args:
            global_rewards (list): 全局奖励
            agent_states (list of torch.tensor): 每一个时间步下的 Agents 的隐层状态
            agent_info (list): 每一个时间步下的 Agents 的输出信息
            simulate_grid_states (list of np.array): 每一个时间步下的城市网格状态

        Returns:
            local_v: critic 预测的每个 agent 局部价值，用于 agent_policy 的更新（不需要记梯度）
            local_return: 根据 TD lambda 算法计算的 local return
            critic_loss: Critic 的 TD loss 总和
        """
        # 需要将 simulate_grid_states 转换为 tensor
        simulate_grid_states = np.array(simulate_grid_states)
        simulate_grid_states = torch.FloatTensor(simulate_grid_states).to(self.device)
        # 先分别使用 target_critic 和 critic 计算目标网络的 v 值与当前网络的 v 值
        with torch.no_grad():
            target_v_local, target_v_global, _ = self.run_critic(agent_states, agent_info, simulate_grid_states, is_target=True)
        v_local, v_global, agent_global_weight = self.run_critic(agent_states, agent_info, simulate_grid_states, is_target=False)
        # 基于 TD 算法计算 return 值（使用目标网络估算）
        target_local_return, start_local_return = build_local_lambda_return(env_agent_list, target_v_local, global_rewards, local_rewards,
                                                                            self.local_gamma, self.td_lambda)
        target_global_return = td_lambda_return_forward_view(global_rewards, target_v_global, self.global_gamma, self.td_lambda)
        target_local_return_flatten = torch.cat(target_local_return)
        v_local_flatten = torch.cat(v_local)
        # 计算 TD loss
        local_td_loss = ((target_local_return_flatten.detach() - v_local_flatten) ** 2).mean()
        global_td_loss = ((target_global_return.detach() - v_global) ** 2).mean()
        if new_region:
            td_loss = local_td_loss
        else:
            td_loss = local_td_loss + global_td_loss
        self.critic_optimizer.zero_grad()
        td_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm_clip)
        self.critic_optimizer.step()
        self.critic_training_steps += 1
        return v_local, target_local_return, start_local_return, v_global, target_global_return, agent_global_weight, td_loss.cpu().item()

    def run_critic(self, agent_states, agent_info, simulate_grid_states, is_target=False):
        """
        运行 critic 输出预测的 v 值
        Args:
            agent_states (tensor):
            is_target:

        Returns:

        """
        episode_v_local = []
        episode_v_global = []
        episode_agent_weight = []
        for index, agent_state in enumerate(agent_states):
            if is_target:
                v_local, v_global, agent_weight = self.target_critic(agent_state, agent_info[index],  simulate_grid_states[:index+2])
            else:
                v_local, v_global, agent_weight = self.critic(agent_state, agent_info[index],  simulate_grid_states[:index+2])
            episode_v_local.append(v_local.squeeze(1))
            episode_v_global.append(v_global)
            episode_agent_weight.append(agent_weight)
        episode_v_global = torch.cat(episode_v_global)
        return episode_v_local, episode_v_global, episode_agent_weight

    def train_agent_policy(self, local_v, local_return, global_v, global_return, agent_global_weight, log_pi_taken, new_region=False):
        # 先把 v_mix 和 mix_return 这些全部转成一个一维 tensor
        local_v_flatten = torch.cat(local_v)  # (agent_size * episode_len) 这里的 agent_size 并不是一个固定的数
        local_return_flatten = torch.cat(local_return)
        log_pi_taken_flatten = torch.cat(log_pi_taken)
        local_advantages = (local_return_flatten - local_v_flatten.detach()).detach()
        # 归一化
        local_advantages = (local_advantages - local_advantages.mean()) / (local_advantages.std() + 1e-8)  # (agent_size * episode_len)
        if not new_region:
            # 计算 agent 每一步的全局优势
            global_advantages = (global_return - global_v.detach()).detach()  # (episode_len)
            # 归一化
            global_advantages = (global_advantages - global_advantages.mean()) / (global_advantages.std() + 1e-8)  # (episode_len)
            # 乘以 agent_global_weight
            weighted_global_advantages = []
            for i in range(len(agent_global_weight)):
                weighted_global_advantages.append(global_advantages[i] * agent_global_weight[i])
            weighted_global_advantages = torch.cat(weighted_global_advantages)  # (agent_size * episode_len)
            # 加和全局与局部优势
            advantages = local_advantages + self.global_alpha * weighted_global_advantages
        else:
            advantages = local_advantages
        loss = - (advantages.detach() * log_pi_taken_flatten).mean()
        self.agent_optimizer.zero_grad()
        loss.backward()
        # grads = torch.autograd.grad(loss, self.agent_policy.parameters(), create_graph=True, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.agent_policy.parameters(), self.grad_norm_clip)
        self.agent_optimizer.step()
        return loss

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.info("Updated target network")

    def _soft_update_targets(self):
        for param, param_target in zip(self.critic.parameters(), self.target_critic.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
