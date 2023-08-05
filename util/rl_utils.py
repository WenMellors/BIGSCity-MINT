import torch


def build_td_lambda_targets(external_rewards, env_agent_list, target_critic_mix_values, target_critic_ex_val,
                            critic_r_in, gamma, td_lambda, intrinsic_lambda):
    """
    由于我们 agent 数目不固定，所以这里的计算还挺复杂的。
    Args:
        external_rewards: 外部奖励值
        env_agent_list (list): 当前 episode 每一个时间步下的 agent id 列表。因为 agent 走到目的地后就会退出环境，
            所以 agent_size 是不定长的。
        target_critic_mix_values: 目标 critic 网络输出的 mix value (episode_len, agent_size)
        target_critic_ex_val (tensor): 目标 critic 网络输出的 ex value (episode_len)
        critic_r_in (list of tensor): 目标 critic 网络输出的 r_in (episode_len, agent_size) 感觉应该是目标网络才对呀
        gamma: 折现率
        td_lambda: TD lambda 的值
        intrinsic_lambda: intrinsic reward 与 external reward 相加的 lambda 权重

    Returns:
        mix_return: 混合奖励的回报值
        ex_return: 外部奖励的回报值
    """
    # 新建 return 变量
    res_mix_return = []  # 因为每一个时间步的 agent 数目是不一样的，所以这里我们只能做成 list 而不是一个 tensor
    # ex_return 可以做成 tensor
    ex_return = td_lambda_return_forward_view(external_rewards, target_critic_ex_val, gamma, td_lambda)
    agent_dict = {}  # 存储每个 agent_id 对应的 mix_value 与 mix_reward 值用于计算它的 mix_return
    for step in range(len(env_agent_list)):
        for index, agent_id in enumerate(env_agent_list[step]):
            if agent_id not in agent_dict:
                agent_dict[agent_id] = {'mix_reward': [], 'mix_value': [], 'iter': 0}
            # 获取当前 agent 的 mix_value
            cur_mix_value = target_critic_mix_values[step][index].cpu().item()  # type int
            agent_dict[agent_id]['mix_value'].append(cur_mix_value)
            # 计算当前 agent 的 mix_reward
            # 这里不能够把梯度消除，不然没法 update Intrinsic Reward 了
            cur_r_in = critic_r_in[step][index]
            cur_r_ex = external_rewards[step]
            cur_mix_reward = intrinsic_lambda * cur_r_in + cur_r_ex
            agent_dict[agent_id]['mix_reward'].append(cur_mix_reward)
    for agent_id in agent_dict:
        # 计算当前 agent 的 mix_return
        mix_return = td_lambda_return_forward_view(agent_dict[agent_id]['mix_reward'],
                                                   torch.FloatTensor(agent_dict[agent_id]['mix_value']), gamma,
                                                   td_lambda)
        agent_dict[agent_id]['mix_return'] = mix_return
    # 重新按照时间步，组织 agent 的 mix_return
    for step in range(len(env_agent_list)):
        cur_step_mix_return = target_critic_ex_val.new_zeros(len(env_agent_list[step]))
        for index, agent_id in enumerate(env_agent_list[step]):
            # 获取当前时间步该 agent 的 mix_return
            agent_iter = agent_dict[agent_id]['iter']
            cur_mix_return = agent_dict[agent_id]['mix_return'][agent_iter]
            cur_step_mix_return[index] = cur_mix_return
            # 迭代指针加 1
            agent_iter += 1
            agent_dict[agent_id]['iter'] = agent_iter
        # cat
        res_mix_return.append(cur_step_mix_return)
    return res_mix_return, ex_return


def build_local_lambda_return(env_agent_list, local_v, global_rewards, local_rewards, gamma, td_lambda, use_local=True):
    local_return = []
    agent_dict = {}  # 存储每个 agent_id 对应的 local_value 与 local_reward 值用于计算它的 local_return
    start_local_return = []  # 统计每个 agent 最开始的 return 用于输出，方便分析实验效果
    for step in range(len(env_agent_list)):
        for index, agent_id in enumerate(env_agent_list[step]):
            if agent_id not in agent_dict:
                agent_dict[agent_id] = {'local_reward': [], 'local_value': [], 'iter': 0}
            # 获取当前 agent 的 local_value
            cur_local_value = local_v[step][index].cpu().item()  # type int
            agent_dict[agent_id]['local_value'].append(cur_local_value)
            # 计算当前 agent 的 local_reward
            if use_local:
                cur_local_reward = local_rewards[step][index]
            else:
                cur_local_reward = global_rewards[step]
            agent_dict[agent_id]['local_reward'].append(cur_local_reward)
    for agent_id in agent_dict:
        # 计算当前 agent 的 local_return
        cur_local_return = td_lambda_return_forward_view(agent_dict[agent_id]['local_reward'],
                                                         torch.FloatTensor(agent_dict[agent_id]['local_value']),
                                                         gamma,
                                                         td_lambda)
        agent_dict[agent_id]['local_return'] = cur_local_return
        start_local_return.append(cur_local_return[0].cpu().item())
    # 重新按照时间步，组织 agent 的 mix_return
    for step in range(len(env_agent_list)):
        cur_step_local_return = local_v[0].new_zeros(len(env_agent_list[step]))
        for index, agent_id in enumerate(env_agent_list[step]):
            # 获取当前时间步该 agent 的 mix_return
            agent_iter = agent_dict[agent_id]['iter']
            cur_local_return = agent_dict[agent_id]['local_return'][agent_iter]
            cur_step_local_return[index] = cur_local_return
            # 迭代指针加 1
            agent_iter += 1
            agent_dict[agent_id]['iter'] = agent_iter
        # cat
        local_return.append(cur_step_local_return)
    return local_return, start_local_return


def td_lambda_return_forward_view(reward, value, gamma, td_lambda):
    """
    以 forward view 的方式计算 TD lambda 回报值
    Args:
        reward (list): 每一时间步的奖励值 (episode_len)
        value (tensor): 对应的 critic 预测的状态价值 (episode_len)
        gamma: 折现率
        td_lambda: lambda 值

    Returns:
        td_lambda_return: TD lambda 回报值 (episode_len - 1)
    """
    # 初始化 td_lambda_return
    td_lambda_return = value.new_zeros((value.shape[0] + 1))
    # 终止状态的价值与回报都是 0，所以终止状态前面状态的回报就是 reward
    td_lambda_return[-2] = reward[-1]
    for t in range(td_lambda_return.shape[0] - 3, -1, -1):
        td_lambda_return[t] = td_lambda * gamma * td_lambda_return[t+1] + reward[t] + \
                              (1 - td_lambda) * gamma * value[t + 1]
    # 最后一个终止状态的 return 是不要的
    return td_lambda_return[:-1]
