import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):

    # 所有 Agent 共享同一网络, 因此 input_shape = obs_shape + n_actions + n_agents（one_hot_code）
    def __init__(self, input_shape, args):
        super().__init__()

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)     # GRUCell(input_size, hidden_size)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)                # GRUCell 的输入要求（current_input, last_hidden_state）
        q = self.fc2(h)                      # h 是这一时刻的隐状态，用于输到下一时刻的RNN网络中去，q 是真实行为Q值输出
        return q, h
    

class QMIXNet(nn.Module):

    def __init__(self, arglist):
        super().__init__()
        self.arglist = arglist

        # 因为生成的 hyper_w1 需要是一个矩阵，而 pytorch 神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵

        # hyper_w1 网络用于输出推理网络中的第一层神经元所需的 weights，
        # 推理网络第一层需要 qmix_hidden * n_agents 个偏差值，因此 hyper_w1 网络输出维度为 qmix_hidden * n_agents
        self.hyper_w1 = nn.Sequential(nn.Linear(arglist.state_shape, arglist.hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(arglist.hyper_hidden_dim, arglist.n_agents * arglist.qmix_hidden_dim))

        # hyper_w2 生成推理网络需要的从隐层到输出 Q 值的所有 weights，共 qmix_hidden 个
        self.hyper_w2 = nn.Sequential(nn.Linear(arglist.state_shape, arglist.hyper_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(arglist.hyper_hidden_dim, arglist.qmix_hidden_dim))

        # hyper_b1 生成第一层网络对应维度的偏差 bias
        self.hyper_b1 = nn.Linear(arglist.state_shape, arglist.qmix_hidden_dim)
        # hyper_b2 生成对应从隐层到输出 Q 值层的 bias
        self.hyper_b2 =nn.Sequential(nn.Linear(arglist.state_shape, arglist.qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(arglist.qmix_hidden_dim, 1)
                                     )

    def forward(self, q_values, states):  # states的shape为(episode_num, max_episode_len， state_shape)
        # 传入的q_values是三维的，shape为(episode_num, max_episode_len， n_agents)
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.arglist.n_agents)  # (episode_num * max_episode_len, 1, n_agents)
        states = states.reshape(-1, self.arglist.state_shape)  # (episode_num * max_episode_len, state_shape)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)

        w1 = w1.view(-1, self.arglist.n_agents, self.arglist.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.arglist.qmix_hidden_dim)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)    # torch.bmm(a, b) 计算矩阵 a 和矩阵 b 相乘

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = w2.view(-1, self.arglist.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)
        return q_total
    
    def learn(self, batch):

        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)

        # 把 batch 里的数据转化成 tensor
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)

        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'],  batch['avail_u'], batch['avail_u_next'],\
                                                             batch['terminated']

        # 得到每个agent对应的Q值列表
        q_evals, q_targets = self.get_q_values(batch)

        # 取出每个 agent 所选择动作的对应 Q 值
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        # 得到target_q，取所有行为中最大的 Q 值
        q_targets[avail_u_next == 0.0] = - 9999999      # 如果该行为不可选，则把该行为的Q值设为极小值，保证不会被选到
        q_targets = q_targets.max(dim=3)[0]

        # qmix更新过程，evaluate网络输入的是每个agent选出来的行为的q值，target网络输入的是每个agent最大的q值，和DQN更新方式一样
        q_total_eval = self.eval_qmix_net(q_evals, s)
        q_total_target = self.target_qmix_net(q_targets, s_next)

        targets = r + self.arglist.gamma * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.arglist.grad_norm_clip)
        self.optimizer.step()

        # 在指定周期更新 target network 的参数
        if train_step > 0 and train_step % self.arglist.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())