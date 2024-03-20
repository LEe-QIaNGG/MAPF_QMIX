import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Env import NUM_OBSTACLE,NUM_AGENTS,NUM_DIRECTIONS,OBSERVATION_SIZE
N_ACTIONS=NUM_DIRECTIONS*NUM_AGENTS    #动作空间大小
GAMMA=0.9 

device=torch.device('cuda:0')

class RNN(nn.Module):
    #输出q：(-1，N_ACTIONS)   h：(-1,rnn_hidden_dim)
    # 所有 Agent 共享同一网络, 因此 input_shape = obs_shape + n_actions + n_agents（one_hot_code）
    def __init__(self, input_shape, rnn_hidden_dim):
        super().__init__()
        self.rnn_hidden_dim=rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.fc1.weight.data.normal_(0, 0.1)  # 初始化
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)     # GRUCell(input_size, hidden_size)
        self.fc2 = nn.Linear(rnn_hidden_dim, N_ACTIONS)
        self.fc2.weight.data.normal_(0, 0.1)  # 初始化

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)                # GRUCell 的输入要求（current_input, last_hidden_state）
        q = self.fc2(h)                      # h 是这一时刻的隐状态，用于输到下一时刻的RNN网络中去，q 是真实行为Q值输出
        return q, h
    
    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

class MIXNet(nn.Module):
    def __init__(self, arglist):
        super().__init__()
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
    
    

class QMIX(nn.Module):

    def __init__(self, arglist):
        super().__init__()
        self.arglist = arglist
        self.memory_counter = 0  # 经验回放计数器
        self.eval_rnn,self.target_rnn=RNN().to(device),RNN().to(device)#初始化agent网络
        self.target_mix_net,self.eval_mix_net=MIXNet().to(device),MIXNet.to(device)
        self.optimizer=torch.optim.Adam()
        #保存隐藏层参数
        self.eval_hidden


    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, env,maven_z=None, evaluate=False):
        inputs = obs.copy()
        # avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # # transform agent_num to onehot vector
        # agent_id = np.zeros(self.n_agents)
        # agent_id[agent_num] = 1.

        # if self.args.last_action:
        #     inputs = np.hstack((inputs, last_action))
        # if self.args.reuse_network:
        #     inputs = np.hstack((inputs, agent_id))

        hidden_state = self.eval_hidden[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

        # if self.args.cuda:
        #     inputs = inputs.cuda()
        #     hidden_state = hidden_state.cuda()

        # 计算Q值
        q_value, self.eval_hidden[:, agent_num, :] = self.eval_rnn(inputs, hidden_state)

        # choose action from q value
        q_value[avail_actions == 0.0] = - float("inf")
        if np.random.uniform() < epsilon:
            action=env.action_space.sample()
        else:
            Index = torch.argmax(q_value)
            #动作空间的序号转动作向量
            action=tuple([int(Index/NUM_DIRECTIONS),int(Index%NUM_DIRECTIONS)])
        return action

    def store_transition(self, s, a, r, s_,ExperienceBuffer):
        #存储经验
        transition=np.concatenate([s.reshape(1,-1), np.array(a).reshape(1,-1),np.array(r).reshape(1,-1), s_.reshape(1,-1)],axis=1) #水平叠加这些矢向量
        #如果容量已满，则使用index将旧内存替换为新内存
        index=self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :]=transition
        self.memory_counter += 1
        return ExperienceBuffer

    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号

        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            # 把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数，max_episode_len，n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'], batch['avail_u'], batch['avail_u_next'], \
                                                             batch['terminated']
        mask = 1 - batch["padded"].float()  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习

        # 得到每个agent对应的Q值，维度为(episode个数，max_episode_len， n_agents，n_actions)
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        # 得到target_q
        q_targets[avail_u_next == 0.0] = - 9999999
        q_targets = q_targets.max(dim=3)[0]

        q_total_eval = self.eval_mix_net(q_evals, s)
        q_total_target = self.target_mix_net(q_targets, s_next)

        targets = r + GAMMA * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.optimizer.zero_grad()
        loss.backward()
        #梯度截断
        # torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())