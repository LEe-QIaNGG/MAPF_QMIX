import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Training import MEMORY_CAPACITY
from Env import NUM_OBSTACLE,NUM_AGENTS,NUM_DIRECTIONS,OBSERVATION_SIZE
from MyQmix.utils import NUM_STEP

N_ACTIONS=NUM_DIRECTIONS    #动作空间大小
GAMMA=0.9         #衰减因子
LR=0.01           #学习率
TARGET_NETWORK_UPDATE_FREQ=3
#网络大小参数
HYPER_HIDDEN_DIM=40
QMIX_HIDDEN_DIM=40
RNN_HIDDEN_STATE=40
INPUT_SHAPE=4+2*OBSERVATION_SIZE    #CETE的state取id号，在加上id
STATE_SHAPE=INPUT_SHAPE*NUM_AGENTS
BATCH_SIZE = 16
GRAD_NORM_CLIP=10

device=torch.device('cuda:0')

class RNN(nn.Module):
    #输出q：(-1，N_ACTIONS)   h：(-1,rnn_hidden_dim)
    # 所有 Agent 共享同一网络, 因此 input_shape = obs_shape + n_actions + n_agents（one_hot_code）
    def __init__(self):
        super().__init__()
        self.rnn_hidden_dim=RNN_HIDDEN_STATE
        self.fc1 = nn.Linear(INPUT_SHAPE, self.rnn_hidden_dim)
        self.fc1.weight.data.normal_(0, 0.1)  # 初始化
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)     # GRUCell(input_size, hidden_size)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, N_ACTIONS)
        self.fc2.weight.data.normal_(0, 0.1)  # 初始化

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)                # GRUCell 的输入要求（current_input, last_hidden_state）
        q = self.fc2(h)                      # h 是这一时刻的隐状态，用于输到下一时刻的RNN网络中去，q 是真实行为Q值输出
        return q, h


class MIXNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 因为生成的 hyper_w1 需要是一个矩阵，而 pytorch 神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵

        # hyper_w1 网络用于输出推理网络中的第一层神经元所需的 weights，
        # 推理网络第一层需要 qmix_hidden * n_agents 个偏差值，因此 hyper_w1 网络输出维度为 qmix_hidden * n_agents
        self.hyper_w1 = nn.Sequential(nn.Linear(STATE_SHAPE, HYPER_HIDDEN_DIM),
                                      nn.ReLU(),
                                      nn.Linear(HYPER_HIDDEN_DIM, NUM_AGENTS * QMIX_HIDDEN_DIM))

        # hyper_w2 生成推理网络需要的从隐层到输出 Q 值的所有 weights，共 qmix_hidden 个
        self.hyper_w2 = nn.Sequential(nn.Linear(STATE_SHAPE, HYPER_HIDDEN_DIM),
                                      nn.ReLU(),
                                      nn.Linear(HYPER_HIDDEN_DIM, QMIX_HIDDEN_DIM))

        # hyper_b1 生成第一层网络对应维度的偏差 bias
        self.hyper_b1 = nn.Linear(STATE_SHAPE, QMIX_HIDDEN_DIM)
        # hyper_b2 生成对应从隐层到输出 Q 值层的 bias
        self.hyper_b2 =nn.Sequential(nn.Linear(STATE_SHAPE, QMIX_HIDDEN_DIM),
                                     nn.ReLU(),
                                     nn.Linear(QMIX_HIDDEN_DIM, 1)
                                     )

    def forward(self, q_values, states):  # states的shape为(episode_num, max_episode_len， state_shape)
        # 传入的q_values是三维的，shape为(episode_num, max_episode_len， n_agents)
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, NUM_AGENTS)  # (episode_num * max_episode_len, 1, n_agents)
        states = states.reshape(-1, STATE_SHAPE)  # (episode_num * max_episode_len, state_shape)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)

        w1 = w1.view(-1, NUM_AGENTS, QMIX_HIDDEN_DIM)
        b1 = b1.view(-1, 1, QMIX_HIDDEN_DIM)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)    # torch.bmm(a, b) 计算矩阵 a 和矩阵 b 相乘

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = w2.view(-1, QMIX_HIDDEN_DIM, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)
        return q_total
    
    

class QMIX(nn.Module):

    def __init__(self,load_checkpoint=False,path=None):
        super().__init__()
        self.eval_rnn, self.target_rnn = RNN().to(device), RNN().to(device)  # 初始化agent网络
        self.target_mix_net, self.eval_mix_net = MIXNet().to(device), MIXNet().to(device)
        self.eval_parameters = list(self.eval_mix_net.parameters()) + list(self.eval_rnn.parameters())
        self.optimizer=torch.optim.Adam(self.eval_parameters,lr=LR)
        self.loss_func=nn.MSELoss().to(device)
        self.loss = []
        self.episode=0

        if load_checkpoint:
            checkpoint = torch.load(path)
            self.eval_rnn.load_state_dict(checkpoint['eval_rnn_dict'])
            self.target_rnn.load_state_dict(checkpoint['target_rnn_dict'])
            self.eval_mix_net.load_state_dict(checkpoint['eval_qmix_dict'])
            self.target_mix_net.load_state_dict(checkpoint['target_qmix_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.eval_hidden=checkpoint['eval_hidden']
            self.target_hidden=checkpoint['target_hidden']
            self.episode = checkpoint['episode']
        else:
            #保存隐藏层参数
            self.eval_hidden=np.zeros((NUM_AGENTS,RNN_HIDDEN_STATE))
            self.target_hidden = np.zeros((NUM_AGENTS, RNN_HIDDEN_STATE))

    def init_hidden(self):
        self.eval_hidden= np.zeros((NUM_AGENTS, RNN_HIDDEN_STATE))
        self.target_hidden = np.zeros((NUM_AGENTS, RNN_HIDDEN_STATE))

    def choose_action(self, state, env,agent_id,epsilon):
        #返回数值

        if np.random.uniform() > epsilon:
            action=env.action_space.sample()
        else:
            hidden_state = self.eval_hidden[agent_id, :].reshape(-1,RNN_HIDDEN_STATE)
            hidden_state=torch.tensor(hidden_state, dtype=torch.float32).to(device)
            # 输入的state是完整state，要取出id号智能体的部分观察
            inputs = state[agent_id].copy()

            # transform the shape of inputs from (42,) to (1,42)
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(device)

            # inputs=torch.cat([torch.tensor(agent_id).reshape(1,1).to(device),inputs.reshape(1,-1)],dim=1)

            # avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

            # if self.args.cuda:
            #     inputs = inputs.cuda()
            #     hidden_state = hidden_state.cuda()

            # 计算Q值
            q_value,  h= self.eval_rnn(inputs, hidden_state)
            self.eval_hidden[agent_id]=h.cpu().detach().numpy()
            # choose action from q value
            # q_value[avail_actions == 0.0] = - float("inf")
            action = torch.argmax(q_value).cpu().detach().item()
        return action

    def get_q_value(self,batch_s,batch_s_):
        #输入  S: batch*NUM_STEP*NUM_AGENTS*INPUT_SHAPE
        #     H: batch*NUM AGENTS*HIDDEN　RIM    不能直接通过RNN得到Q值
        #输出  Q: batch*NUM STEP*NUM AGENTS* ACTION SPACE

        Q_EVALS,Q_TARGETS=[],[]
        for i_batch in range(BATCH_SIZE):
            Q_EVAL, Q_TARGET = [], []   #NUM STEP*NUM AGENTS*ACTION SPACE
            eval_hidden = torch.tensor(np.zeros([NUM_AGENTS, RNN_HIDDEN_STATE]),
                                       dtype=torch.float32).cuda()
            target_hidden = torch.tensor(np.zeros([NUM_AGENTS, RNN_HIDDEN_STATE]),
                                         dtype=torch.float32).cuda()
            for i in range(NUM_STEP):
                q_eval, eval_hidden = self.eval_rnn(batch_s[i_batch,i,:,:].reshape(-1, INPUT_SHAPE), eval_hidden)
                q_target, target_hidden = self.target_rnn(batch_s_[i_batch,i,:,:].reshape(-1, INPUT_SHAPE), target_hidden)
                Q_EVAL.append(q_eval)
                Q_TARGET.append(q_target)
            Q_EVAL=torch.stack(Q_EVAL,dim=0)
            Q_TARGET=torch.stack(Q_TARGET,dim=0)
            Q_TARGETS.append(Q_TARGET)
            Q_EVALS.append(Q_EVAL)
        Q_EVALS=torch.stack(Q_EVALS,dim=0)
        Q_TARGETS=torch.stack(Q_TARGETS,dim=0)

        # Q_EVAL, Q_TARGET = [], []  # NUM STEP*NUM AGENTS*ACTION SPACE
        # eval_hidden = torch.tensor(np.zeros([BATCH_SIZE*NUM_AGENTS, RNN_HIDDEN_STATE]),
        #                            dtype=torch.float32).cuda()
        # target_hidden = torch.tensor(np.zeros([BATCH_SIZE*NUM_AGENTS, RNN_HIDDEN_STATE]),
        #                              dtype=torch.float32).cuda()
        # for i in range(NUM_STEP):
        #     q_eval, eval_hidden = self.eval_rnn(batch_s[:, i, :, :].reshape(-1, INPUT_SHAPE), eval_hidden)
        #     q_target, target_hidden = self.target_rnn(batch_s_[:, i, :, :].reshape(-1, INPUT_SHAPE),
        #                                               target_hidden)
        #     Q_EVAL.append(q_eval)
        #     Q_TARGET.append(q_target)
        # Q_EVAL = torch.stack(Q_EVAL, dim=1)
        # Q_TARGET = torch.stack(Q_TARGET, dim=1)

        return Q_EVALS.cuda(),Q_TARGETS.cuda()


    def learn(self,buffer, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        batch_s, batch_action, batch_r, batch_s_,batch_done,padded_batch,sample_index,range_in_episode=buffer.sample(BATCH_SIZE)
        # #初始化rnn
        mask = 1 - padded_batch  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习

        batch_action = torch.tensor(batch_action, dtype=torch.int64).cuda()
        batch_s = torch.tensor(batch_s, dtype=torch.float32).cuda()
        batch_r = torch.tensor(batch_r, dtype=torch.float32).cuda()
        batch_s_ = torch.tensor(batch_s_, dtype=torch.float32).cuda()
        batch_done = torch.tensor(batch_done, dtype=torch.float32).cuda()
        mask=torch.tensor(mask,dtype=torch.float32).cuda()

        q_evals,q_targets=self.get_q_value(batch_s,batch_s_)

        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        q_evals = torch.gather(q_evals, dim=3, index=batch_action.reshape(BATCH_SIZE,NUM_STEP,NUM_AGENTS,1))

        # 得到target_q
        # q_targets[avail_u_next == 0.0] = - 9999999
        q_targets = q_targets.reshape(BATCH_SIZE,NUM_STEP,NUM_AGENTS,-1).max(dim=3)[0]

        q_total_eval = self.eval_mix_net(q_evals, batch_s)
        q_total_target = self.target_mix_net(q_targets.reshape(BATCH_SIZE,NUM_STEP,NUM_AGENTS,1), batch_s_)

        targets = batch_r + GAMMA * q_total_target.reshape(BATCH_SIZE,NUM_STEP)*mask

        # td_error = (q_total_eval.reshape(BATCH_SIZE,NUM_STEP)- targets.detach())
        # masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        # loss = (masked_td_error ** 2).sum() / mask.sum()

        loss=self.loss_func(q_total_eval.reshape(BATCH_SIZE,NUM_STEP), targets.detach())

        self.optimizer.zero_grad()
        loss.backward()

        #梯度截断
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, GRAD_NORM_CLIP)
        self.optimizer.step()

        if train_step > 0 and train_step % TARGET_NETWORK_UPDATE_FREQ == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
            print('loss: ', loss.item(), end='\n')
            self.loss.append(loss.item())

    def save_checkpoint(self,path):
        #保存模型，状态字典
        path="{}/checkpoint_QMIX_{}agent_{}obstacle_{}directions.pkl".format(path,NUM_AGENTS,NUM_OBSTACLE,NUM_DIRECTIONS)
        torch.save({'eval_rnn_dict':self.eval_rnn.state_dict(),
                    'target_rnn_dict':self.target_rnn.state_dict(),
                    'eval_qmix_dict':self.eval_mix_net.state_dict(),
                    'target_qmix_dict':self.target_mix_net.state_dict(),
                    'optimizer_state_dict':self.optimizer.state_dict(),
                    'eval_hidden':self.eval_hidden,
                    'target_hidden':self.target_hidden,
                    'episode':self.episode
                    },path)
        return