import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Env import NUM_OBSTACLE,NUM_AGENTS,NUM_DIRECTIONS,OBSERVATION_SIZE

BATCH_SIZE=128    #从缓冲区采样过程的批大小
LR=0.01           #学习率
EPSILON=0.9       #epsilon greedy方法
GAMMA=0.9         #衰减因子
TARGET_NETWORK_REPLACE_FREQ=100       #目标网络更新的频率
N_STATES=NUM_AGENTS*(4+2*OBSERVATION_SIZE)   #状态空间大小
N_ACTIONS=NUM_DIRECTIONS*NUM_AGENTS    #动作空间大小

device=torch.device('cuda:0')

#目标网和训练网使用的网络
class Net(nn.Module):
    def __init__(self):
        #全连接网络，接受状态，输出动作空间中所有动作对应的Q值
        super(Net, self).__init__()
        #网络结构
        self.fc1=nn.Linear(N_STATES, 48)
        self.fc1.weight.data.normal_(0, 0.1) #初始化
        self.fc2=nn.Linear(48, 64)
        self.fc2.weight.data.normal_(0, 0.1) #初始化
        self.out=nn.Linear(64, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)
        
        
    def forward(self, x):
        x=self.fc1(x.reshape(-1,N_STATES))
        x=F.relu(x)
        x=self.fc2(x)
        x=F.leaky_relu(x)
        actions_value=self.out(x)
        return actions_value
        



class DQNet(object):
    def __init__(self,MEMORY_CAPACITY,load_checkpoint=False,PATH=None):
        self.MEMORY_CAPACITY=MEMORY_CAPACITY
        self.loss_func=nn.MSELoss().to(device)
        self.learn_step_counter = 0  # 学习次数计数器

        #目标网络和训练网络
        self.eval_net, self.target_net=Net().to(device), Net().to(device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)

        if load_checkpoint:
            checkpoint=torch.load(PATH)
            self.eval_net.load_state_dict(checkpoint['eval_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.memory_counter=checkpoint['memory_counter']
            self.memory=checkpoint['memory']

        else:
            self.memory_counter=0 #经验回放计数器
            #经验回放池 （s, a, r, s_）
            self.memory=np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2)) #r:1*1 a:1*1



        

    def  choose_action(self, x,env):
        #实现epsilon greedy方法，接受当前状态，环境，返回动作
        
        x=torch.unsqueeze(torch.FloatTensor(x), 0).to(device) #为输入状态x增加一个维度
        # x=torch.FloatTensor(x).to(device)
        #只输入一个样本
        if np.random.uniform() < EPSILON:   # 贪心
            #动作空间中每个动作的Q值，选最大
            actions_value=self.eval_net.forward(x)
            action=torch.max(actions_value, 1)[1].cpu().numpy()
        else:   #随机
            action=env.action_space.sample()
        return action
    
        
    def store_transition(self, s, a, r, s_):
        #存储经验
        transition=np.concatenate([s.reshape(1,-1), np.array(a).reshape(1,-1),np.array(r).reshape(1,-1), s_.reshape(1,-1)],axis=1) #水平叠加这些矢向量
        #如果容量已满，则使用index将旧内存替换为新内存
        index=self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :]=transition
        self.memory_counter += 1
        return
        
    
    def learn(self):
        #每固定步更新目标网络
        if self.learn_step_counter % TARGET_NETWORK_REPLACE_FREQ == 0:
            #将eval_net的参数赋值给target_net
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        
        #从缓冲区中确定采样批的索引
        sample_index=np.random.choice(self.MEMORY_CAPACITY, BATCH_SIZE) #随机采样
        #提取batch size大小的经验
        b_memory=self.memory[sample_index, :]
        #提取向量或矩阵s,a,r,s_，并将其转换为torch变量
        b_s=torch.FloatTensor(b_memory[:, :N_STATES]).to(device)
        #将数据放到GPU上
        b_a=torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).to(device)
        b_r=torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).to(device)
        b_s_=torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)

        # #动作转index， b_a：batch*2
        # Index=b_a[:,0]*NUM_DIRECTIONS+b_a[:,1]

        #Q函数计算
        q_eval=self.eval_net(b_s).gather(1, b_a.reshape(-1,1)) # Index:(batch_size, 1)      q_eval:(batch_size, 1)

        #下个状态的Q值
        q_next=self.target_net(b_s_).detach() 
        #选取最大的Q值

        q_target=b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1) # (batch_size, 1)
        loss=self.loss_func(q_eval, q_target)
        if self.learn_step_counter%4000==0:
            print('loss: ',loss)
        
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step() 
        return

    def save_checkpoint(self,path):
        #保存模型，状态字典
        path="{}/checkpoint_DQN_{}agent_{}obstacle_{}directions_{}.pkl".format(path,NUM_AGENTS,NUM_OBSTACLE,NUM_DIRECTIONS,self.learn_step_counter)
        torch.save({'eval_state_dict':self.eval_net.state_dict(),
                    'target_state_dict':self.target_net.state_dict(),
                    'optimizer_state_dict':self.optimizer.state_dict(),
                    'memory_counter':self.memory_counter,
                    'memory':self.memory
                    },path)
        return


if __name__== "__main__":
    pass