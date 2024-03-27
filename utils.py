import collections
import random
import numpy as np
NUM_STEP=500

class EpisodeMemory(object):
    def __init__(self,num_episode):
        self.num_episode=num_episode
        self.buffer = collections.deque(maxlen=num_episode)
        self.num_step=NUM_STEP   #时间步长

    def put(self,episode):
        self.buffer.append(episode)

    def sample(self,batch_size):

        index = random.sample(range(self.num_episode), batch_size)  #返回值是个列表
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        # for experience in mini_batch:
        #     self.num_step = min(self.num_step, len(experience)) #防止序列长度小于预定义长度

        for i in index:
            experience=self.buffer[i]
            idx = np.random.randint(0, len(experience)-self.num_step+1)  #随机选取一个时间步的id
            s, a, r, s_p, done = [],[],[],[],[]
            for i in range(idx,idx+self.num_step):
                e1,e2,e3,e4,e5=experience[i][0]
                s.append(e1),a.append(e2),r.append(e3),s_p.append(e4),done.append(e5)
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        #转换数据格式
        obs_batch=np.array(obs_batch).astype('float32')
        action_batch=np.array(action_batch).astype('float32')
        reward_batch=np.array(reward_batch).astype('float32')
        next_obs_batch=np.array(next_obs_batch).astype('float32')
        done_batch=np.array(done_batch).astype('float32')

        #将列表转换为数组并转换数据类型
        return obs_batch,action_batch,reward_batch,next_obs_batch,done_batch,index


class ReplayMemory(object):
    def __init__(self,e_rpm):
        #创建一个固定长度的队列作为缓冲区域，当队列满时，会自动删除最老的一条信息
        self.e_rpm=e_rpm
        self.buff=[]
    # 增加一条经验到经验池中
    def append(self,exp,done):
        self.buff.append([exp])
        #将一整个episode添加进经验池
        if(done):
            self.e_rpm.put(self.buff)
            self.buff=[]
    #输出队列的长度
    def __len__(self):
        return len(self.buff)