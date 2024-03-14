import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.distance import pdist,squareform
import time

#奖励的参数
ACTION_COST, IDLE_COST, GOAL_REWARD, COLLISION_REWARD,FINISH_REWARD,BLOCKING_COST = -0.3, -.5, 0.0, -2.,20.,-1.
#碰撞的判定距离，智能体的视野距离,智能体的步长
COLLID_E,OBSERVE_DIST,STEP_LEN=0.1,2,0.8
NUM_OBSTACLE,NUM_AGENTS=3,3 #障碍物个数
MAP_WIDTH=10 #地图为正方形
NUM_DIRECTIONS=8


class State():
    def __init__(self,num_agent,len_edge):
        self.num_agent=num_agent
        self.num_obstacle=NUM_OBSTACLE
        self.len_edge=len_edge

        #起点，终点，障碍
        self.map=self.generate_map() #4*num_agent
        self.obstacle=self.generate_obstacle()  #num_obstacle*2
        self.obstacle_id=np.ones((num_agent,self.num_obstacle))#num_agent*num_obstacle

        #周围的智能体索引，对角线为零
        self.obs_agent=np.ones((num_agent,num_agent))
        np.fill_diagonal(self.obs_agent,0)

        self.observation=np.concatenate((self.map,self.obstacle_id.reshape(self.num_agent,-1),self.obs_agent.reshape((self.num_agent,-1))),axis=1)

    def generate_map(self):
        self.map=np.random.rand(self.num_agent,4)*self.len_edge/2
        # print(self.map)
        return self.map

    def generate_obstacle(self):
        self.obstacle=np.random.rand(self.num_obstacle,2)*self.len_edge/2
        return self.obstacle

    def move_agents(self,action):
        #更新位置
        # Action=np.array(action)
        # x_change=np.cos(Action.take([agent_id])*np.pi/NUM_DIRECTIONS).reshape(-1,1)
        # y_cahnge=np.sin(Action.take([agent_id])*np.pi/NUM_DIRECTIONS).reshape(-1,1)
        # change=STEP_LEN*np.stack([x_change,y_cahnge]).reshape(-1,2)
        # self.map[agent_id,0:2]=self.map[agent_id,0:2]+change
        # self.observation=np.concatenate((self.map,self.obstacle_id.reshape(self.num_agent,-1),self.obs_agent.reshape(self.num_agent,-1)),axis=1)
        
        pos=self.map[action[0],0:2]  #当前智能体的二维坐标
        change=STEP_LEN*np.stack([np.cos(action[1]*np.pi/NUM_DIRECTIONS),np.sin(action[1]*np.pi/NUM_DIRECTIONS)]).reshape(-1,2)
        self.map[action[0],0:2]=pos+change
        return 
    
    def observe(self,id):
        #通过简略计算更新id智能体周围智能体和障碍编号

        #所有智能体的二维位置
        pos=self.map[:,0:2]

        #得到N*N的曼哈顿距离矩阵
        dist_mat=squareform(pdist(pos,'cityblock'))

        #更新周围智能体索引矩阵
        self.obs_agent=np.where(dist_mat<OBSERVE_DIST,1,0)
        np.fill_diagonal(self.obs_agent, 0)

        #更新障碍索引
        pos_agent_mat=self.obstacle
        x=pos_agent_mat-pos[id]
        dist_array=pdist(pos_agent_mat-pos[id],'cityblock')
        self.obstacle_id[id]=np.where(dist_array<OBSERVE_DIST,1,0)
        return

    
    def eval(self,id):
        #计算id号智能体的reward，返回r和id，若id号智能体到达终点，返回-1
        reward=0
        reward=reward+self.collide(id)
        if self.finish(id):
            reward=reward+GOAL_REWARD
            id=-1
        return reward,id

    def collide(self,id):
        #返回id号智能体碰撞奖励
        #id号智能体当前二维位置
        pos=self.observation[id,0:2]

        # 障碍 n*2
        obstacle_mat=self.obstacle.take(np.argwhere(self.obstacle_id[id]==1),axis=0).reshape(-1,2)

        near_agent_index=self.observation[id,-self.num_agent:]
        #id号周围智能体二维位置
        agent_mat=self.map[:,0:2].take(np.argwhere(near_agent_index==1),axis=0).reshape(-1,2)

        #周围智能体和障碍的m*2位置矩阵
        pos_tot=np.concatenate((agent_mat,obstacle_mat),axis=0)

        r=COLLISION_REWARD*self.norm_two(pos_tot-pos,COLLID_E)
        return r
    
    def finish(self,id):
        now=self.map[id,0:2]
        goal=self.map[id,2:4]
        if (now==goal).all():
            return True
        else:
            return False
    
    def reset(self):
        self.map=self.generate_map()
        self.obstacle_id=np.ones((self.num_agent,self.num_obstacle))
        self.obs_agent = np.ones((self.num_agent, self.num_agent))
        np.fill_diagonal(self.obs_agent, 0)
        self.observation = np.concatenate((self.map, self.obstacle_id.reshape(self.num_agent, -1), self.obs_agent.reshape((self.num_agent, -1))),axis=1)
        return

    def norm_two(self,mat,e):
        ##小于e即为发生碰撞，返回碰撞个数
        #mat为m*2的矩阵
        dist=np.linalg.norm(mat,ord=2,axis=1)
        return np.sum(dist<e)

    def get_remaining_dist(self):
        #各智能体位置与终点坐标差
        mat=self.map[:,0:2]-self.map[:,2:4]
        return sum(np.linalg.norm(mat,ord=2,axis=1))


    


class MAPFEnv(gym.Env):
    def  __init__(self):
        self.num_agent=NUM_AGENTS
        self.agent_id=list(range(self.num_agent))
        self.len_edge=MAP_WIDTH

        #动作空间简化为离散的N个方向，每次只有一个智能体移动
        # self.action_space=spaces.Box(low=0,high=NUM_DIRECTIONS-1,shape=(1,self.num_agent),dtype=np.int64)
        self.action_space = spaces.Tuple([spaces.Discrete(self.num_agent), spaces.Discrete(NUM_DIRECTIONS)])

        #观察空间为智能体的二维位置，即2*N的nparray
        # self.observation_space=spaces.Discrete(self.num_agent)

        #初始化地图
        self.state=State(self.num_agent,self.len_edge)

    def step(self,action):
        #接受动作，返回状态，奖励，done，info，action为(agent_id,direction)
        if action[0] in self.agent_id:
            self.state.move_agents(action)
            #获取奖励，判断是否到终点
            Reward,id=self.state.eval(action[0])
            if id==-1:
                #智能体到达终点，活动智能体索引中删除该编号
                self.agent_id.remove(action[0])
            else:
                #没有到达终点的话，更新智能体周围智能体和障碍信息
                observation=self.state.observe(action[0])
        else:
            #该智能体已到终点，无需做动作
            Reward=GOAL_REWARD
        
        info={}
        observation=self.state.observation

        if self.agent_id==[]:
            Reward=Reward+FINISH_REWARD
            return observation,Reward,True,info

        return observation,Reward,False,info



    def reset(self):
        self.agent_id=range(self.num_agent)
        self.state.reset()
        return self.state.observation
    
    def render(self,mode='human'):
        pass
    
    def seed(self,seed=None):
        pass
    
    def close(self):
        pass


if __name__== "__main__":
    from stable_baselines3.common.env_checker import check_env 
    env=MAPFEnv()
    # print(env.action_space.sample())
    # # check_env(env)
    #测试
    for epoch in range(5):
        for epoch in range(5):
            env.reset()
            print('Epoch', epoch+1, ': ',end='\n')
            print('remaining distance:',env.state.get_remaining_dist(), end='')
            env.render()    # 刷新画面
            time.sleep(0.5)
            for i in range(5):
                _,r,_,_=env.step(env.action_space.sample())     # 随机选择一个动作执行
                print('remaining distance:', env.state.get_remaining_dist(),' Reward:',r, end='\n')
                env.render()    # 刷新画面
                time.sleep(0.5)
        print()
    env.close()

