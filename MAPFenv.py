import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.distance import pdist,squareform
import time

ACTION_COST, IDLE_COST, GOAL_REWARD, COLLISION_REWARD,FINISH_REWARD,BLOCKING_COST = -0.3, -.5, 0.0, -2.,20.,-1.

class State():
    def __init__(self,num_agent,len_edge):
        self.num_agent=num_agent
        self.num_obstacle=2
        self.len_edge=len_edge
        self.step_len=1

        #起点，终点，障碍
        self.map=self.generate_map() #4*num_agent
        self.obstacle=self.generate_obstacle()  #num_obstacle*2
        self.obstacle_id=np.ones((num_agent*self.num_obstacle))#num_agent*num_obstacle
        self.obs_agent=np.ones((num_agent*num_agent))
        self.observation=np.concatenate((self.map,self.obstacle_id.reshape(self.num_agent,-1),self.obs_agent.reshape((self.num_agent,-1))),axis=1)

    def generate_map(self):
        self.map=np.random.rand(self.num_agent,4)*self.len_edge/2
        # print(self.map)
        return self.map

    def generate_obstacle(self):
        self.obstacle=np.random.rand(self.num_obstacle,2)*self.len_edge/2
        return self.obstacle

    def move_agents(self,action,agent_id):
        #更新位置
        Action=np.array(action)
        print(agent_id)
        x_change=np.cos(Action.take([agent_id])*np.pi/36).reshape(-1,1)
        y_cahnge=np.sin(Action.take([agent_id])*np.pi/36).reshape(-1,1)
        change=np.stack([x_change,y_cahnge]).reshape(-1,2)
        self.map[agent_id,0:2]=self.map[agent_id,0:2]+change
        self.observation=np.concatenate((self.map,self.obstacle_id.reshape(self.num_agent,-1),self.obs_agent.reshape((self.num_agent,-1))),axis=1)
        return 
    
    def observe(self):
        squareform(pdist())
        return 
    
    def eval(self,id):
        reward=0
        reward=reward+self.collide(id)
        if self.finish(id):
            reward=reward+FINISH_REWARD
            id=-1
        return reward,id

    def collide(self,id):
        pos=self.observation[id,0:2]
        print(self.obstacle_id,id)
        obstacle_mat=self.obstacle.take(np.argwhere(self.obstacle_id[id]==1))#障碍 n*2
        print(obstacle_mat)
        near_agent=self.observation[id,-self.num_agent:]
        pos_tot=np.stack(self.map[near_agent],self.obstacle[near_agent])
        pos_tot-pos#范数
        return 
    
    def finish(self,id):
        now=self.map[id,0:1]
        goal=now=self.map[id,2:3]
        if now==goal:
            return True
        else:
            return False
    
    def reset(self):
        self.map=self.generate_map() 
        self.obs_agent=np.array((self.num_agent*self.num_agent))
        self.observation=[self.map,self.obstacle,self.obs_agent]
        return


    


class MAPFEnv(gym.Env):
    def  __init__(self,num_agent,len_edge):
        self.num_agent=num_agent
        self.agent_id=list(range(self.num_agent))
        self.len_edge=len_edge
        #动作空间简化为离散的36个方向
        # self.action_space=spaces.Box(low=0,high=36,dtype=np.int64)
        self.action_space = spaces.Tuple([spaces.Discrete(self.num_agent), spaces.Discrete(36)])

        #观察空间为智能体的二维位置，即2*N的nparray
        # self.observation_space=spaces.Discrete(self.num_agent)

        #初始化地图
        self.state=State(self.num_agent,self.len_edge)

    def step(self,action):
        print('action: ',action)
        self.state.move_agents(action,self.agent_id)

        Reward=0
        ID=[]
        
        for i in self.agent_id:
            r,id=self.state.eval(i)
            Reward=Reward+r
            ID.append(id)
        self.agent_id=ID
        observation=self.state.observe()

        info={}

        if self.agent_id==[]:
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
    env=MAPFEnv(2,10)
    # check_env(env)
    #测试
    for epoch in range(5):
        for epoch in range(5):
            env.reset()
            print('Epoch', epoch+1, ': ',end='')
            print(env.state, end='')
            env.render()    # 刷新画面
            time.sleep(0.5)
            for i in range(5):
                env.step(env.action_space.sample())     # 随机选择一个动作执行
                print(' -> ', env.state, end='')
                env.render()    # 刷新画面
                time.sleep(0.5)
        print()
    env.close()

