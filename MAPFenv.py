import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.distance import pdist,squreform
import time

ACTION_COST, IDLE_COST, GOAL_REWARD, COLLISION_REWARD,FINISH_REWARD,BLOCKING_COST = -0.3, -.5, 0.0, -2.,20.,-1.

class State():
    def __init__(self,num_agent,len_edge):
        self.num_agent=num_agent
        self.num_obstacle=100
        self.len_edge=len_edge
        self.step_len=1

        

        #起点，终点，障碍
        self.map=self.generate_map() #4*num_agent
        self.obstacle=self.generate_obstacle()  #num_obstacle*num_agent
        self.obs_agent=np.array((num_agent*num_agent))
        self.observation=[self.map,self.obstacle,self.obs_agent]

    def generate_map(self):
        pass

    def generate_obstacle(self):
        pass

    def move_agents(self,action):
        #更新位置
        self.map[0]=self.map[0]###################
        self.observation=[self.map,self.obstacle,self.obs_agent]
        return 
    
    def observe(self):
        squreform(pdist())
        return 
    
    def eval(self,id):
        reward=0
        reward=reward+self.collide(id)
        if self.finish(id):
            reward=reward+FINISH_REWARD
            id=-1
        return reward,id

    def collide(self,id):
        pos=self.observation[id,0:1]
        near_obstacle=self.observation[id,3:-self.num_agent]
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
        self.agent_id=range(self.num_agent)
        self.len_edge=len_edge
        #动作空间简化为离散的36个方向，加上静止动作
        self.action_space=spaces.Tuple([spaces.Discrete(self.num_agent), spaces.Discrete(37)])

        #观察空间为智能体的二维位置，即2*N的nparray
        # self.observation_space=spaces.Discrete(self.num_agent)

        #初始化地图
        self.state=State(self.num_agent,self.len_edge)

    def step(self,action):
        self.state.move_agent(action)

        Reward=0
        ID=[]
        
        for i in self.agent_id:
            r,id=self.state.eval(i)
            Reward=Reward+r
            ID.append(id)
        
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
    env=MAPFEnv(1,10)
    check_env(env)
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

