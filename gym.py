import gym
import numpy as np
from gym import spaces

class State():
    def __init__(self,num_agent,len_edge):
        self.num_agent=num_agent
        self.len_edge=len_edge
        self.step_len=1

        #起点，终点，障碍
        self.map=generate_map() #3*len_edge*len_edge

    def generate_map():
        pass

    def move_agents(self,action,agent_id):
        #更新位置
        self.map[0]=self.map[0]############################################

        N_finish,agent_id=eval_finish(agent_id)

        #观察空间为智能体自身位置，智能体终点位置，周围智能体位置，周围智能体终点位置，周围障碍
        pos,goal,obs_pos,obs_goal,obs_obstacle=observe(self)

        return agent_id,[pos,goal,obs_pos,obs_goal,obs_obstacle]
    
    def observe(self):
        ###########################
        return pos,goal,obs_pos,obs_goal,obs_obstacle
    
    def eval_finish(agent_id):
        #########################
        return N_finish,agent_id




    

    

class MAPFEnv(gym.env):
    def  __init__(self,num_agent,len_edge):
        self.num_agent=num_agent
        self.agent_id=range(self.num_agent)
        #动作空间简化为离散的36个方向，加上静止动作
        self.action_space=spaces.Tuple([spaces.Discrete(self.num_agent), spaces.Discrete(37)])

        #观察空间为智能体的二维位置，即2*N的nparray
        self.observation_space=None

        #初始化地图
        self.state=State(self.num_agent,self.len_edge)

    def step(self,action):

        observation,self.agent_id=self.state.move_agents(action,self.agent_id)

        reward,done=eval(self.agent_id)

        info={}

        return observation,reward,done,info
    


    def eval(self,agent_id):


    
    
    def reset(self):
        return
    
    def render(self,mode='human'):
        pass
    
    def seed(self,seed=None):
        pass



if __name__== "__main__":
    from stable_baselines3.common.env_checker import check_env 
    env=MAPFEnv()
    check_env(env)
